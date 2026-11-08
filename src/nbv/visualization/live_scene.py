"""Viser presentation shared by live and immutable-bundle playback; all interpolation is display-only."""
import threading
import time
import numpy as np
import trimesh
import roboticstoolbox as rtb
from spatialmath import SE3
from scipy.spatial.transform import Rotation
from nbv import collision,parts
from nbv.visualization import assets
from nbv.visualization.demo import attribution,decode_array
from nbv.visualization.styles import MASK_STYLES

POINT_COLORS={'measured':(0,114,178),'new':(230,159,0),'available':(86,180,233),'unmeasurable':(150,150,150)}


def point_colors(covered,new,available):
    colors=np.tile(POINT_COLORS['unmeasurable'],(len(covered),1)).astype('u1')
    colors[available]=POINT_COLORS['available'];colors[covered]=POINT_COLORS['measured'];colors[new]=POINT_COLORS['new']
    return colors


def rgb(hex_color):return tuple(int(hex_color[i:i+2],16) for i in (1,3,5))


def placed_robot(description):
    dh=description['dh']
    links=[rtb.RevoluteDH(a=a,d=d,alpha=alpha,offset=offset) for a,d,alpha,offset in zip(dh['a_m'],dh['d_m'],dh['alpha_rad'],dh['offset_rad'])]
    model=rtb.DHRobot(links)
    model.base=SE3(np.asarray(description['base_xyz_mm'])/1000)*SE3.Rz(np.deg2rad(description['base_yaw_deg']))
    model.tool=SE3(0,0,description['tcp_offset_mm']/1000)
    return model


class Scene:
    def __init__(self,server,data,robot_info,*,bundle=False):
        self.server=server;self.api=server.scene;self.root=self.api.add_frame('/nbv',show_axes=False)
        self.robot=placed_robot(robot_info);self.home=np.asarray(robot_info['q_home_rad'])
        self.geom=collision.ArmGeometry.from_config(robot_info['collision'])
        self.sensor=parts.load_yaml('configs/sensor_v1.yaml')
        if bundle:
            self.points=decode_array(data['points']);self.vertices=decode_array(data['mesh']['vertices']);self.faces=decode_array(data['mesh']['triangles'])
            self.measured=np.unpackbits(decode_array(data['measured_bits']),axis=1,count=len(self.points),bitorder='big').astype(bool)
            p=data['passes'];self.valid=decode_array(p['valid_q']).astype(bool);self.reasons=decode_array(p['mask_reason'])
            self.starts=decode_array(p['sensor_start_mm']);self.ends=decode_array(p['sensor_end_mm']);self.credit=data['attribution']
        else:
            self.points=data['variant_points'];self.vertices=data['mesh_vertices'];self.faces=data['mesh_triangles'];self.measured=data['variant_measured']
            p=data['plan'];self.valid=p['reachable'].astype(bool);self.reasons=np.where(self.valid,0,np.where(p['ik_reachable'],2,1))
            midpoint=p['center']+p['standoff'][:,None]*p['direction'];half=p['length'][:,None]*p['sweep']/2
            self.starts,self.ends=midpoint-half,midpoint+half;self.credit=attribution(data['attribution'])
        self.available=self.measured[self.valid].any(0);self.covered=np.zeros(len(self.points),bool);self.path=[]
        self.api.add_box('/nbv/table',dimensions=(1.1,1.1,.015),position=(0,0,-.009),color=(195,195,195),opacity=.35)
        self.api.add_frame('/nbv/part-frame',axes_length=.07,axes_radius=.001)
        self.api.add_mesh_simple('/nbv/part',self.vertices/1000,self.faces,color=(175,185,195),opacity=.17,side='double')
        self.cloud=self.api.add_point_cloud('/nbv/samples',self.points/1000,point_colors(self.covered,self.covered,self.available),
            point_size=.0015,point_shape='circle',precision='float32')
        colors=np.array([rgb(MASK_STYLES[('reachable','ik_unreachable','collision_masked')[int(r)]]['color']) for r in self.reasons],dtype='u1')
        segments=np.stack((self.starts,self.ends),axis=1)/1000
        self.candidates=self.api.add_line_segments('/nbv/candidates',segments,np.repeat(colors[:,None,:],2,axis=1),thickness=.8,thickness_units='screen',visible=False)
        self.capsules={};self.draw_q(self.home)

    def draw_q(self,q):
        q=np.asarray(q,dtype=float)
        if q.shape!=(6,) or not np.isfinite(q).all():raise ValueError('finite six-joint animation state required')
        for name,r,a,b in collision.capsule_axes(self.robot,q,self.geom):
            a,b=a/1000,b/1000;direction=b-a;length=np.linalg.norm(direction)
            if name not in self.capsules:
                mesh=trimesh.creation.capsule(height=float(length),radius=r/1000,count=[8,8])
                self.capsules[name]=self.api.add_mesh_simple(f'/nbv/robot/{name}',mesh.vertices,mesh.faces,color=(115,125,135),opacity=.65)
            transform=trimesh.geometry.align_vectors([0,0,1],direction/length) if length>1e-12 else np.eye(4)
            handle=self.capsules[name];handle.position=(a+b)/2;handle.wxyz=Rotation.from_matrix(transform[:3,:3]).as_quat(scalar_first=True)
        T=self.robot.fkine(q).A;lo,hi=self.geom.box_lo/1000,self.geom.box_hi/1000
        self.api.add_box('/nbv/sensor',dimensions=hi-lo,position=T[:3,3]+T[:3,:3]@((lo+hi)/2),
            wxyz=Rotation.from_matrix(T[:3,:3]).as_quat(scalar_first=True),color=(40,45,50),opacity=.8)
        # A visual fan only: it does not create within-pass measurements or change coverage.
        sensor=self.sensor
        w,z=sensor['line_width_mm']/2000,sensor['working_distance_mm']/1000
        fan=np.array([[0,0,0],[-w,0,z],[w,0,z]])@T[:3,:3].T+T[:3,3]
        self.api.add_mesh_simple('/nbv/laser',fan,np.array([[0,1,2]],dtype='u4'),color=(230,159,0),opacity=.22,side='double')

    def complete(self,row):
        if row['kind']=='scan':
            action=row['pass_index']
            if not self.valid[action]:raise ValueError('refusing to animate masked pass')
            new=self.measured[action]&~self.covered;self.covered|=self.measured[action];self.path.append(action)
            self.api.add_line_segments(f'/nbv/path/{len(self.path)}',np.array([[self.starts[action],self.ends[action]]])/1000,
                np.array([[[230,159,0],[230,159,0]]],dtype='u1'),thickness=2,thickness_units='screen')
            self.api.add_label(f'/nbv/order/{len(self.path)}',str(len(self.path)),position=(self.starts[action]+self.ends[action])/2000,font_screen_scale=.65)
        else:new=np.zeros(len(self.points),bool)
        self.cloud.colors=point_colors(self.covered,new,self.available)
        self.draw_q(row['joint_end_rad'])


class DemoController:
    def __init__(self,server,*,recorded=False,load_initial=True):
        self.server=server;self.recorded=recorded;self.store=None;self.session=None;self.bundle=None;self.scene=None
        self.running=False;self.worker=False;self.epoch=0;self.lock=threading.RLock();self.position=0;self.home_done=False
        self.rows=[];self.last_time=0.;self.last_coverage=0.;self.last_pass='—';self.last_ms=0.;self.verdict=None
        server.scene.set_up_direction('+z')
        gui=server.gui
        gui.configure_theme(show_share_button=False,control_width='large')
        gui.add_markdown('# NBV — VAL\n'+('**Kayıtlı paket: SHA-256 doğrulanır.**' if recorded else '**Canlı politika / kayıtlı CAD planı.**'))
        allowed=[str(p['file_id']) for p in parts.load_yaml('configs/split_v1.yaml')['parts'] if p['split']=='val']
        self.part=gui.add_dropdown('Parça',allowed,initial_value=allowed[0])
        self.variant=gui.add_dropdown('Varyant',[str(i) for i in range(1 if recorded else 5)],initial_value='0')
        from nbv.visualization.live import METHODS
        self.method=gui.add_dropdown('Yöntem',METHODS,initial_value='greedy_cost')
        self.seed=gui.add_dropdown('Seed',['0'],initial_value='0')
        self.start=gui.add_button('Başlat / Duraklat');self.next=gui.add_button('Adım');self.reset_button=gui.add_button('Sıfırla')
        self.speed=gui.add_slider('Hız (×)',.25,8.,.25,initial_value=4.)
        self.show_candidates=gui.add_checkbox('48 aday pası göster',False)
        self.status=gui.add_markdown('Hazır. Bölüm henüz çalıştırılmadı.')
        gui.add_markdown('**Noktalar:** mavi = ölçüldü; turuncu = bu pasta yeni; açık mavi = erişilebilir/ölçülmedi; gri = ölçülemez.\n\n**Koordinatlar:** mesh nominal, noktalar seçilen varyant. Geçişler yalnızca gösterim için doğrusal; ölçüm pas sonunda güncellenir.')
        self.attribution=gui.add_markdown('')
        for control in (self.part,self.variant,self.method,self.seed):control.on_update(lambda event:self.reset())
        self.start.on_click(lambda event:self.play())
        self.next.on_click(lambda event:self.advance_once())
        self.reset_button.on_click(lambda event:self.reset())
        self.show_candidates.on_update(lambda event:self.toggle_candidates())
        @server.on_client_connect
        def camera(client):
            client.camera.up_direction=(0,0,1);client.camera.position=(.65,-.85,.65);client.camera.look_at=(-.12,0,.16)
        if load_initial:self.reset()

    def toggle_candidates(self):
        if self.scene:self.scene.candidates.visible=self.show_candidates.value

    def reset(self):
        from nbv.visualization.live import SourceStore,LiveSession,recorded_session
        self.running=False;self.epoch+=1
        with self.lock:
            self.status.content='Hazırlanıyor; henüz doğrulama sonucu yok.'
            try:
                method=self.method.value
                seeds=['0'] if self.recorded or method in ('greedy_cost','ilp_tsp','joint_opt') else [str(i) for i in range(10)]
                if list(self.seed.options)!=seeds:
                    self.seed.options=seeds
                    if self.seed.value not in seeds:self.seed.value='0'
                part,index,seed=int(self.part.value),int(self.variant.value),int(self.seed.value)
                if self.scene:self.scene.root.remove()
                self.position=0;self.rows=[];self.last_time=0.;self.last_coverage=0.;self.last_pass='—';self.last_ms=0.;self.home_done=False;self.verdict=None
                if self.recorded:
                    self.bundle=recorded_session(part,method);self.session=None
                    self.scene=Scene(self.server,self.bundle['part_data'],self.bundle['common']['robot'],bundle=True)
                else:
                    if self.store is None:self.store=SourceStore()
                    self.session=LiveSession(self.store,part,index,method,seed);self.bundle=None
                    self.scene=Scene(self.server,assets.load_scene(part,variant=True,variant_index=index),assets.robot_description())
                self.toggle_candidates()
                a=self.scene.credit
                self.attribution.content=(f'### Veri atfı\n**{a["name"]}** — {a["author"]}\n\n[Thingi10K / kaynak]({a["source_url"]})\n\n{a["license"]}; **sürüm kayıtlı değil**. [CC genel bakış]({a["official_license_overview"]}) (sürüme özel lisans değil).\n\nDeğişiklikler: 200 mm ölçekleme, alan ağırlıklı örnekler, varyant sapmaları ve türetilmiş görünürlük; sadeleştirme yok. BY-SA türevleri aynı lisansla.')
                self.update_status()
            except Exception as exc:
                self.session=None;self.bundle=None;self.status.content=f'**HATA — oynatma durduruldu:** {type(exc).__name__}: {exc}'

    def update_status(self):
        if self.verdict is None:verification='Henüz tamamlanmadı; eşleşme iddiası yok.'
        elif self.verdict.get('exact'):verification='Kayıtla birebir ✓' if not self.recorded else 'SHA-256 doğrulandı ✓; kayıtlı doğrulama sonucu korunuyor.'
        else:verification='**UYUŞMAZLIK:** '+', '.join(self.verdict.get('mismatches',[]))
        mode='Kayıtlı paket' if self.recorded else 'Kayıtlı plan' if self.method.value in ('ilp_tsp','joint_opt') else 'Canlı politika; ε = 0'
        timing='Kayıtlı oynatma; canlı karar ölçümü yok.' if self.recorded or self.method.value in ('ilp_tsp','joint_opt') else f'{self.last_ms:.3f} ms — canlı, bu makine, görüntüleme yükü dahil; animasyon beklemesi hariç. D-066 ile karşılaştırılmaz.'
        self.status.content=(f'**{mode}** · VAL {self.part.value}_v{self.variant.value} · {self.method.value} · seed {self.seed.value}\n\n'
            f'Adım: **{self.position}** · Pas: **{self.last_pass}**\n\nBirikmiş süre: **{self.last_time:.6f} s**\n\nBirincil kapsama: **{self.last_coverage:.6%}** (erişilebilir birleşim)\n\nSon karar/adım: {timing}\n\n**Doğrulama:** {verification}\n\nKapsama yeniden oynatmadan/canlı ENV adımından türetildi; kayıtlı kısmi oracle ile tutarlılık bölüm sonunda kontrol edilir.')

    def play(self):
        self.running=not self.running
        if self.running:self._launch(True)

    def advance_once(self):
        if not self.running:self._launch(False)

    def _launch(self,auto):
        if self.worker:return
        self.worker=True
        threading.Thread(target=self._work,args=(auto,self.epoch),daemon=True).start()

    def _next_row(self):
        if self.recorded:
            if self.position>=len(self.bundle['steps']):return None
            return self.bundle['steps'][self.position]
        if self.session is None:return None
        if not self.session.done:
            row=self.session.step()
            if row is not None:return row
        if not self.home_done and self.session.steps:
            self.home_done=True;s=self.session.steps[-1]
            return dict(kind='home',pass_index=None,t_move_s=0.,t_scan_s=0.,t_home_s=s['t_home'],
                cumulative_time_s=self.session.summary['cycle_time'],primary_coverage=s['coverage'],
                joint_start_rad=self.session.joints[-1]['q_end'],joint_end_rad=self.scene.home.tolist())
        return None

    def _animate(self,row,epoch):
        stages=[]
        if row['kind']=='scan':
            q=np.asarray(row['joint_samples_rad'])
            stages.append((np.asarray(row['joint_start_rad']),q[0],row['t_move_s']))
            stages.extend((a,b,row['t_scan_s']/(len(q)-1)) for a,b in zip(q[:-1],q[1:]))
        else:stages.append((np.asarray(row['joint_start_rad']),np.asarray(row['joint_end_rad']),row['t_home_s']))
        render_s=0.
        for start,end,duration in stages:
            seconds=duration/float(self.speed.value);frames=max(1,int(seconds*12));began=time.monotonic()
            for i in range(1,frames+1):
                if epoch!=self.epoch:return None
                before=time.perf_counter();self.scene.draw_q(start+(end-start)*(i/frames));render_s+=time.perf_counter()-before
                wait=began+seconds*i/frames-time.monotonic()
                if wait>0:time.sleep(min(wait,.1))
        return render_s

    def _work(self,auto,epoch):
        try:
            while epoch==self.epoch:
                with self.lock:
                    start=time.perf_counter();row=self._next_row();decision_s=time.perf_counter()-start
                    if row is None:
                        self.running=False;break
                    render_s=self._animate(row,epoch)
                    if render_s is None:break
                    before=time.perf_counter();self.scene.complete(row);render_s+=time.perf_counter()-before
                    self.last_ms=(decision_s+render_s)*1000;self.rows.append(row);self.position+=1
                    self.last_time=row['cumulative_time_s'];self.last_coverage=row['primary_coverage'];self.last_pass='eve dönüş' if row['kind']=='home' else row['pass_index']
                    if row['kind']=='home':
                        self.verdict=self.bundle['verification'] if self.recorded else self.session.verification
                        self.running=False
                    self.update_status()
                if not auto or not self.running:break
        except Exception as exc:
            self.running=False;self.status.content=f'**HATA / doğrulama tamamlanmadı:** {type(exc).__name__}: {exc}'
        finally:self.worker=False
