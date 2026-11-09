"""Two immutable VAL episodes in two cells, advanced by one simulated clock."""
import threading
import time
import numpy as np
from nbv import parts
from nbv.visualization.live_scene import Scene
from nbv.visualization.live import recorded_session
from nbv.visualization.playback import RecordedTimeline,SharedClock
from nbv.visualization.presentation import METHODS,COLORS,lights,camera,coverage_html,attribution_text,brief_credit,CoverageChart,load_audit


class RecordedController:
    def __init__(self,server,*,load_initial=True):
        self.server=server;self.scenes=[];self.episodes=[];self.clock=None;self.audit=load_audit()
        self.running=False;self.epoch=0;self.worker=False;self.worker_epoch=None;self.lock=threading.RLock();self.setting=False
        server.scene.set_up_direction('+z');lights(server)
        gui=server.gui;gui.configure_theme(show_share_button=False,control_width='large',brand_color=(0,114,178))
        gui.add_markdown('# NBV · tarama planlama\n**VAL · kayıtlı, SHA-256 doğrulanmış veri**')
        tabs=gui.add_tab_group()
        with tabs.add_tab('Oynat'):
            ids=[str(p['file_id']) for p in parts.load_yaml('configs/split_v1.yaml')['parts'] if p['split']=='val']
            self.part=gui.add_dropdown('Parça',ids,initial_value=ids[0])
            self.method=gui.add_dropdown('Yöntem',METHODS,initial_value='greedy_cost')
            self.start=gui.add_button('Başlat / Duraklat');self.next=gui.add_button('Adım');self.reset_button=gui.add_button('Sıfırla')
            self.speed=gui.add_slider('Hız (×)',.25,8.,.25,initial_value=4.)
            self.time_slider=gui.add_slider('Gösterim zamanı (s)',0.,1.,.01,initial_value=0.)
            self.show_candidates=gui.add_checkbox('48 aday pası göster',False)
            self.show_collision=gui.add_checkbox('Çarpışma modeli',False)
            gui.add_button_group('Kamera',('Genel','Parça','Yan')).on_click(lambda e:camera(server,e.target.value,comparison=self.comparison.value))
            self.coverage_bar=gui.add_html(coverage_html(0.));self.status=gui.add_markdown('Hazır.')
            self.credit_brief=gui.add_html('')
            self.chart=CoverageChart(gui)
            gui.add_markdown('Kayıtlı kapsama–zaman eğrileri; imleç iki hücrenin ortak gösterim zamanıdır. Eve dönüş ayrı hareket olarak korunur.')
        with tabs.add_tab('Karşılaştır'):
            self.comparison=gui.add_checkbox('İki hücreyi karşılaştır',False)
            self.other=gui.add_dropdown('İkinci yöntem',METHODS,initial_value='mcts')
            gui.add_markdown('Her iki hücre aynı simüle zamanla oynar. Kısa bölüm eve döndükten sonra son pozda bekler. Hücreler y yönünde 900 mm ayrıdır; bu ofset yalnızca gösterim içindir. Seed 0, varyant v0; yeni politika veya çözücü çağrısı yok.')
        with tabs.add_tab('Bilgi'):
            gui.add_markdown('Mavi: ölçüldü · turuncu: son pasta yeni · açık mavi: erişilebilir/ölçülmedi · gri: ölçülemez. Mesh nominal, noktalar varyant v0; sapma yaklaşık 1,5 mm’ye kadar. Ölçüm pas sonunda güncellenir. Geçişler yalnızca gösterim için eklem uzayında doğrusaldır. **Kırmızı: çarpışma modelinde saptanan gövde; yol düzeltilmez.**')
            self.visual_status=gui.add_markdown('');self.attribution=gui.add_markdown('')
        for control in (self.part,self.method,self.comparison,self.other):control.on_update(lambda _:self.reset())
        self.show_candidates.on_update(lambda _:self.toggles());self.show_collision.on_update(lambda _:self.toggles())
        self.start.on_click(lambda _:self.play());self.next.on_click(lambda _:self.advance_once());self.reset_button.on_click(lambda _:self.reset())
        self.time_slider.on_update(lambda _:self.seek(self.time_slider.value) if not self.setting else None)
        @server.on_client_connect
        def connected(_):camera(server,'Genel',comparison=self.comparison.value)
        if load_initial:self.reset()

    def toggles(self):
        for scene in self.scenes:
            scene.candidates.visible=self.show_candidates.value;scene.show_model=self.show_collision.value;scene.highlight(scene.hits)

    def reset(self):
        self.running=False;self.epoch+=1
        with self.lock:
            try:
                for scene in self.scenes:scene.root.remove()
                self.scenes=[];self.labels=[]
                bundle=recorded_session(int(self.part.value),self.method.value);data=bundle['part_data']
                choices=[self.method.value]+([self.other.value] if self.comparison.value else [])
                self.episodes=[next(e for e in data['episodes'] if e['method']==method) for method in choices]
                for index,episode in enumerate(self.episodes):
                    scene=Scene(self.server,data,bundle['common']['robot'],bundle=True,
                        prefix=f'/cell_{index}',offset=(0.,index*.9,0.))
                    self.scenes.append(scene)
                    self.labels.append(scene.api.add_label('/nbv/title',episode['method'],position=(0.,0.,.62),font_screen_scale=.7))
                self.clock=SharedClock([RecordedTimeline(e['steps']) for e in self.episodes])
                if self.audit:
                    for episode,track in zip(self.episodes,self.clock.tracks):
                        self.audit.validate_timeline(episode['source_run_id'],episode['variant'],track)
                self.setting=True;self.time_slider.max=self.clock.duration;self.time_slider.value=0.;self.setting=False
                if any(s.visual is None for s in self.scenes):self.show_collision.value=True
                self.toggles();self.chart.reset(data['episodes'],self.method.value)
                self.attribution.content=attribution_text(self.scenes[0].credit)
                self.credit_brief.content=brief_credit(self.scenes[0].credit)
                self.visual_status.content=self.scenes[0].visual_note
                self.render();camera(self.server,'Genel',comparison=self.comparison.value)
            except Exception as exc:
                self.clock=None;self.status.content=f'**HATA — oynatma durdu:** {type(exc).__name__}: {exc}'

    def render(self):
        if self.clock is None:return
        states=self.clock.states();messages=[]
        with self.server.atomic():
            for scene,episode,state,label in zip(self.scenes,self.episodes,states,self.labels):
                scans=[r for r in episode['steps'] if r['kind']=='scan']
                if len(scene.path)>state['completed']:scene.reset_measurements()
                while len(scene.path)<state['completed']:scene.complete(scans[len(scene.path)])
                scene.select(state['pass_index']);hits=self.audit.active(episode['source_run_id'],episode['variant'],state['elapsed_s']) if self.audit else []
                scene.highlight(hits);scene.draw_q(state['q'])
                label.text=f"{episode['method']}\n{state['elapsed_s']:.2f} s · {state['coverage']:.1%}"
                warning=(f"**denetlenmeyen geçiş: çarpışma ({max(h['penetration_mm'] for h in hits):.3f} mm)**" if hits else
                         'D-071: bu kayıt çarpışma aralıkları içeriyor.' if self.audit and self.audit.events[(episode['source_run_id'],episode['variant'])] else
                         'D-071 örneklenmiş denetim: bu anda temas yok.' if self.audit else 'D-071 ölçümü yüklenmedi.')
                verdict='SHA-256 ✓; '+episode['verification'].get('final_coverage_label_tr','kayıtlı doğrulama sonucu korunuyor.')
                messages.append(f"**{episode['method']}** · adım {state['completed']} · pas {state['pass_index'] if state['pass_index'] is not None else 'eve dönüş'}\n\n"
                    f"{state['elapsed_s']:.6f} s · kapsama {state['coverage']:.6%}\n\n{warning}\n\n{verdict}")
            self.coverage_bar.content=coverage_html(states[0]['coverage'])
            visual_warning=('**'+self.scenes[0].visual_note+'**\n\n') if self.scenes[0].visual is None else ''
            self.status.content=visual_warning+f"**VAL {self.part.value}_v0 · seed 0 · ortak zaman {self.clock.time:.6f} s**\n\n"+'\n\n---\n\n'.join(messages)
            self.setting=True;self.time_slider.value=self.clock.time;self.setting=False
            self.chart.update(self.clock.time)

    def seek(self,time_s):
        with self.lock:
            if self.clock:self.clock.seek(time_s);self.render()

    def advance_once(self):
        self.running=False
        with self.lock:
            if not self.clock:return
            points=[r['cumulative_time_s'] for e in self.episodes for r in e['steps'] if r['cumulative_time_s']>self.clock.time]
            self.clock.seek(min(points) if points else self.clock.duration);self.render()

    def play(self):
        self.running=not self.running
        if self.running and (not self.worker or self.worker_epoch!=self.epoch):
            self.worker=True;self.worker_epoch=self.epoch
            threading.Thread(target=self._work,args=(self.epoch,),daemon=True).start()

    def _work(self,epoch):
        previous=time.monotonic()
        try:
            while self.running and epoch==self.epoch:
                with self.lock:
                    if epoch!=self.epoch:break
                    if not self.clock:break
                    now=time.monotonic();target=self.clock.time+(now-previous)*self.speed.value;previous=now
                    # Visit detected interval boundaries even when a frame would skip a one-sample collision.
                    if self.audit:
                        boundaries=[t for e in self.episodes for h in self.audit.events[(e['source_run_id'],e['variant'])]
                            for t in (h['start_s'],h['end_s']) if self.clock.time<t<target]
                        if boundaries:target=min(boundaries)
                    self.clock.seek(target);self.render()
                    if self.clock.time>=self.clock.duration:self.running=False
                time.sleep(.08)
        except Exception as exc:
            self.running=False;self.status.content=f'**HATA:** {type(exc).__name__}: {exc}'
        finally:
            if self.worker_epoch==epoch:self.worker=False
