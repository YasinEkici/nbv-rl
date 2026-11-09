"""Recorded trajectories and audited intervals on one display-only simulated clock."""
from collections import defaultdict
import numpy as np
from nbv.visualization.live import allowed_variant
from nbv import parts


class AuditIndex:
    def __init__(self,report):
        if not report['complete']:raise ValueError('incomplete collision audit')
        manifest=parts.load_yaml('configs/split_v1.yaml')
        splits={p['file_id']:p['split'] for p in manifest['parts']}
        self.allowed={v['variant_id'] for v in manifest['variants'] if v['split']=='val'}
        self.events=defaultdict(list);self.raw_events=defaultdict(list)
        for event in report['events']:
            if splits.get(event['part']) not in ('train','val'):raise ValueError('closed test event in audit')
            if splits[event['part']]=='train':continue
            if event['variant'] not in self.allowed:raise ValueError('unknown VAL audit variant')
            self.raw_events[(event['source_run'],event['variant'])].append(event)
            a,b=event['time_s']
            for i in report['geometries'][event['geometry_id']]['intervals']:
                self.events[(event['source_run'],event['variant'])].append(i|dict(
                    start_s=a+(b-a)*i['start_fraction'],end_s=a+(b-a)*i['end_fraction'],
                    phase=event['phase'],step=event['step'],geometry_id=event['geometry_id']))
        self.covered=({(e['source_run'],e['variant']) for e in report['episodes'] if e['variant'] in self.allowed}
                      if 'episodes' in report else set(self.raw_events))

    def active(self,source_run,variant,time_s):
        if variant not in self.allowed:raise ValueError('closed test/non-VAL variant in collision overlay')
        if (source_run,variant) not in self.covered:raise ValueError('audit episode unavailable; collision state unknown')
        return [i for i in self.events[(source_run,variant)] if i['start_s']<=time_s<=i['end_s']]

    def validate_timeline(self,source_run,variant,track,*,partial=False):
        from nbv.collision_audit import segment_id
        from nbv.visualization.replay import require_exact
        if variant not in self.allowed:raise ValueError('closed test/non-VAL timeline')
        if (source_run,variant) not in self.covered:raise ValueError('audit episode unavailable; collision state unknown')
        segments={(s['phase'],s['step'],s.get('subsegment')):s for s in track.segments}
        maximum=max((s['step'] for s in track.segments),default=-1);checked=0
        for event in self.raw_events[(source_run,variant)]:
            key=(event['phase'],event['step'],event.get('subsegment'))
            if key not in segments:
                if partial and event['step']>maximum:continue
                raise ValueError('audited geometry segment missing from playback')
            s=segments[key]
            if segment_id(event['part'],s['start'],s['end'])!=event['geometry_id']:
                raise ValueError('audited geometry endpoints differ from playback')
            require_exact([s['start_s'],s['end_s']],event['time_s'],'overlay time')
            checked+=1
        return checked


class RecordedTimeline:
    def __init__(self,rows):
        self.rows=rows;self.segments=[];previous=0.;coverage=0.
        for index,row in enumerate(rows):
            finish=row['cumulative_time_s']
            if row['kind']=='scan':
                q=np.asarray(row['joint_samples_rad']);scan_start=finish-row['t_scan_s']
                self.segments.append(dict(start_s=previous,end_s=scan_start,start=np.asarray(row['joint_start_rad']),end=q[0],
                    phase='transfer',step=index,pass_index=row['pass_index'],coverage=coverage))
                for j in range(4):
                    self.segments.append(dict(start_s=scan_start+row['t_scan_s']*j/4,end_s=scan_start+row['t_scan_s']*(j+1)/4,
                        start=q[j],end=q[j+1],phase='intra_pass',step=index,subsegment=j,pass_index=row['pass_index'],coverage=coverage))
                coverage=row['primary_coverage']
            else:
                self.segments.append(dict(start_s=previous,end_s=finish,start=np.asarray(row['joint_start_rad']),
                    end=np.asarray(row['joint_end_rad']),phase='home',step=index,pass_index=None,coverage=coverage))
            previous=finish
        self.duration=previous

    def sample(self,time_s):
        if not self.segments:raise ValueError('empty recorded trajectory')
        local=min(max(float(time_s),0.),self.duration)
        segment=next((s for s in self.segments if local<s['end_s']),self.segments[-1])
        duration=segment['end_s']-segment['start_s'];fraction=np.clip((local-segment['start_s'])/duration,0.,1.) if duration else 1.
        q=segment['end'].copy() if fraction==1 else segment['start']+fraction*(segment['end']-segment['start'])
        completed=[r for r in self.rows if r['kind']=='scan' and r['cumulative_time_s']<=local]
        return dict(simulation_time_s=float(time_s),elapsed_s=local,q=q,coverage=completed[-1]['primary_coverage'] if completed else 0.,
            phase=segment['phase'],pass_index=segment['pass_index'],step=segment['step'],completed=len(completed),fraction=float(fraction))


class SharedClock:
    def __init__(self,tracks):self.tracks=tracks;self.time=0.;self.duration=max(t.duration for t in tracks)
    def seek(self,time_s):self.time=min(max(float(time_s),0.),self.duration)
    def states(self):return [t.sample(self.time) for t in self.tracks]
