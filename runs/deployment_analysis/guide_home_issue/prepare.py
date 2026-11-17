"""Build a local, shareable issue draft and flight evidence; no network or hardware."""
import os
os.environ['MPLCONFIGDIR']='/tmp/guide-home-matplotlib'
import json,csv,zipfile
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(__file__).resolve().parents[3];out=Path(__file__).resolve().parent
names=['deploy_20261117T100017.801841Z','deploy_20261117T100827.697140Z']
metrics=[];fig,axes=plt.subplots(2,2,figsize=(13,10))
for col,name in enumerate(names):
 path=root/'runs'/name;es=[json.loads(l) for l in (path/'events.jsonl').read_text().splitlines()]
 obs=[e for e in es if e['kind']=='observation'];zero=obs[0]['monotonic_ns'];tm=lambda e:(e['monotonic_ns']-zero)/1e9
 end=next(e for e in es if e['kind']=='race_end');home=next(e for e in es if e['kind']=='go_to_sent');commands=[e for e in es if e['kind']=='attitude_command_sent'];terminal=[e for e in es if e['kind']=='transition'][-1]
 poses=[e for e in es if e['kind']=='external_pose_sent'];t=np.array([tm(e) for e in poses]);xyz=np.array([e['pos'] for e in poses]);tilt=np.degrees(np.arccos(np.clip(R.from_quat([e['quat'] for e in poses]).as_matrix()[:,2,2],-1,1)))
 ft=tm(end);ht=tm(home);interval=(t>=ht)&(t<=ht+3);mi=np.flatnonzero(interval)[xyz[interval,2].argmin()];breach=np.flatnonzero((t>ht)&(xyz[:,1]>1.5))[0]
 reset=next(e for e in es if e['kind']=='reset');metadata=json.loads((path/'metadata.json').read_text())
 states={'run':name,'terminal_transition_time_s':tm(terminal),'terminal_observation':terminal['next_obs'],'last_policy_command':[e for e in es if e['kind']=='action'][-1], 'race_end':end,'first_go_to':home,'first_go_to_time_s':ht,'race_end_time_s':ft,'first_go_to_delay_after_race_end_ms':(home['monotonic_ns']-end['monotonic_ns'])/1e6,'policy_commands_after_first_go_to':sum(e['monotonic_ns']>home['monotonic_ns'] for e in commands),'near_floor_time_after_race_end_s':float(t[mi]-ft),'near_floor_time_after_first_go_to_s':float(t[mi]-ht),'near_floor_pose':poses[mi],'first_y_breach_time_after_first_go_to_s':float(t[breach]-ht),'first_y_breach_pose':poses[breach],'max_y_first_3s_return_m':float(xyz[interval,1].max()),'track':reset['track'],'config':metadata['config'],'code':metadata['code'],'policy_identity':next(e for e in es if e['kind']=='policy_metadata')}
 metrics.append(states)
 with (out/f'{name}_trajectory.csv').open('w',newline='') as f:
  w=csv.writer(f, lineterminator="\n");w.writerow(['race_time_s','time_since_first_return_command_s','phase','x_m','y_m','z_m','qx','qy','qz','qw','tilt_from_vertical_deg'])
  for j,e in enumerate(poses):
   phase='preflight' if t[j]<0 else 'policy' if t[j]<ft else 'return_and_landing'
   w.writerow([t[j],t[j]-ht,phase,*e['pos'],*e['quat'],tilt[j]])
 ax=axes[0,col];ax.plot(t-ht,xyz[:,2],label='Mocap z');ax.axvline(0,color='C3',label='First return command');ax.scatter(t[mi]-ht,xyz[mi,2],color='k',label='Unplanned near-floor dip');ax.axhline(xyz[0,2],color='gray',ls=':',label='Preflight resting z');land=[e for e in es if e['kind']=='go_to_sent'][-1];ax.axvline(tm(land)-ht,color='C2',ls='--',label='Planned landing command');ax.set(title=name,xlabel='Seconds relative to first return command',ylabel='Mocap height (m)',xlim=(-1,12));ax.legend(fontsize=8)
 ax=axes[1,col];ax.plot(xyz[t<ht,0],xyz[t<ht,1],color='gray',alpha=.6,label='Policy flight');ax.plot(xyz[interval,0],xyz[interval,1],color='C3',label='First 3 s of return');ax.scatter(home['pos'][0],home['pos'][1],marker='*',s=160,color='C1',label='Requested braking target');ax.scatter(xyz[mi,0],xyz[mi,1],marker='x',color='black',s=70,label='Unplanned near-floor dip');bounds=metadata['config']['env']['track']['safety_limits'];lo,hi=bounds['pos_limit_low'],bounds['pos_limit_high'];ax.plot([lo[0],hi[0],hi[0],lo[0],lo[0]],[lo[1],lo[1],hi[1],hi[1],lo[1]],'k--',lw=1,label='Configured bounds');ax.set(xlabel='x (m)',ylabel='y (m)',aspect='equal');ax.legend(fontsize=8)
for ax in axes.flat:ax.grid(alpha=.25)
fig.tight_layout();fig.savefig(out/'return_home_evidence.png',dpi=150);fig.savefig(out/'return_home_evidence.pdf')
(out/'terminal_states_and_events.json').write_text(json.dumps(metrics,indent=2)+'\n')
