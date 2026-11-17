"""CPU-only replay of the flight with obstacle 1's height corrected."""
import os
os.environ['JAX_PLATFORMS']='cpu'
os.environ['SCIPY_ARRAY_API']='1'
os.environ['MPLCONFIGDIR']='/tmp/drone-analysis-matplotlib'
import json,hashlib
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
from scipy.spatial.transform import Rotation as R
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from lsy_drone_racing.control.rl_race.controller import load_policy,policy_mean_action
from lsy_drone_racing.control.rl_race.env_wrapper.observations import encode,ObsConfig
from lsy_drone_racing.control.rl_race.env_wrapper.actions import to_command,ActionConfig
root=Path(__file__).resolve().parents[2];out=Path(__file__).resolve().parent
p=root/'runs/deploy_20261117T113735.863815Z';es=[json.loads(l) for l in (p/'events.jsonl').read_text().splitlines()];oo=[e for e in es if e['kind']=='observation'];aa={e['step']:e['action'] for e in es if e['kind']=='action'};pp={e['step']:e for e in es if e['kind']=='policy'}
model=root/'lsy_drone_racing/control/rl_race/deploy_model';bundle,params=load_policy(model);params=jax.tree.map(jnp.asarray,params)
for f in next(e for e in es if e['kind']=='policy_metadata')['model_files']:assert hashlib.sha256((model/Path(f['path']).name).read_bytes()).hexdigest()==f['sha256']
@jax.jit
def predict(x):return to_command(policy_mean_action(params,x,bundle['network']['activation']),ActionConfig(**bundle['action']))
f=np.array([pp[i]['features'] for i in range(len(oo))]);held=f.copy();height=[];gate=[];orders=[]
meta=json.loads((p/'metadata.json').read_text());nominal=meta['config']['env']['track']['gates'][2]
for i,e in enumerate(oo):
 s=e['obs'];order=np.argsort(np.linalg.norm(np.array(s['obstacles_pos'])[:,:2]-s['pos'][:2],axis=1));orders.append(order)
 if i:
  slots=[list(order).index(k) for k in orders[i-1]];held[i,-12:]=f[i,-12:].reshape(4,3)[slots].reshape(-1)
 obs={k:jnp.asarray(v)[None] for k,v in s.items()};prev=jnp.asarray(pp[i]['previous_action'])[None]
 obs['obstacles_pos']=obs['obstacles_pos'].at[0,0,2].set(1.55)
 height.append(np.asarray(encode(obs,prev,ObsConfig(**bundle['obs']))[0]))
 obs={k:jnp.asarray(v)[None] for k,v in s.items()};obs['gates_pos']=obs['gates_pos'].at[0,2].set(jnp.array(nominal['pos']));obs['gates_quat']=obs['gates_quat'].at[0,2].set(jnp.array(R.from_euler('xyz',nominal['rpy']).as_quat()))
 gate.append(np.asarray(encode(obs,prev,ObsConfig(**bundle['obs']))[0]))
actual=np.array([aa[i] for i in range(len(oo))]);replayed=np.asarray(predict(jnp.asarray(f)));hold_pred=np.asarray(predict(jnp.asarray(held)));height_pred=np.asarray(predict(jnp.asarray(np.array(height))));gate_pred=np.asarray(predict(jnp.asarray(np.array(gate))));error=float(np.max(np.abs(actual-replayed)));assert error<1e-4
zero=oo[0]['monotonic_ns'];t=np.array([(e['monotonic_ns']-zero)/1e9 for e in oo]);pos=np.array([e['obs']['pos'] for e in oo]);raw=[e for e in es if e['kind']=='external_pose_sent'];rt=np.array([(e['monotonic_ns']-zero)/1e9 for e in raw]);rp=np.array([e['pos'] for e in raw]);mask=(rt>1)&(rt<4);idx=np.flatnonzero(mask)[rp[mask,2].argmin()]
metrics={'run':p.name,'factual_replay_max_error':error,'minimum_mocap_z_after_1s':float(rp[idx,2]),'minimum_time_s':float(rt[idx]),'selected_steps':[]}
for i in [29,30,104,105,106,107,108,109,110,111]:
 metrics['selected_steps'].append({'step':i,'time_s':float(t[i]),'actual_command':actual[i].tolist(),'same_state_previous_obstacle_order_command':hold_pred[i].tolist(),'same_state_obstacle1_height_1_55_command':height_pred[i].tolist(),'same_state_gate3_nominal_pose_command':gate_pred[i].tolist(),'obstacle_order_zero_based':orders[i].tolist(),'position':pos[i].tolist(),'gates_visited':oo[i]['obs']['gates_visited']})
(out/'flight_113735_metrics.json').write_text(json.dumps(metrics,indent=2)+'\n');print(json.dumps(metrics,indent=2))
fig,ax=plt.subplots(3,2,figsize=(14,11))
ax[0,0].plot(t,[e['obs']['obstacles_pos'][0][2] for e in oo]);ax[0,0].set(xlim=(0,3),ylabel='Obstacle 1 input z (m)',title='Corrected obstacle height: 1.55 -> 1.584 m',xlabel='Race time (s)')
ax[0,1].plot(rt,rp[:,2],label='Mocap height');ax[0,1].axhline(.6603,color='gray',ls=':',label='Gate 3 center');ax[0,1].axvline(t[107],color='C3',ls='--',label='Obstacle-order swap');ax[0,1].set(xlim=(1.5,3.7),ylabel='Height (m)',xlabel='Race time (s)');ax[0,1].legend(fontsize=8)
for a,k,label in [(ax[1,0],3,'Thrust (N)'),(ax[1,1],0,'Roll command (degrees)')]:
 scale=1 if k==3 else 180/np.pi
 a.plot(t,actual[:,k]*scale,marker='.',label='Recorded');a.plot(t,hold_pred[:,k]*scale,ls='--',label='Same state/history; previous obstacle order');a.plot(t,height_pred[:,k]*scale,ls=':',label='Same state/history; obstacle 1 z=1.55 m');a.axvline(t[107],color='k',ls=':');a.set(xlim=(2.04,2.27),ylabel=label,xlabel='Race time (s)',title='Slot swap at 2.149 s');a.legend(fontsize=7)
rpy=R.from_quat([e['obs']['quat'] for e in oo]).as_euler('xyz',degrees=True)
for k,label in [(0,'Roll'),(1,'Pitch')]:ax[2,0].plot(t,actual[:,k]*180/np.pi,label=f'{label} command');ax[2,0].plot(t,rpy[:,k],ls='--',label=f'{label} measured')
ax[2,0].set(xlim=(1.5,2.8),ylabel='Degrees',xlabel='Race time (s)',title='Attitude response during descent');ax[2,0].legend(fontsize=7)
ax[2,1].plot(t,[e['obs']['vel'][2] for e in oo]);ax[2,1].set(xlim=(1.5,2.9),ylabel='Estimated vertical velocity (m/s)',xlabel='Race time (s)',title='Already descending before the command jump');ax[2,1].axvline(t[107],color='k',ls=':')
for a in ax.flat:a.grid(alpha=.25)
fig.tight_layout();fig.savefig(out/'flight_113735.png',dpi=150);fig.savefig(out/'flight_113735.pdf')
