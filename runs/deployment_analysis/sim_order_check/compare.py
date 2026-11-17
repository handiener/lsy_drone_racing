"""Compare the same obstacle-1/3 rank crossing in official simulation and hardware."""
import os
os.environ.setdefault('JAX_PLATFORMS','cpu')
os.environ.setdefault('SCIPY_ARRAY_API','1')
os.environ['MPLCONFIGDIR']='/tmp/drone-analysis-matplotlib'
import json,csv
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
out=Path(__file__).resolve().parent;root=out.parents[2]
flights=[]
for p in sorted(out.glob('sim_episode_*.json')):
 d=json.loads(p.read_text());flights.append((p.stem,d['records'],d['terminal_obs']))
p=root/'runs/deploy_20261117T113735.863815Z';es=[json.loads(l) for l in (p/'events.jsonl').read_text().splitlines()];oo=[e for e in es if e['kind']=='observation'];aa={e['step']:e['action'] for e in es if e['kind']=='action'};zero=oo[0]['monotonic_ns'];rr=[{'step':e['step'],'time_s':(e['monotonic_ns']-zero)/1e9,'obs':e['obs'],'action':aa[e['step']]} for e in oo];flights.append(('hardware_113735',rr,None))
summary=[];all_swaps=[];fig,axes=plt.subplots(2,2,figsize=(13,9))
for name,rr,terminal in flights:
 orders=[np.argsort(np.linalg.norm(np.array(e['obs']['obstacles_pos'])[:,:2]-e['obs']['pos'][:2],axis=1)).tolist() for e in rr]
 changes=[]
 for i in range(1,len(rr)):
  old,new=orders[i-1],orders[i]
  if old.index(0)==new.index(2) and old.index(2)==new.index(0) and old!=new:
   changes.append(i);all_swaps.append({'flight':name,'step':i,'time_s':rr[i]['time_s'],'gate_count':rr[i]['obs']['n_gates_passed'],'old_order_one_based':[x+1 for x in old],'new_order_one_based':[x+1 for x in new],'thrust_delta_N':rr[i]['action'][3]-rr[i-1]['action'][3]})
 i=next(i for i in changes if rr[i]['obs']['n_gates_passed']==2);e=rr[i];s=e['obs'];tilts=np.degrees(np.arccos(np.clip(R.from_quat([q['obs']['quat'] for q in rr]).as_matrix()[:,2,2],-1,1)))
 t=np.array([q['time_s']-e['time_s'] for q in rr]);actions=np.array([q['action'] for q in rr]);pos=np.array([q['obs']['pos'] for q in rr]);vel=np.array([q['obs']['vel'] for q in rr]);gate3time=next(q['time_s'] for q in rr if q['obs']['n_gates_passed']==3)
 metrics={'flight':name,'swap_time_s':e['time_s'],'gate3_first_observation_time_s':gate3time,'old_order_one_based':[k+1 for k in orders[i-1]],'new_order_one_based':[k+1 for k in orders[i]],'pos_m':s['pos'],'vel_m_s':s['vel'],'measured_tilt_deg':float(tilts[i]),'previous_command':rr[i-1]['action'],'current_command':e['action'],'thrust_delta_N':e['action'][3]-rr[i-1]['action'][3],'roll_delta_deg':float((e['action'][0]-rr[i-1]['action'][0])*180/np.pi),'finished_track':terminal is not None and terminal['n_gates_passed']==5}
 summary.append(metrics)
 style={'color':'black','lw':2.3} if name.startswith('hardware') else {'alpha':.85}
 axes[0,0].plot(t,actions[:,3],label=name,**style);axes[0,1].plot(t,pos[:,2],label=name,**style);axes[1,0].plot(t,tilts,label=name,**style);axes[1,1].plot(t,vel[:,2],label=name,**style)
 with (out/f'{name}_ordering.csv').open('w',newline='') as f:
  w=csv.writer(f, lineterminator="\n");w.writerow(['step','time_s','gates_passed','sorted_obstacle_numbers','x','y','z','vx','vy','vz','measured_tilt_deg','roll_cmd_rad','pitch_cmd_rad','yaw_cmd_rad','thrust_cmd_N'])
  for j,q in enumerate(rr):w.writerow([q['step'],q['time_s'],q['obs']['n_gates_passed'],' '.join(str(x+1) for x in orders[j]),*q['obs']['pos'],*q['obs']['vel'],tilts[j],*q['action']])
for ax,y in zip(axes.flat,['Commanded thrust (N)','Height (m)','Measured tilt from vertical (degrees)','Vertical velocity (m/s)']):
 ax.set(xlim=(-.5,.6),xlabel='Seconds relative to obstacle 1/3 slot swap before gate 3',ylabel=y);ax.axvline(0,color='gray',ls=':');ax.grid(alpha=.25);ax.legend(fontsize=8)
fig.tight_layout();fig.savefig(out/'sim_vs_real_swap.png',dpi=150);fig.savefig(out/'sim_vs_real_swap.pdf')
# Reproduce the single-step simulation ordering counterfactuals as well.
import jax
import jax.numpy as jnp
from lsy_drone_racing.control.rl_race.controller import load_policy, policy_mean_action
from lsy_drone_racing.control.rl_race.env_wrapper.actions import ActionConfig, to_command
bundle,params=load_policy(root/'lsy_drone_racing/control/rl_race/deploy_model')
params=jax.tree.map(jnp.asarray,params)
@jax.jit
def predict(x):
 return to_command(policy_mean_action(params,x,bundle['network']['activation']),ActionConfig(**bundle['action']))
for metric in summary:
 if not metric['flight'].startswith('sim_episode_'):continue
 episode=json.loads((out/(metric['flight']+'.json')).read_text())
 e=next(q for q in episode['records'] if q['time_s']==metric['swap_time_s'])
 features=np.array(e['policy']['features']);held=features.copy()
 old=[x-1 for x in metric['old_order_one_based']];new=[x-1 for x in metric['new_order_one_based']]
 held[-12:]=features[-12:].reshape(4,3)[[new.index(k) for k in old]].reshape(-1)
 values=np.asarray(predict(jnp.asarray(np.stack([features,held]))))
 np.testing.assert_allclose(values[0],e['action'],atol=1e-5)
 metric['same_state_previous_order_command']=values[1].tolist()
 metric['same_state_order_effect_on_thrust_N']=float(values[0,3]-values[1,3])
(out/'comparison_metrics.json').write_text(json.dumps({'main_swap_comparison':summary,'all_obstacle_1_3_exchanges':all_swaps},indent=2)+'\n')
print(json.dumps(summary,indent=2))
