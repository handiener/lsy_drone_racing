"""Analyze the 11:00:59 UTC flight; CPU-only offline policy replay."""
import os
os.environ['JAX_PLATFORMS']='cpu'
os.environ['SCIPY_ARRAY_API']='1'
os.environ['MPLCONFIGDIR']='/tmp/drone-analysis-matplotlib'
import json,hashlib
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from lsy_drone_racing.control.rl_race.controller import load_policy,policy_mean_action
from lsy_drone_racing.control.rl_race.env_wrapper.observations import encode,ObsConfig
from lsy_drone_racing.control.rl_race.env_wrapper.actions import to_command,ActionConfig
root=Path(__file__).resolve().parents[2];out=Path(__file__).resolve().parent
p=root/'runs/deploy_20261117T110059.075060Z';es=[json.loads(l) for l in (p/'events.jsonl').read_text().splitlines()]
oo=[e for e in es if e['kind']=='observation'];pp={e['step']:e for e in es if e['kind']=='policy'};aa={e['step']:e['action'] for e in es if e['kind']=='action'}
model=root/'lsy_drone_racing/control/rl_race/deploy_model';bundle,params=load_policy(model);params=jax.tree.map(jnp.asarray,params)
for f in next(e for e in es if e['kind']=='policy_metadata')['model_files']:
 assert hashlib.sha256((model/Path(f['path']).name).read_bytes()).hexdigest()==f['sha256']
@jax.jit
def predict(features):return to_command(policy_mean_action(params,features,bundle['network']['activation']),ActionConfig(**bundle['action']))
t=np.array([(e['monotonic_ns']-oo[0]['monotonic_ns'])/1e9 for e in oo]);features=np.array([pp[i]['features'] for i in range(len(oo))]);acts=np.array([aa[i] for i in range(len(oo))]);corrected=[]
for i,e in enumerate(oo):
 obs={k:jnp.asarray(v)[None] for k,v in e['obs'].items()};obs['obstacles_pos']=obs['obstacles_pos'].at[0,0,2].set(1.55)
 corrected.append(np.asarray(encode(obs,jnp.asarray(pp[i]['previous_action'])[None],ObsConfig(**bundle['obs']))[0]))
factual=np.asarray(predict(jnp.asarray(features)));fixed=np.asarray(predict(jnp.asarray(np.array(corrected))));err=float(np.abs(factual-acts).max());assert err<1e-4
raw=[e for e in es if e['kind']=='external_pose_sent'];rt=np.array([(e['monotonic_ns']-oo[0]['monotonic_ns'])/1e9 for e in raw]);rp=np.array([e['pos'] for e in raw]);rq=R.from_quat([e['quat'] for e in raw]);tilt=np.rad2deg(np.arccos(np.clip(rq.as_matrix()[:,2,2],-1,1)))
reset=next(e for e in es if e['kind']=='reset');g=np.array(reset['track']['gates_pos'][2]);gr=R.from_quat(reset['track']['gates_quat'][2]);local=gr.inv().apply(rp-g)
chosen=[]
for target in [.602,1.988,2.169,2.611,2.631,2.651,2.691,2.712]:
 i=int(np.abs(t-target).argmin());chosen.append({'step':i,'time_s':float(t[i]),'recorded_command':acts[i].tolist(),'same_state_corrected_height_command':fixed[i].tolist(),'position_m':oo[i]['obs']['pos'],'obstacle_1':oo[i]['obs']['obstacles_pos'][0]})
metrics={'run':p.name,'replay_max_command_error':err,'steps':chosen,'mocap_near_gate3':[{'time_s':float(rt[i]),'pos_m':rp[i].tolist(),'gate3_local_pos_m':local[i].tolist(),'tilt_deg':float(tilt[i])} for i in range(len(rt)) if 2.55<rt[i]<2.8]}
(out/'flight_110059_metrics.json').write_text(json.dumps(metrics,indent=2)+'\n');print(json.dumps(metrics,indent=2))
fig,ax=plt.subplots(3,2,figsize=(14,12))
ax[0,0].plot(rt,rp[:,2]);ax[0,0].set(xlim=(0,4.7),ylabel='Mocap z (m)',xlabel='Race time (s)',title='Race before gate 3 crossing')
ax[0,0].axvline(.602,color='C3',ls='--',label='Obstacle 1 reveal');ax[0,0].axvline(2.625,color='k',ls=':',label='Rapid pitch rotation begins');ax[0,0].legend()
ax[0,1].plot(t,[e['obs']['obstacles_pos'][0][2] for e in oo]);ax[0,1].set(xlim=(0,4.7),ylabel='Obstacle 1 input z (m)',xlabel='Race time (s)',title='One height jump at 0.602 s; constant afterward')
ax[1,0].plot(t,acts[:,3],marker='.',label='Recorded');ax[1,0].plot(t,fixed[:,3],ls='--',label='Same state/history; obstacle 1 z=1.55 m');ax[1,0].set(xlim=(.48,.85),ylabel='Commanded thrust (N)',xlabel='Race time (s)',title='Height reveal before gate 1');ax[1,0].legend(fontsize=8)
ax[1,1].plot(t,acts[:,3],marker='.',label='Recorded');ax[1,1].plot(t,fixed[:,3],ls='--',label='Same state/history; obstacle 1 z=1.55 m');ax[1,1].set(xlim=(2.45,2.85),ylabel='Commanded thrust (N)',xlabel='Race time (s)',title='Thrust falls after rotation begins');ax[1,1].axvline(2.625,color='k',ls=':');ax[1,1].legend(fontsize=8)
ax[2,0].plot(rt,tilt,label='Measured tilt from vertical');cmdrot=R.from_euler('xyz',acts[:,:3]);cmdtilt=np.degrees(np.arccos(np.clip(cmdrot.as_matrix()[:,2,2],-1,1)));ax[2,0].plot(t,cmdtilt,label='Commanded tilt');ax[2,0].axhline(90,color='gray',ls='--');ax[2,0].set(xlim=(2.45,2.9),ylabel='Tilt (degrees)',xlabel='Race time (s)',title='Measured rotation departs sharply from command');ax[2,0].legend(fontsize=8)
m=(rt>2.35)&(rt<2.85);sc=ax[2,1].scatter(local[m,0],local[m,2],c=rt[m],cmap='viridis');ax[2,1].plot(local[m,0],local[m,2],lw=.7);ax[2,1].plot([0,0],[-.36,-.2],color='C3',lw=8,label='Approx. lower gate frame (0.72 m outer / 0.40 m opening)');ax[2,1].set(xlabel='Gate-local forward position (m)',ylabel='Gate-local vertical position (m)',title='Approach near bottom of gate 3');ax[2,1].legend(fontsize=7);fig.colorbar(sc,ax=ax[2,1],label='Race time (s)')
for a in ax.flat:a.grid(alpha=.25)
fig.tight_layout();fig.savefig(out/'flight_110059.png',dpi=150);fig.savefig(out/'flight_110059.pdf')
