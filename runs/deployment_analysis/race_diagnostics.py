"""Replay recorded policy inputs on CPU; counterfactuals are single-step only."""
import os
os.environ.setdefault('SCIPY_ARRAY_API', '1')
os.environ.setdefault('MPLCONFIGDIR', '/tmp/drone-analysis-matplotlib')
os.environ['JAX_PLATFORMS'] = 'cpu'
os.environ['MPLCONFIGDIR'] = '/tmp/drone-analysis-matplotlib'
import json, hashlib
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from lsy_drone_racing.control.rl_race.controller import load_policy, policy_mean_action
from lsy_drone_racing.control.rl_race.env_wrapper.actions import ActionConfig, to_command
from lsy_drone_racing.control.rl_race.env_wrapper.observations import ObsConfig, encode
from scipy.spatial.transform import Rotation
root=Path(__file__).resolve().parents[2]; out=Path(__file__).resolve().parent
model=root/'lsy_drone_racing/control/rl_race/deploy_model'
bundle,params=load_policy(model)
params=jax.tree.map(jnp.asarray,params)
@jax.jit
def predict(features):
    return to_command(policy_mean_action(params,features,bundle['network']['activation']),ActionConfig(**bundle['action']))
results=[];fig,axes=plt.subplots(4,2,figsize=(15,15),squeeze=False)
for path in ['deploy_20261117T100017.801841Z', 'deploy_20261117T100827.697140Z']:
    path=root/'runs'/path
    es=[json.loads(line) for line in (path/'events.jsonl').read_text().splitlines()]
    obs=[e for e in es if e['kind']=='observation']
    if not obs:continue
    col=len(results); pp={e['step']:e for e in es if e['kind']=='policy'};aa={e['step']:e for e in es if e['kind']=='action'}
    identity=next(e for e in es if e['kind']=='policy_metadata')
    for f in identity['model_files']:
        assert hashlib.sha256((model/Path(f['path']).name).read_bytes()).hexdigest()==f['sha256']
    features=np.array([pp[e['step']]['features'] for e in obs]);actual=np.array([aa[e['step']]['action'] for e in obs])
    time=np.array([(e['monotonic_ns']-obs[0]['monotonic_ns'])/1e9 for e in obs])
    corrected=[];held_order=features.copy();orders=[]
    for i,e in enumerate(obs):
        s=e['obs'];order=np.argsort(np.linalg.norm(np.array(s['obstacles_pos'])[:,:2]-s['pos'][:2],axis=1));orders.append(order)
        modified={k:jnp.asarray(v)[None] for k,v in s.items()}
        modified['obstacles_pos']=modified['obstacles_pos'].at[0,0,2].set(1.55)
        corrected.append(np.asarray(encode(modified,jnp.asarray(pp[i]['previous_action'])[None],ObsConfig(**bundle['obs']))[0]))
        if i:
            indices=[list(order).index(k) for k in orders[i-1]]
            held_order[i,-12:]=features[i,-12:].reshape(4,3)[indices].reshape(-1)
    factual=np.asarray(predict(jnp.asarray(features)));fixed=np.asarray(predict(jnp.asarray(np.array(corrected))));held=np.asarray(predict(jnp.asarray(held_order)))
    error=float(np.max(np.abs(factual-actual)));assert error<1e-4,error
    reveal=next(i for i,e in enumerate(obs) if e['obs']['obstacles_visited'][0]);switch=next(i for i in range(1,len(obs)) if 1.7<time[i]<1.9 and not np.array_equal(orders[i],orders[i-1]))
    items=[]
    for label,i in [('obstacle_1_reveal',reveal),('obstacle_rank_swap_after_gate_2',switch)]:
        items.append({'event':label,'step':i,'time_s':float(time[i]),'position':obs[i]['obs']['pos'],
        'previous_command':actual[i-1].tolist(),'actual_command':actual[i].tolist(),
        'correct_obstacle_1_height_command':fixed[i].tolist(),'keep_previous_obstacle_slot_order_command':held[i].tolist(),
        'previous_order':orders[i-1].tolist(),'current_order':orders[i].tolist()})
    results.append({'run':path.name,'factual_replay_max_command_error':error,'events':items})
    pos=np.array([e['obs']['pos'] for e in obs]);rpy=Rotation.from_quat([e['obs']['quat'] for e in obs]).as_euler('xyz',degrees=True)
    axes[0,col].plot(time,pos[:,2]);axes[0,col].set(title=path.name,ylabel='Estimated height (m)',xlim=(0,4.5))
    for k,label in [(0,'roll'),(1,'pitch')]:axes[1,col].plot(time,actual[:,k]*180/np.pi,label=f'Command {label}');axes[1,col].plot(time,rpy[:,k],ls='--',label=f'Measured {label}')
    axes[1,col].set(ylabel='Degrees',xlim=(1.6,3.9));axes[1,col].legend(fontsize=8)
    for ax,i,label in [(axes[2,col],reveal,'Before gate 1: obstacle-height reveal'),(axes[3,col],switch,'After gate 2: obstacle-slot swap')]:
        ax.plot(time,actual[:,3],marker='.',label='Recorded command')
        ax.plot(time,fixed[:,3],ls='--',label='Same state/history; obstacle 1 z = 1.55 m')
        if i==switch:ax.plot(time,held[:,3],ls=':',label='Same state/history; previous obstacle slot order')
        ax.axvline(time[i],color='black',lw=1);ax.set(xlim=(time[i]-.1,time[i]+.18),ylabel='Commanded thrust (N)',xlabel='Race time (s)',title=label);ax.legend(fontsize=8)
    for ax in axes[:,col]:ax.grid(alpha=.25)
fig.tight_layout();fig.savefig(out/'race_only.png',dpi=150);fig.savefig(out/'race_only.pdf')
(out/'race_counterfactuals.json').write_text(json.dumps(results,indent=2)+'\n')
print(json.dumps(results,indent=2))
