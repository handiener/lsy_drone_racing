"""Instrument the official sim.simulate runner without changing its dynamics or actions."""
import os
os.environ['JAX_PLATFORMS']='cpu'
os.environ['SCIPY_ARRAY_API']='1'
os.environ['MPLCONFIGDIR']='/tmp/drone-analysis-matplotlib'
import sys,json,copy,logging
from pathlib import Path
from unittest.mock import patch
root=Path(__file__).resolve().parents[3];sys.path.insert(0,str(root))
import numpy as np
import jax
import jax.numpy as jnp
import hashlib
from lsy_drone_racing.control.rl_race.env_wrapper.observations import ObsConfig, encode
from scripts import sim as official
out=Path(__file__).resolve().parent
original_loader=official.load_controller
original_config=official.load_config
episodes=[]

def clean(v):
 if isinstance(v,np.ndarray):return v.tolist()
 if isinstance(v,np.generic):return v.item()
 if hasattr(v,'__array__'):return np.asarray(v).tolist()
 raise TypeError(type(v).__name__)

def config_loader(path):
 cfg=original_config(path);cfg.env.seed=0;cfg.sim.render=False
 (out/'config.json').write_text(json.dumps(cfg.to_dict(),indent=2)+'\n')
 return cfg

def controller_loader(path):
 base=original_loader(path)
 class RecordingController(base):
  def __init__(self,obs,info,config):
   super().__init__(obs,info,config)
   self.records=[];self.episode={'initial_obs':copy.deepcopy(obs),'records':self.records}
   episodes.append(self.episode)
   model=Path(config.controller.get('model_dir','deploy_model')).expanduser()
   if not model.is_absolute():model=root/'lsy_drone_racing/control/rl_race'/model
   bundle=json.loads((model/'config.json').read_text())
   self._capture_features=jax.jit(lambda observation,previous:encode(
       {key:jnp.asarray(value)[None] for key,value in observation.items()},
       previous[None],ObsConfig(**bundle['obs']))[0])
   self._capture_features(obs,self._previous).block_until_ready()
   self.episode['policy_metadata']={'model_files':[
       {'path':str((model/name).resolve()),'sha256':hashlib.sha256((model/name).read_bytes()).hexdigest()}
       for name in ('config.json','policy.msgpack')]}
  def compute_control(self,obs,info=None):
   snapshot=copy.deepcopy(obs)
   previous=self._previous
   features=np.asarray(self._capture_features(obs,previous))
   action=super().compute_control(obs,info)
   self.records.append({'step':len(self.records),'time_s':len(self.records)/50,
      'obs':snapshot,'action':np.array(action,copy=True),'policy':{'features':features,'previous_action':np.asarray(previous),'policy_action':np.asarray(self._previous)}})
   return action
  def step_callback(self,action,obs,reward,terminated,truncated,info):
   self.episode.update(terminal_obs=copy.deepcopy(obs),terminated=bool(terminated),truncated=bool(truncated))
   return super().step_callback(action,obs,reward,terminated,truncated,info)
  def episode_callback(self):
   super().episode_callback()
   index=episodes.index(self.episode) if len(episodes)==1 else len(episodes)-1
   dest=out/f'sim_episode_{index:02d}.json'
   dest.write_text(json.dumps(self.episode,default=clean)+'\n')
   print('RECORDED',dest.name,'steps',len(self.records),'gates',self.episode['terminal_obs']['n_gates_passed'],flush=True)
 return RecordingController

logging.basicConfig(level=logging.INFO)
with patch.object(official,'load_controller',controller_loader),patch.object(official,'load_config',config_loader):
 result=official.simulate(config='level2.toml',n_runs=3,render=False)
(out/'runner_results.json').write_text(json.dumps({'seed':0,'n_runs':3,'episode_times':result},default=clean,indent=2)+'\n')
print('RESULTS',result,flush=True)
