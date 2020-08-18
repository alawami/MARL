from unityagents import UnityEnvironment
import numpy as np
import gym

# from env_wrapper import SubprocVecEnv, DummyVecEnv
from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines.common import set_global_seeds
from stable_baselines.common.cmd_util import make_vec_env
# from baselines.common.vec_env import subproc_vec_env as SubprocVecEnv
from stable_baselines.common.cmd_util import make_vec_env

class TennisEnv():
  """
  :param env: (gym.Env) Gym environment that will be wrapped
  """
  def __init__(self, path=None, train_mode=True, verbose=True, max_steps=None, seed=0):
    if path is None:
      self.env = UnityEnvironment(file_name="../Tennis_Linux_NoVis/Tennis.x86_64", seed=seed)
    else:
      self.env = UnityEnvironment(file_name=path, seed=seed)
    
    self.train_mode = train_mode
    self.max_steps = max_steps
    # Counter of steps per episode
    self.current_step = 0
    
    # get the default brain
    self.brain_name = self.env.brain_names[0]
    self.brain = self.env.brains[self.brain_name]

    # reset the environment
    self.env_info = self.env.reset(train_mode=self.train_mode)[self.brain_name]
    
    # number of agents 
    self.num_agents = len(self.env_info.agents)
    if verbose:
      print('Number of agents:', self.num_agents)

    # size of each action
    self.action_size = self.brain.vector_action_space_size
    if verbose:
      print('Size of each action:', self.action_size)

    # examine the state space 
    states = self.env_info.vector_observations
    self.state_size = states.shape[1]
    
#     self.action_space = gym.spaces.Box(low=-1, high=1, shape=(2,2))
#     self.observation_space = gym.spaces.Box(low=-100, high=100, shape=(2,24))
    
    
    if verbose:
      print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], self.state_size))
      print('The state for the first agent looks like:', states[0])
  
  def reset(self):
    """
    Reset the environment 
    """
    self.env_info = self.env.reset(train_mode=self.train_mode)[self.brain_name]
    obs = self.env_info.vector_observations
    
    self.current_step = 0
    
    return obs, obs.reshape(1,48)

  def step(self, action):
    """
    :param action: ([float] or int) Action taken by the agent
    :return: (np.ndarray, float, bool, dict) observation, reward, is the episode over?, additional informations
    """
    # Single Process/Env situation
    action = action[0]
    
    self.env_info = self.env.step(action)[self.brain_name] # send all actions to tne environment
    next_state = self.env_info.vector_observations         # get next state (for each agent)
    reward = self.env_info.rewards                         # get reward (for each agent)
    done = self.env_info.local_done                        # see if episode finished
    
    obs = next_state
    info = dict()
    
    self.current_step += 1
    # Overwrite the done signal when 
    if self.max_steps is not None and self.current_step >= self.max_steps:
      done = [True, True]
      # Update the info dict to signal that the limit was exceeded
      info['time_limit_reached'] = True
    
    # To Do: Does this generalizes to parallel environments
    
    return obs, obs.reshape(1,48), reward, done, info


    def close(self):
        self.env.close()

        
def make_env(env_id, rank, seed=0):
  """
  Utility function for multiprocessed env.

  :param env_id: (str) the environment ID
  :param seed: (int) the inital seed for RNG
  :param rank: (int) index of the subprocess
  """

  if not env_id == 'Tennis':
    raise 
  TennisEnv(train_mode=True, 
                           verbose=False, 
                           max_steps=None, 
                           seed=seed + rank) # Important: use a different seed for each environment
  
  def _init():
    env = TennisEnv(train_mode=True, 
                           verbose=False, 
                           max_steps=None, 
                           seed=seed + rank) # Important: use a different seed for each environment
    return env

  set_global_seeds(seed)

  return _init

def make_parallel_env(n_rollout_threads, seed=1):
  def get_env_fn(rank):
      def init_env():
          env = make_env("Tennis", seed + rank * 1000)
          np.random.seed(seed + rank * 1000)
          return env
      return init_env
#    if n_rollout_threads == 1:
#        return DummyVecEnv([get_env_fn(0)])
#    else:
  return SubprocVecEnv([get_env_fn(i) for i in range(n_rollout_threads)], start_method='spawn')

def make_parallel_env2(n_procs, seed=0):
  return SubprocVecEnv([make_env('Tennis', i, seed) for i in range(n_procs)], start_method='spawn')
#   return make_vec_env(env_id, n_envs=n_procs, vec_env_cls=SubprocVecEnv, vec_env_kwargs=dict(start_method='spawn'))