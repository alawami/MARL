from unityagents import UnityEnvironment
import numpy as np
import gym

class TennisEnvWrapper(gym.Wrapper):
  """
  :param env: (gym.Env) Gym environment that will be wrapped
  """
  def __init__(self, path=None, train_mode=True, verbose=True, max_steps=None):
    if path is None:
        self.env = UnityEnvironment(file_name="Tennis_Linux_NoVis/Tennis.x86_64")
    else:
        self.env = UnityEnvironment(file_name=path)
    
    # Call the parent constructor, so we can access self.env later
    super(TennisEnvWrapper, self).__init__(self.env)
    
    self.train_mode = train_mode
    self.max_steps = max_steps
    # Counter of steps per episode
    self.current_step = 0
    
    # get the default brain
    self.brain_name = self.env.brain_names[0]
    self.brain = self.env.brains[brain_name]

    # reset the environment
    self.env_info = env.reset(train_mode=self.train_mode)[self.brain_name]
    
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
    
    if verbose:
        print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], self.state_size))
        print('The state for the first agent looks like:', states[0])
  
  def reset(self):
    """
    Reset the environment 
    """
    self.env_info = env.reset(train_mode=self.train_mode)[self.brain_name]
    obs = self.env_info.vector_observations
    
    self.current_step = 0
    
    return obs

  def step(self, action):
    """
    :param action: ([float] or int) Action taken by the agent
    :return: (np.ndarray, float, bool, dict) observation, reward, is the episode over?, additional informations
    """
    self.env_info = self.env.step(action)[self.brain_name]           # send all actions to tne environment
    next_state = self.env_info.vector_observations         # get next state (for each agent)
    reward = self.env_info.rewards                         # get reward (for each agent)
    done = self.env_info.local_done                        # see if episode finished
    
    obs = next_state
    info = dict()
    
    self.current_step += 1
    # Overwrite the done signal when 
    if self.current_step >= self.max_steps:
      done = True
      # Update the info dict to signal that the limit was exceeded
      info['time_limit_reached'] = True
    
    return obs, reward, done, info


    def close(self):
        self.env.close()

