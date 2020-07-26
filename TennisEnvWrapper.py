from unityagents import UnityEnvironment
import numpy as np
import gym

class TennisEnvWrapper(gym.Wrapper):
  """
  :param env: (gym.Env) Gym environment that will be wrapped
  """
  def __init__(self, path=None, train_mode=True, verbose=True):
    if path is None:
        env = UnityEnvironment(file_name="Tennis_Linux_NoVis/Tennis.x86_64")
    else:
        env = UnityEnvironment(file_name=path)
    
    # Call the parent constructor, so we can access self.env later
    super(CustomWrapper, self).__init__(env)
    
    # get the default brain
    self.brain_name = env.brain_names[0]
    self.brain = env.brains[brain_name]

    # reset the environment
    self.env_info = env.reset(train_mode=train_mode)[self.brain_name]
    
    # number of agents 
    self.num_agents = len(self.env_info.agents)
    if verbose:
        print('Number of agents:', self.num_agents)

    # size of each action
    self.action_size = brain.vector_action_space_size
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
    obs = self.env.reset()
    return obs

  def step(self, action):
    """
    :param action: ([float] or int) Action taken by the agent
    :return: (np.ndarray, float, bool, dict) observation, reward, is the episode over?, additional informations
    """
    obs, reward, done, info = self.env.step(action)
    return obs, reward, done, info






for i in range(1, 6):                                      # play game for 5 episodes
    env_info = env.reset(train_mode=False)[brain_name]     # reset the environment    
    states = env_info.vector_observations                  # get the current state (for each agent)
    scores = np.zeros(num_agents)                          # initialize the score (for each agent)
    while True:
        actions = np.random.randn(num_agents, action_size) # select an action (for each agent)
        actions = np.clip(actions, -1, 1)                  # all actions between -1 and 1
        env_info = env.step(actions)[brain_name]           # send all actions to tne environment
        next_states = env_info.vector_observations         # get next state (for each agent)
        rewards = env_info.rewards                         # get reward (for each agent)
        dones = env_info.local_done                        # see if episode finished
        scores += env_info.rewards                         # update the score (for each agent)
        states = next_states                               # roll over states to next time step
        if np.any(dones):                                  # exit loop if episode finished
            break
    print('Score (max over agents) from episode {}: {}'.format(i, np.max(scores)))
    
env.close()

