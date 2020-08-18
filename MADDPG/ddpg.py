# individual network settings for each actor + critic pair
# see networkforall for details

from networkforall import Network
from utilities import hard_update, gumbel_softmax, onehot_from_logits
from utilities import print_variable
from torch.optim import Adam
import torch
import numpy as np

import logging

# add OU noise for exploration
from OUNoise import OUNoise

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = 'cpu'

WEIGHT_DECAY = 0 # 1.e-5
OU_SIGMA = 0.1

class DDPGAgent:
    """Reinforcement agent implementing DDPG to be used with MADDPG"""
    def __init__(self, in_actor, hidden_in_actor, hidden_out_actor, out_actor, in_critic, hidden_in_critic, hidden_out_critic, lr_actor=1.0e-2, lr_critic=1.0e-2):
        self.actor = Network(in_actor, hidden_in_actor, hidden_out_actor, out_actor, actor=True).to(device)
        self.critic = Network(in_critic, hidden_in_critic, hidden_out_critic, 1).to(device)
        self.target_actor = Network(in_actor, hidden_in_actor, hidden_out_actor, out_actor, actor=True).to(device)
        self.target_critic = Network(in_critic, hidden_in_critic, hidden_out_critic, 1).to(device)

        self.noise = OUNoise(out_actor, scale=1.0, sigma=OU_SIGMA)

        
        # initialize targets same as original networks
        hard_update(self.target_actor, self.actor)
        hard_update(self.target_critic, self.critic)

        self.actor_optimizer = Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=lr_critic, weight_decay=WEIGHT_DECAY)


    def act(self, obs, noise=0.0):
        """returns the actions a given state"""
      
        logging.debug('######### DDPG.PY - ACT ######### INPUT OBS')
        print_variable(obs, 'obs')
        logging.debug('#########')
        logging.debug('#########')
        logging.debug('')
        logging.debug('')
        
    
        obs = obs.to(device).unsqueeze(0) # Add dimension for minibatch for single process
        
        logging.debug('######### DDPG.PY - ACT ######### OBS (SQUEEZED) INPUT TO ACTOR NET')
        print_variable(obs, 'unsqueezed obs')
        logging.debug('#########')
        logging.debug('#########')
        logging.debug('')
        logging.debug('')

        
        self.actor.eval()
        with torch.no_grad():
            action = self.actor(obs) + noise*self.noise.noise()
        self.actor.train()
        
        
        logging.debug('######### DDPG.PY - ACT ######### ACTION RETURNED')
        print_variable(action, 'action')
        logging.debug('#########')
        logging.debug('#########')
        logging.debug('')
        logging.debug('')
        
        
        return action

    def target_act(self, obs, noise=0.0):
        """returns the actions for a given state for target networks"""
        obs = obs.to(device)
        
        # No need to change mode to eval for target network
        self.actor.eval()
        with torch.no_grad():
          action = self.target_actor(obs) + noise*self.noise.noise()
        self.actor.train()
        
        return action
      
    def reset(self):
        """reset noise process"""
        self.noise.reset()
