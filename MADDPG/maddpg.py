# main code that contains the neural network setup
# policy + critic updates
# see ddpg.py for other details in the network

from ddpg import DDPGAgent
import torch
from utilities import soft_update, transpose_to_tensor, transpose_list
from utilities import to_tensor
from utilities import print_variable

import torch.nn.functional as f

import logging

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = 'cpu'
TAU = 0.02

class MADDPG:
    def __init__(self, discount_factor=0.95, tau=0.02):
        super(MADDPG, self).__init__()

        # critic input = obs_full + actions = 24+24+2+2=52
#         ddpg = DDPGAgent(24, 400, 300, 2, 52, 400, 300, lr_actor=1e-4, lr_critic=1e-3)
        self.maddpg_agent = [DDPGAgent(24, 512, 256, 2, 52, 512, 256, lr_actor=1e-4, lr_critic=1e-3), 
                             DDPGAgent(24, 512, 256, 2, 52, 512, 256, lr_actor=1e-4, lr_critic=1e-3)]
        # Use same agent for both
#         self.maddpg_agent = [ddpg, ddpg]
        
        self.discount_factor = discount_factor
        self.tau = tau
        self.iter = 0

    def get_actors(self):
        """get actors of all the agents in the MADDPG object"""
        actors = [ddpg_agent.actor for ddpg_agent in self.maddpg_agent]
        return actors

    def get_target_actors(self):
        """get target_actors of all the agents in the MADDPG object"""
        target_actors = [ddpg_agent.target_actor for ddpg_agent in self.maddpg_agent]
        return target_actors

    def act(self, obs_all_agents, noise=0.0):
        """get actions from all agents in the MADDPG object"""
            
        logging.debug('######### MADDPG.PY - ACT ######### PASS OBS TO ACTOR NET')
        for agent, obs in zip(self.maddpg_agent, obs_all_agents):
          logging.debug('Agent observation:')
          print_variable(obs, 'obs')
          logging.debug('#########')
        
        logging.debug('#########')
        logging.debug('#########')
        logging.debug('')
        logging.debug('')
        
        
        actions = [agent.act(obs, noise) for agent, obs in zip(self.maddpg_agent, obs_all_agents)]
        actions = [torch.clamp(action, -1, 1) for action in actions]
        
        logging.debug('######### MADDPG.PY - ACT ######### ACTIONS RETURNED')
        print_variable(actions, 'actions')
        logging.debug('#########')
        logging.debug('#########')
        logging.debug('')
        logging.debug('')
        
        return actions

    def target_act(self, obs_all_agents, noise=0.0):
        """get target network actions from all the agents in the MADDPG object """
        target_actions = [ddpg_agent.target_act(obs, noise) for ddpg_agent, obs in zip(self.maddpg_agent, obs_all_agents)]

        return target_actions

    def update(self, samples, agent_number, logger):
        """update the critics and actors of all the agents """

        # need to transpose each element of the samples
        # to flip obs[parallel_agent][agent_number] to
        # obs[agent_number][parallel_agent]
        
        logging.debug('######### STEP 8 ######### UPDATE Agent: ' + str(agent_number))
        print_variable(samples, 'samples')
        logging.debug('#########')
        logging.debug('#########')
        logging.debug('')
        logging.debug('')
        
        obs, obs_full, action, reward, next_obs, next_obs_full, done = map(transpose_to_tensor, samples)
        
        logging.debug('######### MADDPG.PY - UPDATE ######### UNPACK SAMPLE')
        print_variable(obs, 'obs')
        print_variable(obs_full, 'obs_full')
        print_variable(obs_full[0], 'obs_full[0]')
        print_variable(action, 'action')
        print_variable(reward, 'reward')
        print_variable(next_obs, 'next_obs')
        print_variable(next_obs_full, 'next_obs_full')
        print_variable(done, 'done')
        logging.debug('#########')
        logging.debug('#########')
        logging.debug('')
        logging.debug('')
        

        obs_full = torch.stack(obs_full)
        next_obs_full = torch.stack(next_obs_full)
        
        
        logging.debug('######### MADDPG.PY - UPDATE ######### STACKED FULL OBS')
        print_variable(obs_full, 'stacked obs_full')
        print_variable(next_obs_full, 'stacked next_obs_full')
        print_variable(done, 'done')
        logging.debug('#########')
        logging.debug('#########')
        logging.debug('')
        logging.debug('')
        
        logging.debug('# ---------------------------- update critic ---------------------------- #')
        
        agent = self.maddpg_agent[agent_number]
        agent.critic_optimizer.zero_grad()

        #critic loss = batch mean of (y- Q(s,a) from target network)^2
        #y = reward of this timestep + discount * Q(st+1,at+1) from target network
        target_actions = self.target_act(next_obs)
        
        
        logging.debug('######### MADDPG.PY - UPDATE ######### TARGET ACTIONS')
        print_variable(target_actions, 'target_actions')
        logging.debug('#########')
        logging.debug('#########')
        logging.debug('')
        logging.debug('')
        
        
        target_actions = torch.cat(target_actions, dim=1)
        
        logging.debug('######### MADDPG.PY - UPDATE ######### TARGET ACTIONS - CONCAT')
        print_variable(target_actions, 'target_actions')
        logging.debug('#########')
        logging.debug('#########')
        logging.debug('')
        logging.debug('')
        
        target_critic_input = torch.cat((torch.squeeze(next_obs_full.t()),target_actions), dim=1).to(device)
        
        logging.debug('######### MADDPG.PY - UPDATE ######### TARGET CRITIC NETWORK - INPUT')
        print_variable(torch.cat((torch.squeeze(next_obs_full.t()),target_actions), dim=1), 'torch.cat((torch.squeeze(next_obs_full.t()),target_actions), dim=1)')
        logging.debug('#########')
        logging.debug('#########')
        logging.debug('')
        logging.debug('')
        
        with torch.no_grad():
            q_next = agent.target_critic(target_critic_input)
        
        logging.debug('######### MADDPG.PY - UPDATE ######### TARGET CRITIC NETWORK - Q Next')
        print_variable(q_next, 'q_next')
        logging.debug('#########')
        logging.debug('#########')
        logging.debug('')
        logging.debug('')
        
        y = reward[agent_number].view(-1, 1) + self.discount_factor * q_next * (1 - done[agent_number].view(-1, 1))
        action = torch.cat(action, dim=1)
        
        logging.debug('######### MADDPG.PY - UPDATE ######### y')
        print_variable(y, 'y')
        logging.debug('#########')
        logging.debug('#########')
        logging.debug('')
        logging.debug('')
        
        logging.debug('######### MADDPG.PY - UPDATE ######### CRITIC INPUT - ACTION')
        print_variable(action, 'action')
        logging.debug('#########')
        logging.debug('#########')
        logging.debug('')
        logging.debug('')
               
        critic_input = torch.cat((torch.squeeze(obs_full.t()), action), dim=1).to(device)
        q = agent.critic(critic_input)
        
        logging.debug('######### MADDPG.PY - UPDATE ######### CRITIC INPUT')
        print_variable(critic_input, 'critic_input')
        logging.debug('#########')
        logging.debug('#########')
        logging.debug('')
        logging.debug('')

#         huber_loss = torch.nn.SmoothL1Loss()
#         critic_loss = huber_loss(q, y.detach())
        critic_loss = f.mse_loss(q, y.detach())
        critic_loss.backward()
        #torch.nn.utils.clip_grad_norm_(agent.critic.parameters(), 0.5)
        torch.nn.utils.clip_grad_norm_(agent.critic.parameters(), 0.5)
        agent.critic_optimizer.step()

        logging.debug('# ---------------------------- update actor ---------------------------- #')
        
        logging.debug('######### MADDPG.PY - UPDATE ######### ACTOR NETWORK - INPUT')
        print_variable(obs, 'obs')
        logging.debug('#########')
        logging.debug('#########')
        logging.debug('')
        logging.debug('')
        
        # update actor network using policy gradient
        agent.actor_optimizer.zero_grad()
        # make input to agent
        # detach the other agents to save computation
        # saves some time for computing derivative
        q_input = [ self.maddpg_agent[i].actor(ob) if i == agent_number \
                   else self.maddpg_agent[i].actor(ob).detach()
                   for i, ob in enumerate(obs) ]
        
        logging.debug('######### MADDPG.PY - UPDATE ######### CRITIC NETWORK - INPUT')
        print_variable(q_input, 'q_input')
        logging.debug('#########')
        logging.debug('#########')
        logging.debug('')
        logging.debug('')
        
        q_input = torch.cat(q_input, dim=1)
        # combine all the actions and observations for input to critic
        # many of the obs are redundant, and obs[1] contains all useful information already
        q_input2 = torch.cat((torch.squeeze(obs_full.t()), q_input), dim=1)
        
                
        logging.debug('######### MADDPG.PY - UPDATE ######### CRITIC NETWORK - INPUT FORMATTED')
        print_variable(q_input2, 'q_input2')
        logging.debug('#########')
        logging.debug('#########')
        logging.debug('')
        logging.debug('')
        
        logging.debug('######### MADDPG.PY - UPDATE ######### UPDATE FINISHED')
        
        # get the policy gradient
        actor_loss = -agent.critic(q_input2).mean()
        actor_loss.backward()
        #torch.nn.utils.clip_grad_norm_(agent.actor.parameters(),0.5)
        torch.nn.utils.clip_grad_norm_(agent.actor.parameters(),0.5)
        agent.actor_optimizer.step()

        al = actor_loss.cpu().detach().item()
        cl = critic_loss.cpu().detach().item()
        logger.add_scalars('agent%i/losses' % agent_number,
                           {'critic loss': cl,
                            'actor_loss': al},
                           self.iter)

    def update_targets(self):
        """soft update targets"""
        self.iter += 1
        for ddpg_agent in self.maddpg_agent:
            soft_update(ddpg_agent.target_actor, ddpg_agent.actor, self.tau)
            soft_update(ddpg_agent.target_critic, ddpg_agent.critic, self.tau)
            
            
            
    def reset(self):
      for ddpg_agent in self.maddpg_agent:
        ddpg_agent.reset()


