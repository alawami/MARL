import torch
import torch.nn as nn
import torch.nn.functional as f
import numpy as np

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class Network(nn.Module):
    def __init__(self, input_dim, hidden_in_dim, hidden_out_dim, output_dim, actor=False):
        super(Network, self).__init__()

        """self.input_norm = nn.BatchNorm1d(input_dim)
        self.input_norm.weight.data.fill_(1)
        self.input_norm.bias.data.fill_(0)"""
        
        if actor:
#           self.bn0 = nn.BatchNorm1d(input_dim)
          self.fc1 = nn.Linear(input_dim,hidden_in_dim)
          self.bn1 = nn.BatchNorm1d(hidden_in_dim)
          self.fc2 = nn.Linear(hidden_in_dim,hidden_out_dim)
          self.bn2 = nn.BatchNorm1d(hidden_out_dim)
          self.fc3 = nn.Linear(hidden_out_dim,output_dim)
          
          self.drop1 = nn.Dropout(p=0.2)
          self.drop2 = nn.Dropout(p=0.2)
        else:
#           self.bn0 = nn.BatchNorm1d(input_dim)
          self.fc1 = nn.Linear(input_dim,hidden_in_dim)
          self.bn1 = nn.BatchNorm1d(hidden_in_dim)
          self.fc2 = nn.Linear(hidden_in_dim,hidden_out_dim)
          self.bn2 = nn.BatchNorm1d(hidden_out_dim)
          self.fc3 = nn.Linear(hidden_out_dim,output_dim)
          
#           state_dim = 48
#           action_dim = 4
#           self.fc1 = nn.Linear(in_features=state_dim, out_features=hidden_in_dim)
#           self.bn1 = nn.BatchNorm1d(num_features=state_dim)
#           self.fc2 = nn.Linear(in_features=hidden_in_dim+action_dim, out_features=hidden_out_dim)
#           self.fc3 = nn.Linear(in_features=hidden_out_dim, out_features=1)
          
          
#           self.drop1 = nn.Dropout(p=0.2)
#           self.drop2 = nn.Dropout(p=0.2)
        
#         self.fc3 = nn.Linear(hidden_out_dim,100)
#         self.bn3 = nn.BatchNorm1d(100)
#         self.fc4 = nn.Linear(128,output_dim)
        
        self.nonlin = f.leaky_relu #f.leaky_relu
        self.actor = actor
        self.reset_parameters()
        

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-1e-3, 1e-3)

    def forward(self, x):
        if self.actor:
            # return a vector of the force
#             x = self.bn0(x)
#             h1 = self.nonlin(self.bn1(self.fc1(x)))
            h1 = self.nonlin(self.fc1(x))

#             h2 = self.nonlin(self.bn2(self.fc2(h1)))
            h2 = self.nonlin(self.fc2(h1))
            h3 = self.fc3(h2)
            
#             h3 = self.nonlin(self.bn3(self.fc3(h2)))
#             h4 = self.nonlin(self.fc4(h3))

#             norm = torch.norm(h3)
            
#             # h3 is a 2D vector (a force that is applied to the agent)
#             # we bound the norm of the vector to be between 0 and 10
#             return 10.0*(f.tanh(norm))*h3/norm if norm > 0 else 10*h3
            return f.tanh(h3)
        
        else:
            # critic network simply outputs a number
#             x = self.bn0(x)
#             h1 = self.nonlin(self.bn1(self.fc1(x)))
            h1 = self.nonlin(self.fc1(x))
#             h2 = self.nonlin(self.bn2(self.fc2(h1)))
            h2 = self.nonlin(self.fc2(h1))
            h3 = (self.fc3(h2))

#             h3 = self.nonlin(self.bn3(self.fc3(h2)))
#             h4 = self.nonlin(self.fc4(h3))

            # Alternative arch:
            
#             state, action = torch.split(x, [48,4], dim=1) # it returns a tuple

#             h1 = f.relu(self.fc1(self.bn1(state)))

#             h1 = torch.cat((h1, action), dim=1)
#             h2 = f.relu(self.fc2(h1))
#             h3 = self.fc3(h2)
    
            return h3

