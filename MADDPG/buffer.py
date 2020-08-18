from collections import deque
import random
from utilities import transpose_list


class ReplayBuffer:
    """Implements Replay Buffer"""
    def __init__(self,size):
        """Initialize parameters and build replay buffer.
        Params
        ======
            size (int): Size of the replay buffer
        """
        self.size = size
        self.deque = deque(maxlen=self.size)

    def push(self,transition):
        """push into the buffer"""
        
        # To Do: Generalize to parallel environment
        self.deque.append(transition)
#         input_to_buffer = transpose_list(transition)
        
#         for item in input_to_buffer:
#             self.deque.append(item)

    def sample(self, batchsize):
        """sample from the buffer"""
        samples = random.sample(self.deque, batchsize)

        # transpose list of list
        return transpose_list(samples)

    def __len__(self):
        """returns length of the buffer"""
        return len(self.deque)



