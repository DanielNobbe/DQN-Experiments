import random

import torch

def get_epsilon(it):
    # YOUR CODE HERE
    annealing_time = 1000
    progress = it/annealing_time
    
    max_eps = 1
    min_eps = 0.05
    epsilon = max(max_eps - (max_eps - min_eps) * progress, min_eps)
    
    return epsilon


class EpsilonGreedyPolicy(object):
    """
    A simple epsilon greedy policy.
    """
    def __init__(self, Q, epsilon):
        self.Q = Q
        self.epsilon = epsilon
    
    def sample_action(self, obs):
        """
        This method takes a state as input and returns an action sampled from this policy.  

        Args:
            obs: current state

        Returns:
            An action (int).
        """
        # YOUR CODE HERE
        # So we first need to choose whether we are taking a random action or a policy action
        if random.choices([True, False], weights=[self.epsilon, 1-self.epsilon], k=1)[0]:
            # This means we need to make a random choice for the action to be performed
            # The size of the output layer of Q_net is hardcoded as 2, so we will do that here too
            return random.choice(range(2))
        else:
            # This means we need to use the policy network
            obs = torch.Tensor(obs) # Stays Tensor if it was already one, becomes tensor if not
            action = torch.argmax(self.Q(obs)).item()
            return action
        
    def set_epsilon(self, epsilon):
        self.epsilon = epsilon