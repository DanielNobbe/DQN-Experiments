import numpy as np
import random
import torch
from torch import nn
import torch.nn.functional as F
from torch import optim
from tqdm import tqdm as _tqdm
import gym

from DQN_model import QNetwork
from DQN_replay import ReplayMemory
from DQN_policy import EpsilonGreedyPolicy, get_epsilon
from DQN_training import train
from DQN_plots import plot_smooth

# Note sure if necessary TODO
def tqdm(*args, **kwargs):
    return _tqdm(*args, **kwargs, mininterval=1)  # Safety, do not overflow buffer

def run_episodes(train, Q, policy, memory, env, num_episodes, batch_size, discount_factor, learn_rate):
    
    optimizer = optim.Adam(Q.parameters(), learn_rate)
    
    global_steps = 0  # Count the steps (do not reset at episode start, to compute epsilon)
    episode_durations = []  
    for i in range(num_episodes):
        state = env.reset()
        
        steps = 0
        while True:
            # So it seems like here we should sample an episode,
            # and every step update the weights
            
            # So first sample an action
            sampled_action = policy.sample_action(state)
            
            # Then step 
            state_tuple = env.step(sampled_action)
            
            # Store this transition in memory:
            s_next, r, done, _ = state_tuple
            memory.push((state, sampled_action, r, s_next, done))
            state = s_next
            
            # Now that we have added a transition, we should try to train based on our memory
            loss = train(Q, memory, optimizer, batch_size, discount_factor)
            # This is like online learning, we could also only train once per episode
            
            steps += 1
            global_steps += 1
            
            # Update epsilon
            policy.set_epsilon(get_epsilon(global_steps))
            
            if done:
                if i % 10 == 0:
                    print("{2} Episode {0} finished after {1} steps"
                          .format(i, steps, '\033[92m' if steps >= 195 else '\033[99m'))
                    print("epsilon: ", policy.epsilon)
                episode_durations.append(steps)
                #plot_durations()
                break
    return episode_durations

def main():
    print("Running DQN")

    env = gym.envs.make("CartPole-v1")

    num_episodes = 1000
    batch_size = 64
    discount_factor = 0.8
    learn_rate = 1e-3
    memory = ReplayMemory(50000)
    num_hidden = 128
    seed = 48  # This is not randomly chosen

    # We will seed the algorithm (before initializing QNetwork!) for reproducibility
    random.seed(seed)
    torch.manual_seed(seed)
    env.seed(seed)

    Q_net = QNetwork(num_hidden)
    policy = EpsilonGreedyPolicy(Q_net, 0.05)
    episode_durations = run_episodes(train, Q_net, policy, memory, env, num_episodes, batch_size, discount_factor, learn_rate)

    plot_smooth(episode_durations, 10)


if __name__=="__main__":
    main()