import numpy as np
import random
import torch
from torch import nn
import torch.nn.functional as F
from torch import optim
from tqdm import tqdm as _tqdm
import gym
import argparse
import copy

from DQN_model import QNetwork
from DQN_replay import ReplayMemory
from DQN_policy import EpsilonGreedyPolicy, get_epsilon
from DQN_training import train
from DQN_plots import plot_smooth


# Code structure based on lab4 from Reinforcement Learning course at University of Amsterdam

# Note sure if necessary TODO
def tqdm(*args, **kwargs):
    return _tqdm(*args, **kwargs, mininterval=1)  # Safety, do not overflow buffer

def run_episodes(train, Q, policy, memory, env, num_episodes, batch_size, discount_factor, learn_rate, clone_interval):
    
    optimizer = optim.Adam(Q.parameters(), learn_rate)

    if clone_interval is not None:
        target_network = copy.deepcopy(Q)
    else:
        target_network = Q

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
            loss = train(Q, memory, optimizer, batch_size, discount_factor, target_network)
            # This is like online learning, we could also only train once per episode
            
            steps += 1
            global_steps += 1
            
            # Update epsilon
            policy.set_epsilon(get_epsilon(global_steps))
            
            if clone_interval is not None:
                if global_steps % clone_interval == 0:
                    print("Updating target network")
                    target_network = copy.deepcopy(Q)

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

    num_episodes = config.n_episodes
    batch_size = config.batch_size
    discount_factor = config.discount_factor
    learn_rate = config.learn_rate

    if config.memory_size is None:
        memory_size = 10*batch_size
    else:
        memory_size = config.memory_size



    memory = ReplayMemory(memory_size)
    num_hidden = 128
    seed = 48  # This is not randomly chosen

    # We will seed the algorithm (before initializing QNetwork!) for reproducibility
    random.seed(seed)
    torch.manual_seed(seed)
    env.seed(seed)

    Q_net = QNetwork(num_hidden)
    policy = EpsilonGreedyPolicy(Q_net, 0.05)
    episode_durations = run_episodes(train, Q_net, policy, memory, env, num_episodes, batch_size, discount_factor, learn_rate, config.clone_interval)

    plot_smooth(episode_durations, 10)


if __name__=="__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--n_episodes', '-ne', type=int, default=100, help="Number of episodes to train model.")
    parser.add_argument('--batch_size', '-bs', type=int, default=64, help="Number of historical states to batch train with for each present state.")
    parser.add_argument('--discount_factor', '-df', type=float, default=0.8, help="Discount factor for TD target computation.")
    parser.add_argument('--learn_rate', '-lr', type=float, default=1e-3, help="Learning rate for parameter updates.")
    parser.add_argument('--memory_size', '-ms', type=int, default=10000, help="Number of historical states to keep in memory")
    parser.add_argument('--num_hidden', '-nh', type=int, default=128, help="Hidden layer size.")
    parser.add_argument('--seed', '-s', type=int, default=42, help="Random seed number.")
    parser.add_argument('--env', '-e', type=str, default="CartPole-v1", help="Environment name in gym library for chosen environment.") 
    parser.add_argument('--clone_interval', '-tn', type=int, default=None, help="Clone interval for target network updating. If not defined, target network is updated every step.")
    # TODO: Maybe set up something for custom environments
    config = parser.parse_args()

    main()