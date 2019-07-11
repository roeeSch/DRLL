# Reacher - PPO
# Import packages

import argparse
import pickle
from scipy import signal

import tqdm
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from unityagents import UnityEnvironment
from IPython.display import clear_output

from model import PPOPolicyNetwork
from agent import PPOAgent

## Create Unity environment
env = UnityEnvironment(file_name="Reacher_Linux_multAgents/Reacher.x86_64")
# env = UnityEnvironment(file_name="Reacher_Linux_1agent/Reacher.x86_64")
brain_name = env.brain_names[0]
brain = env.brains[brain_name]
env_info = env.reset(train_mode=True)[brain_name]

# Set configurations:
config = {
    'environment': {
        'state_size':  env_info.vector_observations.shape[1],
        'action_size': brain.vector_action_space_size,
        'number_of_agents': len(env_info.agents)
    },
    'pytorch': {
        'device': torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    },
    'hyperparameters': {
        'discount_rate': 0.99,
        'tau': 0.95,
        'gradient_clip': 5,
        'rollout_length': 2048,
        'optimization_epochs': 10,
        'ppo_clip': 0.2,
        'log_interval': 2048,
        'max_steps': 1e5,
        'mini_batch_number': 32,
        'entropy_coefficent': 0.01,
        'episode_count': 250*6,
        'hidden_size': 512,
        'adam_learning_rate': 3e-4,
        'adam_epsilon': 1e-5
    }
}


def play_round(env, brain_name, policy, config, train=True):
    env_info = env.reset(train_mode=train)[brain_name]
    states = env_info.vector_observations                 
    scores = np.zeros(config['environment']['number_of_agents'])                         
    while True:
        actions, _, _, _ = policy(states)
        env_info = env.step(actions.cpu().detach().numpy())[brain_name]
        next_states = env_info.vector_observations         
        rewards = env_info.rewards                         
        dones = env_info.local_done                     
        scores += env_info.rewards                      
        states = next_states                               
        if np.any(dones):                                  
            break
    
    return np.mean(scores)


def ppo(env, brain_name, policy, config, train):
    if train:
        optimizier = optim.Adam(policy.parameters(), config['hyperparameters']['adam_learning_rate'], 
                        eps=config['hyperparameters']['adam_epsilon'])
        agent = PPOAgent(env, brain_name, policy, optimizier, config)
        all_scores = []
        averages = []
        last_max = 30.0

        for i in tqdm.tqdm(range(config['hyperparameters']['episode_count'])):
            agent.step()
            last_mean_reward = play_round(env, brain_name, policy, config)
            if i == 0:
                last_average = last_mean_reward
            else:
                last_average = np.mean(np.array(all_scores[-100:])) if len(all_scores) > 100 else np.mean(np.array(all_scores))

            all_scores.append(last_mean_reward)
            averages.append(last_average)
            if last_average > last_max:
                torch.save(policy.state_dict(), f"reacher-ppo/models/ppo-max-hiddensize-{config['hyperparameters']['hidden_size']}.pth")
                last_max = last_average
            clear_output(True)
            print('Episode: {} Total score this episode: {} Last {} average: {}'.format(i + 1, last_mean_reward, min(i + 1, 100), last_average))
        return all_scores, averages
    else:
        all_scores = []
        for i in range(20):
            score = play_round(env, brain_name, policy, config, train)
            all_scores.append(score)
        return [score], [np.mean(score)]


def arg_parse():
    parser = argparse.ArgumentParser(description='Args for Reacher - PPO')
    parser.add_argument('--learn', help='learn new policy', action='store_true', default=False,
                        dest='learnNewPolicy')
    parser.add_argument('--pltLrn', help='plot progress of previous learning session', action='store_true', default=False,
                        dest='pltLrn')
    parser.add_argument('--playRound', help='play round of last learned policy', action='store_true', default=True,
                        dest='playRound')

    return parser.parse_args()

if __name__=='__main__':

    args = arg_parse()

    if args.learnNewPolicy:
        print("Learning new policy...")
        config['hyperparameters']['episode_count'] = 300
        new_policy = PPOPolicyNetwork(config)
        all_scores, average_scores = ppo(env, brain_name, new_policy, config, train=True)

        dict_results = {'all_scores': all_scores, 'average_scores': average_scores}
        filename = 'results.pckl'
        with open(filename, 'wb') as outfile:
            pickle.dump(dict_results, outfile)

        plt.figure()
        plt.plot(all_scores)
        plt.plot(average_scores)
        plt.legend(('current score', 'average 100 epi'))
        plt.xlabel('episode')
        plt.savefig('learning_rates.png')
        plt.show()

    if args.pltLrn:
        with open('results.pckl', 'rb') as fid:
            dictRes = pickle.load(fid)
        ff = lambda x, n: signal.filtfilt(np.ones(n), float(n), x)
        plt.figure(2)
        plt.plot(dictRes['all_scores'])
        plt.plot(ff(dictRes['all_scores'], 20))
        plt.legend(('epi score', 'avg 20 epi'))
        plt.show()

    if args.playRound:
        policy = PPOPolicyNetwork(config)
        policy.load_state_dict(torch.load('reacher-ppo/models/ppo-max-hiddensize-512_score38.pth'))
        _, _ = ppo(env, brain_name, policy, config, train=False)

    # if False:
    #     policy = PPOPolicyNetwork(config)
    #     policy.load_state_dict(torch.load('reacher-ppo/models/ppo-max-hiddensize-512.pth'))
    #     config['hyperparameters']['episode_count']=30
    #     all_scores, average_scores = ppo(env, brain_name, policy, config, train=True)
    #     print("")
