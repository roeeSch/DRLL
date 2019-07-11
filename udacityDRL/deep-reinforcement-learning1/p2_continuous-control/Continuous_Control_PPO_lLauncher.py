import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
# import gym
import numpy as np
from unityagents import UnityEnvironment

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []

    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, n_var, action_std):
        super(ActorCritic, self).__init__()
        # action mean range -1 to 1
        self.actor = nn.Sequential(
            nn.Linear(state_dim, n_var),
            nn.Tanh(),  # was Tanh
            nn.Linear(n_var, n_var),
            nn.Tanh(),  # was Tanh
            nn.Linear(n_var, action_dim),
            nn.Tanh()
        )
        # critic
        self.critic = nn.Sequential(
            nn.Linear(state_dim, n_var),
            nn.Tanh(),  # was Tanh
            nn.Linear(n_var, n_var),
            nn.Tanh(),  # was Tanh
            nn.Linear(n_var, 1)
        )
        self.action_var = torch.full((action_dim,), action_std * action_std).to(device)

    def forward(self):
        raise NotImplementedError

    def act(self, state, memory):
        action_mean = self.actor(state)
        dist = MultivariateNormal(action_mean, torch.diag(self.action_var).to(device))
        action = dist.sample()
        action_logprob = dist.log_prob(action)

        memory.states.append(state)
        memory.actions.append(action)
        memory.logprobs.append(action_logprob)

        return action.detach()

    def evaluate(self, state, action):
        action_mean = self.actor(state)
        dist = MultivariateNormal(torch.squeeze(action_mean), torch.diag(self.action_var))

        action_logprobs = dist.log_prob(torch.squeeze(action))
        dist_entropy = dist.entropy()
        state_value = self.critic(state)

        return action_logprobs, torch.squeeze(state_value), dist_entropy


class PPO:
    def __init__(self, state_dim, action_dim, n_latent_var, action_std, lr, betas, gamma, K_epochs, eps_clip):
        self.lr = lr
        self.betas = betas
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs

        self.policy = ActorCritic(state_dim, action_dim, n_latent_var, action_std).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(),
                                          lr=lr, betas=betas)
        self.policy_old = ActorCritic(state_dim, action_dim, n_latent_var, action_std).to(device)

        self.MseLoss = nn.MSELoss()

    def select_action(self, state, memory):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.policy_old.act(state, memory).cpu().data.numpy().flatten()

    def update(self, memory):
        # Monte Carlo estimate of rewards:
        rewards = []
        discounted_reward = 0
        for reward in reversed(memory.rewards):
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalizing the rewards:
        rewards = torch.tensor(rewards).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

        # convert list to tensor
        old_states = torch.stack(memory.states).to(device).detach()
        old_actions = torch.stack(memory.actions).to(device).detach()
        old_logprobs = torch.squeeze(torch.stack(memory.logprobs)).to(device).detach()

        # Optimize policy for K epochs:
        for _ in range(self.K_epochs):
            # Evaluating old actions and values :
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

            # Finding the ratio (pi_theta / pi_theta__old):
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss:
            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - 0.01 * dist_entropy

            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        # Copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())


def main():
    ############## Hyperparameters ##############
    import time
    env_name = 'unity_reacher_1agent'
    env = UnityEnvironment(file_name='Reacher_Linux_1agent/Reacher.x86_64')
    # get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    env_info = env.reset(train_mode=False)[brain_name]

    render = False
    solved_reward = 30  # stop training if avg_reward > solved_reward
    log_interval = 10  # print avg reward in the interval
    max_episodes = 10000 #50000  # max training episodes
    max_timesteps = 600  # max timesteps in one episode
    n_latent_var = 64  # number of variables in hidden layer
    update_timestep = 4000 #4000  # update policy every n timesteps
    action_std = 0.4  # constant std for action distribution
    lr = 0.0025
    betas = (0.9, 0.999)
    gamma = 0.99  # discount factor
    K_epochs = 4 #5  # update policy for K epochs
    eps_clip = 0.2  # clip parameter for PPO
    random_seed = None
    #############################################

    # creating environment
    # env = gym.make(env_name)
    # state_dim = env.observation_space.shape[0]
    states = env_info.vector_observations
    state_dim = states.shape[1]
    # action_dim = env.action_space.shape[0]
    action_dim = brain.vector_action_space_size

    # if random_seed:
    #     print("Random Seed: {}".format(random_seed))
    #     torch.manual_seed(random_seed)
    #     env.seed(random_seed)
    #     np.random.seed(random_seed)

    memory = Memory()
    ppo = PPO(state_dim, action_dim, n_latent_var, action_std, lr, betas, gamma, K_epochs, eps_clip)
    print(lr, betas)

    # logging variables
    running_reward = 0
    avg_length = 0
    time_step = 0

    # training loop
    for i_episode in range(1, max_episodes + 1):
        env_info = env.reset(train_mode=True)[brain_name]
        state = env_info.vector_observations
        epiReward = 0
        # state = env.reset()
        for t in range(max_timesteps):
            time_step += 1
            # Running policy_old:
            action = ppo.select_action(state, memory)
            env_info = env.step(action)[brain_name]  # send all actions to tne environment
            next_state = env_info.vector_observations  # get next state (for each agent)
            reward = env_info.rewards[0]  # get reward (for each agent)
            done = env_info.local_done[0]  # see if episode finished

            # state, reward, done, _ = env.step(action)
            # Saving reward:
            memory.rewards.append(reward)

            # update if its time
            if time_step % update_timestep == 0:
                print('**', end='', flush=True)
                ppo.update(memory)
                memory.clear_memory()
                time_step = 0
            running_reward += reward
            epiReward += reward
            # if render:
            #     env.render()
            if done:
                break

        print('rwd:' + '{:.3f}'.format(epiReward) + ', ', end='', flush=True)

        avg_length += t

        # # stop training if avg_reward > solved_reward
        if running_reward > (log_interval * solved_reward):
            print("########## Solved! ##########")
            torch.save(ppo.policy.state_dict(), './PPO_Continuous_{}.pth'.format(env_name))
            break
        # logging
        if i_episode % log_interval == 0:
            print('')
            avg_length = int(avg_length / log_interval)
            running_reward = float((running_reward / log_interval))

            print('Episode {} \t Avg reward: {:.2f}'.format(i_episode, running_reward))
            running_reward = 0
            avg_length = 0

    env.close()

if __name__ == '__main__':
    main()

