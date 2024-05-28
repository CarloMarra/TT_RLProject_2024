import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Normal

def discount_rewards(r, gamma):
    discounted_r = torch.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size(-1))):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r

class Actor(torch.nn.Module):
    def __init__(self, state_space, action_space):
        super().__init__()
        self.state_space = state_space
        self.action_space = action_space
        self.hidden = 64
        self.tanh = torch.nn.Tanh()

        self.fc1_actor = torch.nn.Linear(state_space, self.hidden)
        self.fc2_actor = torch.nn.Linear(self.hidden, self.hidden)
        self.fc3_actor_mean = torch.nn.Linear(self.hidden, action_space)
        
        # Learned standard deviation for exploration at training time
        self.sigma_activation = F.softplus
        init_sigma = 0.5
        self.sigma = torch.nn.Parameter(torch.zeros(self.action_space) + init_sigma)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.normal_(m.weight)
                torch.nn.init.zeros_(m.bias)

    def forward(self, x_state):
        """
            Actor forward pass
        """
        x_actor = self.tanh(self.fc1_actor(x_state))
        x_actor = self.tanh(self.fc2_actor(x_actor))
        action_mean = self.fc3_actor_mean(x_actor)

        sigma = self.sigma_activation(self.sigma)
        normal_dist = Normal(action_mean, sigma)
        return normal_dist

class Critic(torch.nn.Module):
    def __init__(self, state_space, action_space):
        super().__init__()
        self.state_space = state_space
        self.action_space = action_space
        self.hidden = 64
        self.tanh = torch.nn.Tanh()

        self.fc1_critic = torch.nn.Linear(state_space + action_space, self.hidden)
        self.fc2_critic = torch.nn.Linear(self.hidden, self.hidden)
        self.fc3_critic_value = torch.nn.Linear(self.hidden, 1)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.normal_(m.weight)
                torch.nn.init.zeros_(m.bias)

    def forward(self, x_state, x_action):
        """
            Critic forward pass
        """
        x_state_action = torch.cat([x_state, x_action], dim=-1)
        x_state_action = self.tanh(self.fc1_critic(x_state_action))
        x_state_action = self.tanh(self.fc2_critic(x_state_action))
        q_value = self.fc3_critic_value(x_state_action)
        return q_value

class Agent(object):
    
    def __init__(self, actor, critic, device='cuda'):
        self.train_device = device
        self.actor = actor.to(self.train_device)
        self.critic = critic.to(self.train_device)
        self.optimizerA = torch.optim.Adam(actor.parameters(), lr=1e-3)
        self.optimizerC = torch.optim.Adam(critic.parameters(), lr=1e-3)
        self.gamma = 0.99
        self.states = []
        self.next_states = []
        self.actions = []
        self.action_log_probs = []
        self.rewards = []
        self.done = []

    def update_policy(self):
        
        action_log_probs = torch.stack(self.action_log_probs, dim=0).to(self.train_device).squeeze(-1)
        states = torch.stack(self.states, dim=0).to(self.train_device).squeeze(-1)
        actions = torch.stack(self.actions, dim=0).to(self.train_device).squeeze(-1)
        next_states = torch.stack(self.next_states, dim=0).to(self.train_device).squeeze(-1)
        rewards = torch.stack(self.rewards, dim=0).to(self.train_device).squeeze(-1)
        done = torch.Tensor(self.done).to(self.train_device)
        
        # Clear buffers
        self.states, self.actions, self.next_states, self.action_log_probs, self.rewards, self.done = [], [], [], [], [], []

        # Compute discounted rewards
        discounted_rewards = discount_rewards(rewards, self.gamma)

        # Compute state-action values from critic
        q_values = self.critic(states, actions).squeeze(-1)
        next_q_values = self.critic(next_states, actions).squeeze(-1)

        # Compute target Q values using Bellman equation
        target_q_values = rewards + self.gamma * next_q_values * (1 - done)

        # Compute advantage
        advantages = target_q_values - q_values

        # Compute critic loss
        critic_loss = F.mse_loss(q_values, target_q_values.detach())

        # Compute actor loss
        actor_loss = -(action_log_probs * advantages.detach()).mean()

        # Update actor network
        self.optimizerA.zero_grad()
        actor_loss.backward()
        self.optimizerA.step()

        # Update critic network
        self.optimizerC.zero_grad()
        critic_loss.backward()
        self.optimizerC.step()
        
    def get_action(self, state, evaluation=False):
        """ state -> action (3-d), action_log_densities """
        x = torch.from_numpy(state).float().to(self.train_device)

        normal_dist = self.actor(x)

        if evaluation:  # Return mean
            return normal_dist.mean, None

        else:   # Sample from the distribution
            action = normal_dist.sample()

            # Compute Log probability of the action
            action_log_prob = normal_dist.log_prob(action).sum()

            return action, action_log_prob

    def store_outcome(self, state, action, next_state, action_log_prob, reward, done):
        self.states.append(torch.from_numpy(state).float())
        self.actions.append(action)
        self.next_states.append(torch.from_numpy(next_state).float())
        self.action_log_probs.append(action_log_prob)
        self.rewards.append(torch.Tensor([reward]))
        self.done.append(done)