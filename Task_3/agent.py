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


class Policy(torch.nn.Module):
    def __init__(self, state_space, action_space):
        super().__init__()
        self.state_space = state_space
        self.action_space = action_space
        self.hidden = 64
        self.tanh = torch.nn.Tanh()

        """
            Actor network
        """
        self.fc1_actor = torch.nn.Linear(state_space, self.hidden)
        self.fc2_actor = torch.nn.Linear(self.hidden, self.hidden)
        self.fc3_actor_mean = torch.nn.Linear(self.hidden, action_space)
        
        # Learned standard deviation for exploration at training time
        self.sigma_activation = F.softplus
        init_sigma = 0.5
        self.sigma = torch.nn.Parameter(torch.zeros(self.action_space)+init_sigma)


        """
            Critic network
        """
        # TASK 3: critic network for actor-critic algorithm
        self.fc1_critic = torch.nn.Linear(state_space + action_space, self.hidden)
        self.fc2_critic = torch.nn.Linear(self.hidden, self.hidden)
        self.fc3_critic_value = torch.nn.Linear(self.hidden, 1)

        
        self.init_weights()


    def init_weights(self):
        for m in self.modules():
            if type(m) is torch.nn.Linear:
                torch.nn.init.normal_(m.weight)
                torch.nn.init.zeros_(m.bias)


    def forward(self, x_state, x_action=None):
        """
            Actor
        """
        x_actor = self.tanh(self.fc1_actor(x_state))
        x_actor = self.tanh(self.fc2_actor(x_actor))
        action_mean = self.fc3_actor_mean(x_actor)

        sigma = self.sigma_activation(self.sigma)
        normal_dist = Normal(action_mean, sigma)

         # TASK 3: forward in the critic network
        if x_action is None:
            return normal_dist
        else:
            """
                Critic
            """
            x_state_action = torch.cat([x_state, x_action], dim=-1)
            x_state_action = self.tanh(self.fc1_critic(x_state_action))
            x_state_action = self.tanh(self.fc2_critic(x_state_action))
            q_value = self.fc3_critic_value(x_state_action)
            return normal_dist, q_value


class Agent(object):
    def __init__(self, policy, device='cuda'):
        self.train_device = device
        self.policy = policy.to(self.train_device)
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)

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

        self.states, self.actions, self.next_states, self.action_log_probs, self.rewards, self.done = [], [], [], [], [], []
        
        # TASK 3:
        #   - compute boostrapped discounted return estimates
        #   - compute advantage terms
        #   - compute actor loss and critic loss
        #   - compute gradients and step the optimizer
        
        _ , state_action_values = self.policy(states, actions)
        state_action_values = state_action_values.view(-1, 1)  # Ensure shape is [batch_size, 1]

        with torch.no_grad():
            next_actions_distr = self.policy(next_states)
            next_actions = next_actions_distr.mean
            _, next_state_values = self.policy(next_states, next_actions)
            next_state_values = next_state_values.view(-1) * (1 - done)  # Zero out values for terminal states
        
        # Compute target Q-values [from Bellman's equations]
        target_values = rewards + self.gamma * next_state_values
        target_values = target_values.view(-1, 1)  # Ensure shape is [batch_size, 1]

        # Compute advantages
        advantages = target_values - state_action_values
        
        # Compute loss
        policy_loss = -torch.sum(action_log_probs * advantages)
        
        # Compute critic loss
        critic_loss = F.mse_loss(state_action_values, target_values)
        
        # Total loss
        loss = policy_loss + critic_loss
    
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return        


    def get_action(self, state, evaluation=False):
        """ state -> action (3-d), action_log_densities """
        x = torch.from_numpy(state).float().to(self.train_device)

        normal_dist = self.policy(x)

        if evaluation:  # Return mean
            return normal_dist.mean, None

        else:   # Sample from the distribution
            action = normal_dist.sample()

            # Compute Log probability of the action [ log(p(a[0] AND a[1] AND a[2])) = log(p(a[0])*p(a[1])*p(a[2])) = log(p(a[0])) + log(p(a[1])) + log(p(a[2])) ]
            action_log_prob = normal_dist.log_prob(action).sum()

            return action, action_log_prob


    def store_outcome(self, state, action, next_state, action_log_prob, reward, done):
        self.states.append(torch.from_numpy(state).float())
        self.actions.append(action)
        self.next_states.append(torch.from_numpy(next_state).float())
        self.action_log_probs.append(action_log_prob)
        self.rewards.append(torch.Tensor([reward]))
        self.done.append(done)

