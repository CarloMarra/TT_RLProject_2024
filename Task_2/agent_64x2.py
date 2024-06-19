import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Normal

def discount_rewards(r, gamma):
    """
    Compute discounted rewards.

    Args:
        r (Tensor): Rewards.
        gamma (float): Discount factor.

    Returns:
        Tensor: Discounted rewards.
    """
    discounted_r = torch.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size(-1))):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r

class Policy(torch.nn.Module):
    def __init__(self, state_space, action_space):
        """
        Initialize the Policy network.

        Args:
            state_space (int): Dimension of the state space.
            action_space (int): Dimension of the action space.
        """
        super().__init__()
        self.state_space = state_space
        self.action_space = action_space
        self.hidden = 64
        self.tanh = torch.nn.Tanh()

        # Actor network layers
        self.fc1_actor = torch.nn.Linear(state_space, self.hidden)
        self.fc2_actor = torch.nn.Linear(self.hidden, self.hidden)
        self.fc3_actor = torch.nn.Linear(self.hidden, self.hidden)
        self.fc4_actor_mean = torch.nn.Linear(self.hidden, action_space)
        
        # Learned standard deviation for exploration at training time
        self.sigma_activation = F.softplus
        init_sigma = 0.5
        self.sigma = torch.nn.Parameter(torch.zeros(self.action_space) + init_sigma)

        self.init_weights()

    def init_weights(self):
        """
        Initialize the weights of the network.
        """
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.normal_(m.weight)
                torch.nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        Forward pass through the network to obtain the action distribution.

        Args:
            x (Tensor): Input state.

        Returns:
            Normal: Action distribution.
        """
        x_actor = self.tanh(self.fc1_actor(x))
        x_actor = self.tanh(self.fc2_actor(x_actor))
        x_actor = self.tanh(self.fc3_actor(x_actor))
        action_mean = self.fc4_actor_mean(x_actor)

        sigma = self.sigma_activation(self.sigma)
        normal_dist = Normal(action_mean, sigma)

        return normal_dist


class Agent(object):
    def __init__(self, policy, device='cpu'):
        """
        Initialize the Agent.

        Args:
            policy (Policy): Policy network.
            device (str): Device to run the computations on.
        """
        self.train_device = device
        self.policy = policy.to(self.train_device)
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)

        self.gamma = 0.99
        self.states = []
        self.next_states = []
        self.action_log_probs = []
        self.rewards = []
        self.done = []

    def update_policy(self):
        """
        Update the policy network based on collected experiences.
        """
        action_log_probs = torch.stack(self.action_log_probs, dim=0).to(self.train_device).squeeze(-1)
        states = torch.stack(self.states, dim=0).to(self.train_device).squeeze(-1)
        next_states = torch.stack(self.next_states, dim=0).to(self.train_device).squeeze(-1)
        rewards = torch.stack(self.rewards, dim=0).to(self.train_device).squeeze(-1)
        done = torch.Tensor(self.done).to(self.train_device)

        # Clear stored experiences
        self.states, self.next_states, self.action_log_probs, self.rewards, self.done = [], [], [], [], []

        # Compute discounted returns
        discounted_rewards = discount_rewards(rewards, self.gamma)
        
        baseline = 70
        advantages = discounted_rewards - baseline
        policy_loss = -torch.sum(action_log_probs * advantages)
        
        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()

    def get_action(self, state, evaluation=False):
        """
        Get action from the policy network given the current state.

        Args:
            state (numpy.array): Current state.
            evaluation (bool): If true, return the mean action.

        Returns:
            tuple: Action and action log probability.
        """
        x = torch.from_numpy(state).float().to(self.train_device)
        normal_dist = self.policy(x)

        if evaluation:  # Return mean action during evaluation
            return normal_dist.mean, None
        else:  # Sample from the distribution during training
            action = normal_dist.sample()
            action_log_prob = normal_dist.log_prob(action).sum()
            return action, action_log_prob

    def store_outcome(self, state, next_state, action_log_prob, reward, done):
        """
        Store the outcome of an action.

        Args:
            state (numpy.array): Current state.
            next_state (numpy.array): Next state.
            action_log_prob (Tensor): Log probability of the action.
            reward (float): Reward received.
            done (bool): Whether the episode is done.
        """
        self.states.append(torch.from_numpy(state).float())
        self.next_states.append(torch.from_numpy(next_state).float())
        self.action_log_probs.append(action_log_prob)
        self.rewards.append(torch.Tensor([reward]))
        self.done.append(done)