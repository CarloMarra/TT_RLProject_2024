import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Normal

class Actor(torch.nn.Module):
    def __init__(self, state_space, action_space):
        """
        Initialize the Actor network.

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
        self.fc3_actor_mean = torch.nn.Linear(self.hidden, action_space)
        
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
                torch.nn.init.orthogonal_(m.weight)
                torch.nn.init.zeros_(m.bias)

    def forward(self, x_state):
        """
        Forward pass through the Actor network.

        Args:
            x_state (Tensor): Input state.

        Returns:
            Normal: Action distribution.
        """
        x_actor = self.tanh(self.fc1_actor(x_state))
        x_actor = self.tanh(self.fc2_actor(x_actor))
        action_mean = self.fc3_actor_mean(x_actor)

        sigma = self.sigma_activation(self.sigma)
        normal_dist = Normal(action_mean, sigma)
        return normal_dist

class Critic(torch.nn.Module):
    def __init__(self, state_space):
        """
        Initialize the Critic network.

        Args:
            state_space (int): Dimension of the state space.
        """
        super().__init__()
        self.state_space = state_space
        self.hidden = 64
        self.tanh = torch.nn.Tanh()

        # Critic network layers
        self.fc1_critic = torch.nn.Linear(state_space, self.hidden)
        self.fc2_critic = torch.nn.Linear(self.hidden, self.hidden)
        self.fc3_critic_value = torch.nn.Linear(self.hidden, 1)

        self.init_weights()

    def init_weights(self):
        """
        Initialize the weights of the network.
        """
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.orthogonal_(m.weight)
                torch.nn.init.zeros_(m.bias)

    def forward(self, x_state):
        """
        Forward pass through the Critic network.

        Args:
            x_state (Tensor): Input state.

        Returns:
            Tensor: State value.
        """
        x_state = self.tanh(self.fc1_critic(x_state))
        x_state = self.tanh(self.fc2_critic(x_state))
        value = self.fc3_critic_value(x_state)
        return value

class Agent(object):
    def __init__(self, actor, critic, device='cuda'):
        """
        Initialize the Agent.

        Args:
            actor (Actor): Actor network.
            critic (Critic): Critic network.
            device (str): Device to run the computations on.
        """
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
        """
        Update the policy and value networks based on collected experiences.
        """
        action_log_probs = torch.stack(self.action_log_probs, dim=0).to(self.train_device).squeeze(-1)
        states = torch.stack(self.states, dim=0).to(self.train_device).squeeze(-1)
        actions = torch.stack(self.actions, dim=0).to(self.train_device).squeeze(-1)
        next_states = torch.stack(self.next_states, dim=0).to(self.train_device).squeeze(-1)
        rewards = torch.stack(self.rewards, dim=0).to(self.train_device).squeeze(-1)
        done = torch.Tensor(self.done).to(self.train_device)
        
        # Clear buffers
        self.states, self.actions, self.next_states, self.action_log_probs, self.rewards, self.done = [], [], [], [], [], []

        # Compute state values from critic
        values = self.critic(states).squeeze(-1)
        
        
        next_values = self.critic(next_states).squeeze(-1)

        # Compute target values using Bellman equation
        target_values = rewards + self.gamma * next_values * (1 - done)

        # Compute advantage
        advantages = target_values - values

        # Normalize the advantages
        # advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Compute critic loss
        critic_loss = torch.mean((advantages) ** 2)

        # Compute actor loss
        actor_loss = -torch.mean(action_log_probs * advantages.detach())

        # Update actor network
        self.optimizerA.zero_grad()
        actor_loss.backward()
        self.optimizerA.step()

        # Update critic network
        self.optimizerC.zero_grad()
        critic_loss.backward()
        self.optimizerC.step()
        
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
        normal_dist = self.actor(x)
        
        if evaluation:  # Return mean action during evaluation
            return normal_dist.mean, None
        else:  # Sample from the distribution during training
            action = normal_dist.sample()
            # Compute log probability of the action
            action_log_prob = normal_dist.log_prob(action).sum()
            return action, action_log_prob

    def store_outcome(self, state, action, next_state, action_log_prob, reward, done):
        """
        Store the outcome of an action.

        Args:
            state (numpy.array): Current state.
            action (Tensor): Action taken.
            next_state (numpy.array): Next state.
            action_log_prob (Tensor): Log probability of the action.
            reward (float): Reward received.
            done (bool): Whether the episode is done.
        """
        self.states.append(torch.from_numpy(state).float())
        self.actions.append(action)
        self.next_states.append(torch.from_numpy(next_state).float())
        self.action_log_probs.append(action_log_prob)
        self.rewards.append(torch.Tensor([reward]))
        self.done.append(done)