import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class ActorCritic(nn.Module):
    def __init__(self, observation_shape, action_shape):
        super().__init__()
        
        # We expect observation to be [x_dist, y_dist]
        input_dim = np.prod(observation_shape)
        action_dim = np.prod(action_shape)

        self.common = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh()
        )
        
        # Actor: Outputs mean action
        self.actor_mean = nn.Linear(64, action_dim)
        self.actor_logstd = nn.Parameter(torch.zeros(action_dim)) # Learnable parameter
        
        # Critic: Outputs value
        self.critic = nn.Linear(64, 1)

    def forward(self, x):
        x = self.common(x)
        mean = self.actor_mean(x)
        
        # PyTorch cleanly handles broadcasting
        logstd = self.actor_logstd
        
        value = self.critic(x)
        return mean, logstd, value
    
    def get_action_mean(self, x):
        # Fast path for inference (just get mean)
        with torch.no_grad():
            x = self.common(x)
            return self.actor_mean(x)

    def save_weights(self, path):
        torch.save(self.state_dict(), path)

    def load_weights(self, path):
        self.load_state_dict(torch.load(path))

class PPOAgent:
    def __init__(self, model, learning_rate=3e-4, gamma=0.99, eps_clip=0.2, device='cuda'):
        self.model = model.to(device)
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.device = device

    def get_action(self, obs):
        # Convert to tensor and move to device
        if isinstance(obs, np.ndarray):
            obs = torch.FloatTensor(obs).to(self.device)
        
        if len(obs.shape) == 1:
            obs = obs.unsqueeze(0)
            
        with torch.no_grad():
            mean, log_std, value = self.model(obs)
            std = torch.exp(log_std)
            
            # Sample
            dist = torch.distributions.Normal(mean, std)
            action = dist.sample()
            
            # Log Prob
            log_prob = dist.log_prob(action).sum(axis=-1)
            
        return action.cpu().numpy(), log_prob.cpu().numpy(), value.cpu().numpy()

    def update(self, memory, batch_size=64, epochs=4):
        # Unpack memory
        obs_list = torch.FloatTensor(np.array(memory['obs'])).to(self.device)
        actions_list = torch.FloatTensor(np.array(memory['actions'])).to(self.device)
        logprobs_list = torch.FloatTensor(np.array(memory['logprobs'])).to(self.device)
        
        # GAE Calculation using numpy to avoid device transfers/loops on GPU tensors one by one
        rewards_np = memory['rewards']
        dones_np = memory['dones']
        values_np = memory['values']
        
        returns_np = []
        discounted_reward = 0
        for reward, is_done in zip(reversed(rewards_np), reversed(dones_np)):
            if is_done:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            returns_np.insert(0, discounted_reward)
            
        returns = torch.FloatTensor(np.array(returns_np)).to(self.device)
        values = torch.FloatTensor(np.array(values_np)).to(self.device)
        advantages = returns - values
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO Update Loop
        dataset_size = len(obs_list)
        indices = np.arange(dataset_size)
        
        total_loss = 0
        updates = 0
        
        # Set to train mode
        self.model.train()
        
        for _ in range(epochs):
            np.random.shuffle(indices)
            for start in range(0, dataset_size, batch_size):
                end = start + batch_size
                idx = indices[start:end]
                
                # Mini-batch
                mb_obs = obs_list[idx]
                mb_actions = actions_list[idx]
                mb_old_logprobs = logprobs_list[idx]
                mb_advantages = advantages[idx]
                mb_returns = returns[idx]
                
                # Forward pass
                mean, log_std, values = self.model(mb_obs)
                std = torch.exp(log_std)
                values = values.squeeze()
                
                # New log probs
                dist = torch.distributions.Normal(mean, std)
                new_log_probs = dist.log_prob(mb_actions).sum(axis=-1)
                
                ratio = torch.exp(new_log_probs - mb_old_logprobs)
                
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * mb_advantages
                
                loss_actor = -torch.min(surr1, surr2).mean()
                loss_critic = 0.5 * ((values - mb_returns) ** 2).mean()
                
                loss = loss_actor + 0.5 * loss_critic
                
                # Optimization step
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                updates += 1
                
        return total_loss / updates
