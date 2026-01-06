import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
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
        self.actor_logstd = mx.zeros((action_dim,)) # Learnable parameter
        
        # Critic: Outputs value
        self.critic = nn.Linear(64, 1)

    def __call__(self, x):
        x = self.common(x)
        mean = self.actor_mean(x)
        
        # MLX clean way to handle logstd parameter
        logstd = self.actor_logstd
        
        value = self.critic(x)
        return mean, logstd, value
    
    def get_action_mean(self, x):
        # Fast path for inference (just get mean)
        x = self.common(x)
        return self.actor_mean(x)

class PPOAgent:
    def __init__(self, model, learning_rate=3e-4, gamma=0.99, eps_clip=0.2):
        self.model = model
        self.optimizer = optim.Adam(learning_rate=learning_rate)
        self.gamma = gamma
        self.eps_clip = eps_clip

    def get_action(self, obs):
        obs = mx.array(obs)
        if obs.ndim == 1:
            obs = obs[None, :]
            
        mean, log_std, value = self.model(obs)
        std = mx.exp(log_std)
        
        # Sample
        noise = mx.random.normal(mean.shape)
        action = mean + noise * std
        
        # Log Prob
        # log_prob = -0.5 * ((action - mean) / std)**2 - log_std - 0.5 * log(2pi)
        log_prob = -0.5 * ((action - mean) / std) ** 2 - log_std - 0.5 * np.log(2. * np.pi)
        log_prob = mx.sum(log_prob, axis=-1)
        
        return action, log_prob, value

    def update(self, memory, batch_size=64, epochs=4):
        # Unpack memory
        obs_list = mx.array(np.array(memory['obs']))
        actions_list = mx.array(np.array(memory['actions']))
        logprobs_list = mx.array(np.array(memory['logprobs']))
        rewards_list = np.array(memory['rewards'])
        dones_list = np.array(memory['dones'])
        values_list = np.array(memory['values']).flatten()
        
        # Compute GAE / Returns
        returns = []
        discounted_reward = 0
        for reward, is_done in zip(reversed(rewards_list), reversed(dones_list)):
            if is_done:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            returns.insert(0, discounted_reward)
            
        returns = mx.array(np.array(returns))
        advantages = returns - mx.array(values_list)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO Update Loop
        indices = np.arange(len(obs_list))
        
        total_loss = 0
        updates = 0
        
        for _ in range(epochs):
            np.random.shuffle(indices)
            for start in range(0, len(obs_list), batch_size):
                end = start + batch_size
                idx = mx.array(indices[start:end])
                
                # Mini-batch
                mb_obs = obs_list[idx]
                mb_actions = actions_list[idx]
                mb_old_logprobs = logprobs_list[idx]
                mb_advantages = advantages[idx]
                mb_returns = returns[idx]
                
                def loss_fn(params):
                    self.model.update(params)
                    mean, log_std, values = self.model(mb_obs)
                    std = mx.exp(log_std)
                    values = values.squeeze()
                    
                    # New log probs
                    new_log_probs = -0.5 * ((mb_actions - mean) / std) ** 2 - log_std - 0.5 * np.log(2. * np.pi)
                    new_log_probs = mx.sum(new_log_probs, axis=-1)
                    
                    ratio = mx.exp(new_log_probs - mb_old_logprobs)
                    
                    surr1 = ratio * mb_advantages
                    surr2 = mx.clip(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * mb_advantages
                    
                    loss_actor = -mx.minimum(surr1, surr2).mean()
                    loss_critic = 0.5 * ((values - mb_returns) ** 2).mean()
                    
                    return loss_actor + 0.5 * loss_critic

                loss, grads = mx.value_and_grad(loss_fn)(self.model.parameters())
                self.optimizer.update(self.model, grads)
                
                total_loss += loss.item()
                updates += 1
                
        return total_loss / updates
