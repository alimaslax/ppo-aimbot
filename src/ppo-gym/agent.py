
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

class PPOAgent:
    def __init__(self, model, learning_rate=3e-4, gamma=0.99, gae_lambda=0.95, clip_coef=0.2, ent_coef=0.01, vf_coef=0.5):
        self.model = model
        self.optimizer = optim.Adam(learning_rate=learning_rate)
        
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_coef = clip_coef
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef

    def get_action(self, x):
        # x is (batch, input_dim) or (input_dim,)
        if isinstance(x, np.ndarray):
            x = mx.array(x)
        if x.ndim == 1:
            x = x[None, :]
            
        mean, log_std, value = self.model(x)
        
        # Sample from Gaussian
        std = mx.exp(log_std)
        noise = mx.random.normal(mean.shape)
        action = mean + noise * std
        
        # Calculate log prob of this action
        # log_prob = -0.5 * ((action - mean) / std)**2 - log_std - 0.5 * log(2pi)
        # Sum over action dim
        log_prob = -0.5 * ((action - mean) / std) ** 2 - log_std - 0.5 * np.log(2. * np.pi)
        log_prob = mx.sum(log_prob, axis=-1)
        
        return action, log_prob, value

    def compute_gae(self, rewards, values, next_value, dones):
        # rewards: (T,)
        # values: (T,)
        # next_value: scalar or (1,)
        # dones: (T,)
        
        # Convert to numpy for loop efficiency if needed, or use mx if vectorized.
        # Let's use numpy for GAE calculation to be safe and standard, then convert back.
        rewards = np.array(rewards)
        values = np.array(values)
        dones = np.array(dones)
        
        returns = np.zeros_like(rewards)
        advantages = np.zeros_like(rewards)
        
        next_val = next_value
        last_gae_lam = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                nextnonterminal = 1.0 - 0.0 # Assuming last step is not terminal for bootstrapping usually? 
                # If done[t] is true, then next value should be 0 (masked). 
                # But here we pass 'dones' which are booleans of whether step t was terminal.
                nextnonterminal = 1.0 - dones[t]
                nextvalues = next_val
            else:
                nextnonterminal = 1.0 - dones[t]
                nextvalues = values[t + 1]
                
            delta = rewards[t] + self.gamma * nextvalues * nextnonterminal - values[t]
            last_gae_lam = delta + self.gamma * self.gae_lambda * nextnonterminal * last_gae_lam
            advantages[t] = last_gae_lam
            
        returns = advantages + values
        return mx.array(returns), mx.array(advantages)

    def loss(self, params, states, actions, old_log_probs, returns, advantages):
        # Update model parameters temporarily for this function
        self.model.update(params)
        
        mean, log_std, values = self.model(states)
        values = values.squeeze()
        
        # New Log Probs
        std = mx.exp(log_std)
        new_log_probs = -0.5 * ((actions - mean) / std) ** 2 - log_std - 0.5 * np.log(2. * np.pi)
        new_log_probs = mx.sum(new_log_probs, axis=-1)
        
        # Ratio
        ratio = mx.exp(new_log_probs - old_log_probs)
        
        # Policy Loss
        pg_loss1 = -advantages * ratio
        pg_loss2 = -advantages * mx.clip(ratio, 1.0 - self.clip_coef, 1.0 + self.clip_coef)
        pg_loss = mx.maximum(pg_loss1, pg_loss2).mean()
        
        # Value Loss
        v_loss = 0.5 * ((values - returns) ** 2).mean()
        
        # Entropy Loss (Bonus)
        entropy = log_std + 0.5 + 0.5 * np.log(2 * np.pi)
        entropy = entropy.sum(-1).mean()
        ent_loss = -entropy * self.ent_coef
        
        total_loss = pg_loss + v_loss * self.vf_coef + ent_loss
        return total_loss

    def update(self, states, actions, old_log_probs, returns, advantages):
        # Perform one update step
        # Create function that computes loss and gradients
        loss_and_grad_fn = mx.value_and_grad(self.loss)
        
        # Compute loss and gradients
        loss_value, grads = loss_and_grad_fn(self.model.parameters(), states, actions, old_log_probs, returns, advantages)
        
        # Optimize
        self.optimizer.update(self.model, grads)
        
        return loss_value
