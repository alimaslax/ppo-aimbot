
import mlx.core as mx
import mlx.nn as nn
import numpy as np

class ActorCritic(nn.Module):
    def __init__(self, observation_shape, action_shape):
        super().__init__()
        
        input_dim = observation_shape[0]
        action_dim = action_shape[0]
        
        # Shared feature extractor / Backbone (Optional, can separate)
        # Let's use separate networks for Actor and Critic as it's often more stable for PPO
        
        # Actor Network
        self.actor_l1 = nn.Linear(input_dim, 64)
        self.actor_l2 = nn.Linear(64, 64)
        self.actor_mean = nn.Linear(64, action_dim)
        self.actor_log_std = mx.array(np.zeros(action_dim)) # Learnable parameter
        # Note: In MLX we can make it a parameter, but careful with optimization. 
        # Usually easier to have it as an output of the network or a separate variable.
        # Let's make it a variable we optimize.
        
        # Critic Network
        self.critic_l1 = nn.Linear(input_dim, 64)
        self.critic_l2 = nn.Linear(64, 64)
        self.critic_value = nn.Linear(64, 1)

    def actor(self, x):
        x = nn.Tanh()(self.actor_l1(x))
        x = nn.Tanh()(self.actor_l2(x))
        mean = self.actor_mean(x)
        # We want log_std to be learnable.
        # However, MLX Module parameters need to be attributes of the class that are Arrays or Modules.
        # We can implement a small parameter wrapper or just return it.
        # Actually, simpler: Let's just output it from a layer or keep it constant for simplest version?
        # User said: "Log Std (σ): The uncertainty/variance (learnable parameter)."
        # So we should make it learnable.
        return mean, self.actor_log_std

    def critic(self, x):
        x = nn.Tanh()(self.critic_l1(x))
        x = nn.Tanh()(self.critic_l2(x))
        return self.critic_value(x)
        
    def __call__(self, x):
        # Forward pass for both? Usually we call them separately in PPO but convenient here.
        mean, log_std = self.actor(x)
        value = self.critic(x)
        return mean, log_std, value
