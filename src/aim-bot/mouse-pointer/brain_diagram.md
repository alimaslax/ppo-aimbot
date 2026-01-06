# Brain.py (MLX Implementation) Structure

![Whiteboard Diagram](./brain_diagram.png)

This diagram follows the structure of `brain.py` but is visualized in a whiteboard style as requested.

## Mermaid Reference


```mermaid
classDiagram
    class ActorCritic {
        +Sequential common
        +Linear actor_mean
        +Parameter actor_logstd
        +Linear critic
        +__init__(observation_shape, action_shape)
        +__call__(x) : mean, logstd, value
        +get_action_mean(x) : action_mean
    }

    class PPOAgent {
        +ActorCritic model
        +Adam optimizer
        +float gamma
        +float eps_clip
        +__init__(model, learning_rate, gamma, eps_clip)
        +get_action(obs) : action, log_prob, value
        +update(memory, batch_size, epochs) : avg_loss
    }

    PPOAgent --> ActorCritic : uses
    
    note for PPOAgent "Handles training loop,\nGAE calculation, and\nPPO loss computation"
    note for ActorCritic "Neural Network Architecture:\nCommon (Linear->Tanh->Linear->Tanh)\nSplit flows for Actor (Mean) and Critic (Value)"
```

## Flow Description

1. **Initialization**: `PPOAgent` is initialized with an instance of `ActorCritic`.
2. **Action Selection**: `PPOAgent.get_action(obs)` calls `ActorCritic` to get parameters, samples an action using normal distribution, and returns the action, log probability, and value estimate.
3. **Training (`update`)**:
    - Unpacks memory (observations, actions, rewards, etc.).
    - Computes Generalized Advantage Estimation (GAE) and Returns.
    - Runs a PPO update loop for a number of epochs.
    - Shuffles data and processes mini-batches.
    - Calculates PPO loss (Actor loss + Critic loss).
    - Updates `ActorCritic` weights using `optimizer`.
