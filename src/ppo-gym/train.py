
import os
import time
import argparse
import gymnasium as gym
import numpy as np
import mlx.core as mx
import wandb
import math

from src.ppo-gym.env import SimpleAimEnv
from src.ppo-gym.model import ActorCritic
from src.ppo-gym.agent import PPOAgent

def train(args):
    run_name = f"ppo_aimbot_{int(time.time())}"
    if args.track:
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )

    # Environment setup
    env = SimpleAimEnv()
    
    # Model & Agent setup
    observation_shape = env.observation_space.shape
    action_shape = env.action_space.shape
    
    model = ActorCritic(observation_shape, action_shape)
    agent = PPOAgent(
        model, 
        learning_rate=args.learning_rate, 
        gamma=args.gamma, 
        gae_lambda=args.gae_lambda, 
        clip_coef=args.clip_coef, 
        ent_coef=args.ent_coef, 
        vf_coef=args.vf_coef
    )

    # Storage setup
    obs = np.zeros((args.num_steps, *observation_shape), dtype=np.float32)
    actions = np.zeros((args.num_steps, *action_shape), dtype=np.float32)
    logprobs = np.zeros((args.num_steps,), dtype=np.float32)
    rewards = np.zeros((args.num_steps,), dtype=np.float32)
    dones = np.zeros((args.num_steps,), dtype=np.float32)
    values = np.zeros((args.num_steps,), dtype=np.float32)

    # Start game
    global_step = 0
    start_time = time.time()
    next_obs, _ = env.reset(seed=args.seed)
    next_done = False
    
    num_updates = args.total_timesteps // args.num_steps

    print(f"Starting training: {args.total_timesteps} steps, {num_updates} updates.")

    for update in range(1, num_updates + 1):
        # 1. Collect Rollouts
        for step in range(args.num_steps):
            global_step += 1
            obs[step] = next_obs
            dones[step] = next_done

            # Get action
            action, logprob, value = agent.get_action(next_obs)
            
            # Store values (convert from mx to np if needed)
            # agent.get_action returns mx arrays, need to convert to np for storage?
            # Creating huge mx arrays is fine, but constructing update batches might be easier with numpy.
            # Let's convert to numpy for storage.
            values[step] = np.array(value).flatten()[0]
            actions[step] = np.array(action).flatten()
            logprobs[step] = np.array(logprob).flatten()[0]

            # Execute Env
            next_obs, reward, terminated, truncated, info = env.step(actions[step])
            rewards[step] = reward
            next_done = terminated or truncated

            if next_done:
                # Optionally record episode info
                if args.track:
                    wandb.log({"charts/episodic_return": info.get("episode", {}).get("r", 0)}) # If Wrapper exists
                # Reset
                next_obs, _ = env.reset()

        # 2. Bootstrap value for GAE
        _, _, next_value = agent.get_action(next_obs)
        next_value = np.array(next_value).flatten()[0]
        
        # 3. Compute GAE
        returns, advantages = agent.compute_gae(rewards, values, next_value, dones)
        
        # Flatten the batch
        b_obs = obs.reshape((-1,) + observation_shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + action_shape)
        b_advantages = np.array(advantages).reshape(-1)
        b_returns = np.array(returns).reshape(-1)
        b_values = values.reshape(-1)

        # Convert to MLX arrays for update
        mx_obs = mx.array(b_obs)
        mx_actions = mx.array(b_actions)
        mx_logprobs = mx.array(b_logprobs)
        mx_returns = mx.array(b_returns)
        mx_advantages = mx.array(b_advantages)

        # 4. Update Policy (Epochs & Minibatches)
        b_inds = np.arange(args.num_steps)
        clipfracs = []
        
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.num_steps, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                # Slicing in MLX or Numpy
                # It's better to slice numpy then convert to MLX if data is large, 
                # but these are small (2048 steps). MLX slicing is efficiently handled?
                # Actually, slicing `mx.array` creates views or new arrays.
                # Let's slice the MLX arrays directly.
                # Convert indices to MLX array for indexing
                mx_mb_inds = mx.array(mb_inds)
                
                loss = agent.update(
                    mx_obs[mx_mb_inds],
                    mx_actions[mx_mb_inds],
                    mx_logprobs[mx_mb_inds],
                    mx_returns[mx_mb_inds],
                    mx_advantages[mx_mb_inds]
                )

        # Logging
        print(f"Update {update}/{num_updates} | Loss: {loss.item():.4f} | Reward (Last Batch Mean): {rewards.mean():.4f}")
        
        if args.track:
            wandb.log({
                "charts/learning_rate": args.learning_rate,
                "losses/value_loss": 0, # TODO: Return breakdown from agent for better logging
                "losses/policy_loss": loss.item(), 
                "losses/entropy": 0,
                "charts/SPS": int(global_step / (time.time() - start_time)),
            }, step=global_step)
            
        # Save Model
        if update % args.save_frequency == 0:
             if not os.path.exists("models"):
                 os.makedirs("models")
             model.save_weights(f"models/model_{run_name}_{update}.npz")
             print(f"Model saved to models/model_{run_name}_{update}.npz")
             
    # Save final model
    if not os.path.exists("models"):
        os.makedirs("models")
    final_model_path = f"models/model_{run_name}_final.npz"
    model.save_weights(final_model_path)
    print(f"Final model saved to {final_model_path}")
    
    env.close()
    if args.track:
        wandb.finish()
        
    return final_model_path

def play(model_path):
    print(f"Loading model from {model_path} for visualization...")
    env = SimpleAimEnv(render_mode="human")
    model = ActorCritic(env.observation_space.shape, env.action_space.shape)
    model.load_weights(model_path)
    agent = PPOAgent(model) # Params don't matter for inference
    
    obs, _ = env.reset()
    done = False
    
    print("Press Ctrl+C to stop.")
    try:
        while True:
            # env.render() is handled inside step for human mode usually, or explicit call
            env.render()
            
            # Action (Deterministic for eval usually -> Mean)
            # current get_action samples. Let's strictly use Mean?
            # Or just sample with low noise? PPO usually samples.
            # But for "best performance" we should probably take mean.
            # Let's modify agent or just use `model` directly.
            mean, _, _ = model(mx.array(obs[None, :]))
            action = np.array(mean).flatten()
            
            obs, reward, terminated, truncated, _ = env.step(action)
            
            if terminated or truncated:
                obs, _ = env.reset()
                
    except KeyboardInterrupt:
        print("Stopping playback.")
        env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--total_timesteps", type=int, default=500000)
    parser.add_argument("--num_steps", type=int, default=2048)
    parser.add_argument("--minibatch_size", type=int, default=64)
    parser.add_argument("--update_epochs", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--clip_coef", type=float, default=0.2)
    parser.add_argument("--ent_coef", type=float, default=0.01)
    parser.add_argument("--vf_coef", type=float, default=0.5)
    parser.add_argument("--track", action="store_true", help="Track with WandB")
    parser.add_argument("--wandb_project_name", type=str, default="ppo-aimbot")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--save_frequency", type=int, default=10)
    parser.add_argument("--play_only", action="store_true", help="Skip training and play loaded model")
    parser.add_argument("--model_path", type=str, default=None, help="Path to model for playback")

    args = parser.parse_args()
    
    if args.play_only:
        if args.model_path and os.path.exists(args.model_path):
            play(args.model_path)
        else:
            print("Please provide a valid --model_path for play mode.")
    else:
        final_path = train(args)
        # Auto play after training
        play(final_path)
