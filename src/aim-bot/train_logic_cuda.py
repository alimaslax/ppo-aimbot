import numpy as np
import torch
from brain_cuda import ActorCritic, PPOAgent
import time
import os
import wandb
import argparse

# --- Config ---
TOTAL_TIMESTEPS = 1000000
UPDATE_TIMESTEPS = 2048
EPOCHS = 4
BATCH_SIZE = 64
LR = 3e-4
SAVE_INTERVAL = 50000 

def train(resume_path=None):
    # 0. Init WandB
    run_name = f"aimbot_math_cuda_{int(time.time())}"
    wandb.init(
        project="ppo-aimbot",
        name=run_name,
        config={
            "total_timesteps": TOTAL_TIMESTEPS,
            "update_timesteps": UPDATE_TIMESTEPS,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "lr": LR,
            "resume_from": resume_path,
            "device": "cuda" if torch.cuda.is_available() else "cpu"
        }
    )

    # 1. Setup Environment
    obs_dim = (2,)
    action_dim = (2,)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    model = ActorCritic(obs_dim, action_dim).to(device)
    
    # Load Weights if resuming
    if resume_path and os.path.exists(resume_path):
        print(f"Resuming training from {resume_path}...")
        model.load_weights(resume_path)
    else:
        print("Starting training from scratch...")

    agent = PPOAgent(model, learning_rate=LR, device=device)
    
    # State
    cursor_x, cursor_y = 0.0, 0.0
    target_x, target_y = np.random.uniform(-0.8, 0.8), np.random.uniform(-0.8, 0.8)
    
    step = 0
    score_history = []
    
    print(f"Starting Math Training (WandB run: {run_name})...")
    
    memory = {'obs': [], 'actions': [], 'logprobs': [], 'rewards': [], 'dones': [], 'values': []}
    
    next_obs = np.array([target_x - cursor_x, target_y - cursor_y], dtype=np.float32)

    start_time = time.time()
    
    if not os.path.exists("weights"):
        os.makedirs("weights")
    
    while step < TOTAL_TIMESTEPS:
        # Get Action
        action, log_prob, value = agent.get_action(next_obs)
        action_np = np.array(action).flatten()
        
        # --- EXECUTE STEP (Simulate Physics) ---
        
        # 1. Move Cursor
        move_speed = 0.1 # Max movement per step relative to screen
        dx = np.clip(action_np[0], -1, 1) * move_speed
        dy = np.clip(action_np[1], -1, 1) * move_speed
        
        cursor_x += dx
        cursor_y += dy
        
        # Clip cursor
        cursor_x = np.clip(cursor_x, -1, 1)
        cursor_y = np.clip(cursor_y, -1, 1)
        
        # 2. Calculate Distance for Reward
        dist = np.sqrt((target_x - cursor_x)**2 + (target_y - cursor_y)**2)
        
        # Reward: Higher is better. 
        # Aim: Distance < 0.05
        reward = -dist # Simple negative distance
        if dist < 0.05:
            reward += 1.0 # Hit bonus
            
        # 3. New Obs
        # If "hit" or timeout, reset target
        done = False
        if dist < 0.05 or np.random.rand() < 0.01: # 1% chance to respawn target to prevent getting stuck
             done = True
             target_x, target_y = np.random.uniform(-0.8, 0.8), np.random.uniform(-0.8, 0.8)
             # Optionally reset cursor to center or keep it
             # cursor_x, cursor_y = 0.0, 0.0 
             
        next_obs_new = np.array([target_x - cursor_x, target_y - cursor_y], dtype=np.float32)
        
        # --- STORE ---
        memory['obs'].append(next_obs)
        memory['actions'].append(action_np)
        memory['logprobs'].append(np.array(log_prob).flatten()[0])
        memory['rewards'].append(reward)
        memory['dones'].append(done)
        memory['values'].append(np.array(value).flatten()[0])
        
        next_obs = next_obs_new
        step += 1
        
        # --- UPDATE ---
        if step % UPDATE_TIMESTEPS == 0:
            loss = agent.update(memory, batch_size=BATCH_SIZE, epochs=EPOCHS)
            avg_reward = np.mean(memory['rewards'])
            score_history.append(avg_reward)
            
            # Clear memory
            memory = {'obs': [], 'actions': [], 'logprobs': [], 'rewards': [], 'dones': [], 'values': []}
            
            # FPS Calculation
            fps = int(step / (time.time() - start_time))
            
            print(f"Step {step}: Loss={loss:.4f}, Avg Reward={avg_reward:.4f}, FPS={fps}")
            
            # WandB Log
            wandb.log({
                "global_step": step,
                "loss": loss,
                "avg_reward": avg_reward,
                "fps": fps
            })

        # --- SAVE CHECKPOINT ---
        if step % SAVE_INTERVAL == 0:
            save_path = f"weights/model_cuda_{step}.pth"
            model.save_weights(save_path)
            print(f"Checkpoint saved: {save_path}")
            
    # Save Final
    model.save_weights("weights/best_aim_cuda.pth")
    print("Training Complete. Weights saved to weights/best_aim_cuda.pth")
    wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=str, default=None, help="Path to .pth checkpoint to resume from")
    args = parser.parse_args()
    train(resume_path=args.resume)
