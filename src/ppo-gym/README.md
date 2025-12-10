# PPO Aimbot with MLX

Welcome to the PPO Aimbot project! This repository contains a clean, from-scratch implementation of **Proximal Policy Optimization (PPO)** using Apple's **MLX** framework. 

This guide is designed to help you understand the components of a Reinforcement Learning (RL) system, even if you are new to the field.

## Project Structure

The project is broken down into four main components:

1.  **Environment** (`env.py`): The "game" aimed to be solved.
2.  **Model** (`model.py`): The "brain" (Neural Network).
3.  **Agent** (`agent.py`): The "learner" (PPO Algorithm).
4.  **Training** (`train.py`): The "loop" that puts it all together.

---

## 1. The Environment (`env.py`)

The **Environment** is the world the AI interacts with. In this project, we utilize `gymnasium` to create a custom environment called `SimpleAimEnv`.

*   **The Goal**: The Agent controls a "mouse" (green crosshair) and must move it to overlap with a moving "target" (red circle).
*   **Observations (Inputs)**: Instead of "seeing" pixels like a human, the agent receives a list of 4 numbers:
    *   `[rel_x, rel_y]`: The relative distance from the mouse to the target.
    *   `[vel_x, vel_y]`: The velocity (speed and direction) of the target.
*   **Actions (Outputs)**: The agent outputs 2 numbers:
    *   `[mouse_dx, mouse_dy]`: The speed to move the mouse in the X and Y directions.
*   **Rewards**: The environment tells the agent how well it's doing:
    *   **Precision Reward**: Higher points for being closer to the target.
    *   **Hit Bonus**: A large point boost (+1.0) for actually touching the target.
    *   **Effort Penalty**: A small penalty for moving too fast (encourages efficiency).

## 2. The Model (`model.py`)

The **Model** defines the architecture of the Neural Network (the specific type of math used to make decisions). We use a standard architecture called **Actor-Critic**.

*   **Input**: Takes the 4 observation numbers from the environment.
*   **Hidden Layers**: Two layers of 64 neurons each. These process the raw numbers into meaningful features.
*   **Output Heads**:
    *   **Actor**: Decides *what to do* (moves the mouse).
    *   **Critic**: Estimates *how good* the current situation is (predicts future score).

This file utilizes `mlx.nn` to define these layers, similar to PyTorch or TensorFlow but optimized for Apple Silicon.

## 3. The Agent (`agent.py`)

The **Agent** contains the logic for the **PPO (Proximal Policy Optimization)** algorithm. Think of this as the "coach" that trains the brain.

*   **PPO Explained**: Reinforcement learning is trial-and-error. PPO is a specific method that ensures the agent learns *stably*. It prevents the agent from making drastic changes to its behavior based on a single lucky (or unlucky) attempt, which prevents "forgetting" what it has already learned.
*   **Key Functions**:
    *   `get_action()`: Uses the model to decide on a move to make.
    *   `compute_gae()`: Calculates "advantages"—basically figuring out which specific moves led to good rewards.
    *   `update()`: Adjusts the neural network weights to make good actions more likely and bad actions less likely.

## 4. The Training Process (`train.py`)

This file orchestrates the entire learning process. It runs a loop that looks like this:

1.  **Collection Phase**: The agent plays the game for a set number of steps (e.g., 2048 actions) using its current brain. It records everything: what it saw, what it did, and what reward it got.
2.  **Learning Phase**: The agent pauses and reviews the recordings. It uses the PPO algorithm to update its brain (Model) to be slightly better.
3.  **Repeat**: This cycle happens over and over.

By the end of training (typically a few minutes), the agent goes from moving randomly to snapping onto the target instantly.

## 5. Usage

### Prerequisites
Make sure you have `gymnasium`, `numpy`, `mlx`, `wandb`, and `pygame` installed.

### Training
To start training the agent from scratch:

```bash
python train.py
```

Arguments you can tweak:
*   `--total_timesteps`: How long to train (default: 500,000)
*   `--track`: Enable WandB logging to visualize graphs of your training.

### specific Playback
To watch a trained model play:

```bash
python train.py --play_only --model_path models/model_YOUR_RUN_NAME.npz
```
