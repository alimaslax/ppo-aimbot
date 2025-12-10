# PPO Aimbot: Logic & Architecture

https://aiming.pro/app#/play/benchmark/1

This project uses a unique "Abstract Training" approach. Instead of training inside the game (which is slow and complex), we train the AI on **Pure Mathematics** and then apply that logic to the Real World.

## 1. How Training Works (`train_logic.py`)

The training script does **not** look at your screen. It runs a high-speed mathematical simulation in memory (The "Math Gym").

### The Concept
The Agent doesn't need to see graphics to learn how to aim. It just needs to learn the relationship between "Numbers" and "Movement".
- **Input:** "Target is at `(0.5, -0.2)`" relative to you.
- **Output:** "Move mouse `(dx, dy)`".

### The Training Loop
It repeats this cycle endlessly (millions of times):
1.  **Spawn:** The script picks a random coordinate `(target_x, target_y)` to simulate a target.
2.  **Observe:** The Agent aims. It measures the distance to that target.
3.  **Act:** The Agent chooses a mouse movement `(dx, dy)`.
4.  **Reward:**
    *   **Closer?** Small reward.
    *   **Hit?** (Distance < 0.05) **Big Reward (+1.0)**.
5.  **Respawn:** If hit, the target instantly moves to a new random spot, forcing the agent to snap to a new location.

### 📊 Interpreting Training Metrics
When watching the logs, focus on **Avg Reward**:

| Avg Reward | Meaning | Behavior |
| :--- | :--- | :--- |
| **-1.0 to -0.5** | **Novice** | The bot is flailing around, often moving *away* from the target. |
| **-0.1 to 0.0** | **Competent** | The bot is consistently moving towards the target and staying close. |
| **> 0.5** | **God-Like** | The bot is "snapping" to the target instantly and hitting it frequently. |

---

## 2. How Inference Works (`run.py`)

This script acts as the **Translation Layer**. It bridges the gap between the Real World (YOLO) and the Abstract World (The Brain).

### The Pipeline
1.  **YOLO Detects (Real World)**
    *   "I see a blue ball at pixel `x=400, y=300`."
2.  **Translation (The Bridge)**
    *   Calculates relative distance from screen center (e.g., `320, 320`).
    *   Converts pixels to the *Abstract Math Language* the brain understands.
    *   *Result:* "Target is `+0.25, -0.06` relative to center."
3.  **The Brain Reacts (Abstract World)**
    *   The Brain receives `[0.25, -0.06]`.
    *   It recognizes this pattern from training!
    *   *Command:* "Move mouse `[+0.5, -0.1]`."
4.  **Action (Real World)**
    *   `pyautogui` applies that movement to your physical mouse.
    *   The cursor snaps to the target.

### Why this approach?
*   **Speed:** We can train for millions of steps in minutes because there are no graphics.
*   **Generalization:** The specific game doesn't matter. As long as YOLO can detect it, the "Aiming Brain" works.

---

## 3. Usage Guide

### Train from Scratch
```bash
python src/aim-bot/train_logic.py
```

### Resume Training
If your training stopped or you want to continue improving a model:
```bash
# Resume from specific checkpoint
python src/aim-bot/train_logic.py --resume weights/model_50000.npz

# Resume from best model
python src/aim-bot/train_logic.py --resume weights/best_aim.npz
```

### Run the Bot
```bash
python src/aim-bot/run.py
```