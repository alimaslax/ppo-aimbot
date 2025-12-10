
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import math

class SimpleAimEnv(gym.Env):
    """
    A custom environment that simulates a mouse moving on a screen to hit a target.
    
    Observation Space: Box(4,)
        [rel_x, rel_y]: Distance to target (Normalized -1.0 to 1.0)
        [vel_x, vel_y]: Velocity of the target (if moving)
        
    Action Space: Box(2,)
        [mouse_dx, mouse_dy]: Continuous values (-1.0 to 1.0) representing mouse speed.
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode
        self.screen_width = 800
        self.screen_height = 600
        self.window = None
        self.clock = None
        
        # Physics constants
        self.mouse_speed_scale = 50.0  # Max pixels per step
        self.hitbox_radius = 20.0
        self.target_move_speed = 5.0 # Max target movement speed
        
        # Spaces
        # Obs: [rel_x, rel_y, target_vel_x, target_vel_y]
        # rel_x, rel_y are normalized by screen dimensions roughly, but let's say -1 to 1 covers the screen
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        
        # Action: [dx, dy]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        # State
        self.mouse_pos = np.zeros(2, dtype=np.float32)
        self.target_pos = np.zeros(2, dtype=np.float32)
        self.target_vel = np.zeros(2, dtype=np.float32)
        self.steps = 0
        self.max_steps = 200

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Reset positions
        # Start mouse in center
        self.mouse_pos = np.array([self.screen_width / 2, self.screen_height / 2], dtype=np.float32)
        
        # Random target position
        self.target_pos = np.array([
            self.np_random.uniform(0, self.screen_width),
            self.np_random.uniform(0, self.screen_height)
        ], dtype=np.float32)
        
        # Random target velocity (simple moving target)
        self.target_vel = np.array([
            self.np_random.uniform(-self.target_move_speed, self.target_move_speed),
            self.np_random.uniform(-self.target_move_speed, self.target_move_speed)
        ], dtype=np.float32)
        
        self.steps = 0
        
        return self._get_obs(), {}

    def step(self, action):
        self.steps += 1
        
        # 1. Apply Action (Move Mouse)
        # Action is -1 to 1. Scale to pixels.
        move = action * self.mouse_speed_scale
        self.mouse_pos += move
        
        # Clip mouse to screen
        self.mouse_pos = np.clip(self.mouse_pos, [0, 0], [self.screen_width, self.screen_height])
        
        # 2. Update Environment (Move Target)
        self.target_pos += self.target_vel
        
        # Bounce target off walls
        if self.target_pos[0] <= 0 or self.target_pos[0] >= self.screen_width:
            self.target_vel[0] *= -1
        if self.target_pos[1] <= 0 or self.target_pos[1] >= self.screen_height:
            self.target_vel[1] *= -1
            
        self.target_pos = np.clip(self.target_pos, [0, 0], [self.screen_width, self.screen_height])
        
        # 3. Calculate Reward
        # Distance
        dist_vec = self.target_pos - self.mouse_pos
        distance = np.linalg.norm(dist_vec)
        
        # Precision Reward: Higher when closer. 
        # exp(-0.01 * distance) -> at dist 0 = 1.0, at dist 100 = 0.36, at dist 500 = 0.006
        precision_reward = np.exp(-0.01 * distance)
        
        reward = precision_reward
        
        # Hit Bonus
        terminated = False
        if distance < self.hitbox_radius:
            reward += 1.0
            # Optionally reset target or just continue tracking? 
            # User said "Continuous Control", usually tracking is better.
            # But let's say we "hit" it and maybe it respawns or we just get points?
            # For this simple version, let's just count it as a hit and maybe respawn target to keep it interesting
            # OR just keep tracking. Let's keep tracking for continuous control smoothness.
            # Actually, usually aimbots train to SNAP to target.
            
        # Effort Penalty (optional)
        effort = np.linalg.norm(action)
        reward -= 0.01 * effort
        
        # Check termination
        truncated = False
        if self.steps >= self.max_steps:
            truncated = True
            
        return self._get_obs(), reward, terminated, truncated, {}

    def _get_obs(self):
        # Return [rel_x, rel_y, vel_x, vel_y]
        # Normalize relative distance by screen size (approximately)
        rel_pos = (self.target_pos - self.mouse_pos)
        rel_x = rel_pos[0] / self.screen_width
        rel_y = rel_pos[1] / self.screen_height
        
        # Normalize velocity
        vel_x = self.target_vel[0] / self.target_move_speed
        vel_y = self.target_vel[1] / self.target_move_speed
        
        return np.array([rel_x, rel_y, vel_x, vel_y], dtype=np.float32)

    def render(self):
        if self.render_mode is None:
            return

        import pygame
        
        if self.window is None:
            pygame.init()
            if self.render_mode == "human":
                pygame.display.init()
                self.window = pygame.display.set_mode((self.screen_width, self.screen_height))
            else:
                self.window = pygame.Surface((self.screen_width, self.screen_height))
                
        if self.clock is None:
            self.clock = pygame.time.Clock()
            
        canvas = pygame.Surface((self.screen_width, self.screen_height))
        canvas.fill((0, 0, 0)) # Black background
        
        # Draw Target (Red Circle)
        pygame.draw.circle(
            canvas,
            (255, 0, 0),
            self.target_pos.astype(int),
            int(self.hitbox_radius)
        )
        
        # Draw Mouse/Crosshair (Green Cross)
        mx, my = self.mouse_pos.astype(int)
        pygame.draw.line(canvas, (0, 255, 0), (mx - 10, my), (mx + 10, my), 2)
        pygame.draw.line(canvas, (0, 255, 0), (mx, my - 10), (mx, my + 10), 2)
        
        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            import pygame
            pygame.display.quit()
            pygame.quit()
