import pygame
import socket
import json
import time
import random
import math

# --- Config ---
WIDTH, HEIGHT = 640, 640
TARGET_RADIUS = 20
BG_COLOR = (20, 20, 20)
TARGET_COLOR = (255, 0, 0) # Red
FPS = 60
UDP_IP = "127.0.0.1"
UDP_PORT = 9999

def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Aimbot Test Target")
    clock = pygame.time.Clock()

    # UDP Setup
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    
    # Feedback Listener (Receive mouse moves)
    feedback_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    feedback_sock.bind(("0.0.0.0", 9998))
    feedback_sock.setblocking(False)

    # Target State
    target_x = WIDTH // 2
    target_y = HEIGHT // 2
    velocity_x = random.choice([-3, -2, 2, 3])
    velocity_y = random.choice([-3, -2, 2, 3])

    running = True
    print(f"Game started. Broadcasting target coordinates to {UDP_IP}:{UDP_PORT}")

    try:
        while running:
            # Event Handling
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            # Update ID: 1
            target_x += velocity_x
            target_y += velocity_y

            # Bounce off walls
            if target_x - TARGET_RADIUS < 0 or target_x + TARGET_RADIUS > WIDTH:
                velocity_x *= -1
                target_x = max(TARGET_RADIUS, min(WIDTH - TARGET_RADIUS, target_x))
            
            if target_y - TARGET_RADIUS < 0 or target_y + TARGET_RADIUS > HEIGHT:
                velocity_y *= -1
                target_y = max(TARGET_RADIUS, min(HEIGHT - TARGET_RADIUS, target_y))

            # Randomly change direction occasionally
            if random.random() < 0.02:
                 velocity_x = random.choice([-4, -3, -2, 2, 3, 4])
                 velocity_y = random.choice([-4, -3, -2, 2, 3, 4])


            # Process Feedback (Simulate Camera Movement)
            try:
                while True: # Drain the queue
                    data, _ = feedback_sock.recvfrom(1024)
                    feedback = json.loads(data.decode())
                    dx = feedback.get('dx', 0)
                    dy = feedback.get('dy', 0)
                    
                    # Move target OPPOSITE to mouse movement (Simulating view change)
                    # If mouse moves RIGHT, View turns RIGHT, so Target moves LEFT on screen.
                    target_x -= dx
                    target_y -= dy
            except BlockingIOError:
                pass


            # Send Coordinates via UDP
            # We send dictionary as JSON bytes
            data = {
                "x": target_x,
                "y": target_y,
                "screen_width": WIDTH,
                "screen_height": HEIGHT
            }
            message = json.dumps(data).encode('utf-8')
            sock.sendto(message, (UDP_IP, UDP_PORT))

            # Check for "Hit" / Lock
            center_x, center_y = WIDTH // 2, HEIGHT // 2
            dist_to_center = math.sqrt((target_x - center_x)**2 + (target_y - center_y)**2)
            is_locked = dist_to_center < TARGET_RADIUS

            # Render
            screen.fill(BG_COLOR)
            
            # Draw line from center to target (visualize error)
            pygame.draw.line(screen, (50, 50, 50), (center_x, center_y), (int(target_x), int(target_y)), 1)
            
            # Target color changes to GREEN if locked
            current_target_color = (0, 255, 0) if is_locked else TARGET_COLOR
            pygame.draw.circle(screen, current_target_color, (int(target_x), int(target_y)), TARGET_RADIUS)
            
            # Visualize Aimbot Intent (Movement Vector)
            # dx, dy are the last received feedback
            if 'dx' in locals():
                # Scale up for visibility
                end_pos = (center_x + dx*2, center_y + dy*2)
                pygame.draw.line(screen, (255, 255, 0), (center_x, center_y), end_pos, 3) # Yellow Arrow

            # Crosshair (Center)
            # Green if not locked, Red/Yellow if locked
            xhair_color = (100, 100, 100) if not is_locked else (255, 0, 0)
            xhair_width = 2 if not is_locked else 4
            
            # Horizontal (Fixed Center)
            pygame.draw.line(screen, xhair_color, (center_x - 15, center_y), (center_x + 15, center_y), xhair_width)
            # Vertical (Fixed Center)
            pygame.draw.line(screen, xhair_color, (center_x, center_y - 15), (center_x, center_y + 15), xhair_width)

            if is_locked:
                 font = pygame.font.SysFont(None, 36)
                 img = font.render('LOCKED', True, (255, 0, 0))
                 screen.blit(img, (center_x + 20, center_y + 20))

            pygame.display.flip()
            clock.tick(FPS)

    except KeyboardInterrupt:
        pass
    finally:
        pygame.quit()
        sock.close()
        print("Game closed.")

if __name__ == "__main__":
    main()
