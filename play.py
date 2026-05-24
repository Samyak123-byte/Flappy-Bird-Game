import gymnasium as gym
import flappy_bird_gymnasium
import pygame
import sys

# Environment initialize karein
env = gym.make("FlappyBird-v0", render_mode="human")
obs, info = env.reset()

print("Game Start! JUMP karne ke liye 'SPACEBAR' dabayein.")
print("Game se baahar aane ke liye 'ESC' dabayein.")

clock = pygame.time.Clock()

# Yeh track karne ke liye ki kya spacebar daba hua hai
space_pressed = False

while True:
    action = 0  # Default: Kuch mat karo (Bird niche giregi)
    
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            env.close()
            sys.exit()
        
        if event.type == pygame.KEYDOWN:
            # Sirf tabhi jump hoga jab pehle se spacebar daba hua NA HO
            if event.key == pygame.K_SPACE and not space_pressed:
                action = 1  
                space_pressed = True # Lock laga diya
            if event.key == pygame.K_ESCAPE:
                env.close()
                sys.exit()
                
        if event.type == pygame.KEYUP:
            if event.key == pygame.K_SPACE:
                space_pressed = False # Ungli uthane par lock khul gaya

    # Game mein action apply karein
    obs, reward, terminated, truncated, info = env.step(action)

    # Agar bird pipes se takraye ya zameen par gire
    if terminated or truncated:
        print("Game Over! Restarting...")
        pygame.time.wait(600)  # Crash hone par 0.6 second ka break taaki aapko pata chale
        obs, info = env.reset()
        space_pressed = False

    clock.tick(30)  # Game speed ko normal 30 FPS par chalane ke liye
