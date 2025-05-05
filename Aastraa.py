import pygame
import random
import sys
import numpy as np
import pickle
from sklearn.neighbors import KNeighborsClassifier
import os
from PIL import Image

pygame.init()

WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
BLACK = (0, 0, 0)
YELLOW = (255, 255, 0)

WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Hunter")

PLAYER_SIZE = 50
player_x = WIDTH // 2
player_y = HEIGHT - PLAYER_SIZE - 10
PLAYER_SPEED = 5

BULLET_SIZE = 10
bullets = []
BULLET_SPEED = 7

ALIEN_SIZE = 50
alien = None
ALIEN_SPEED = 1.5
misses = 0
speed_multiplier = 1.0
spawn_rate = 0.025

ASSETS_PATH = "assets"


if not os.path.exists(ASSETS_PATH):
    print(f"Error: Assets directory not found at {ASSETS_PATH}")
    print("Creating a default assets directory...")
    os.makedirs(ASSETS_PATH, exist_ok=True) 

def load_image(path):
    try:
        pil_image = Image.open(path)
        mode = pil_image.mode
        size = pil_image.size
        data = pil_image.tobytes()
        
        py_image = pygame.image.fromstring(data, size, mode)
        return py_image
    except Exception as e:
        print(f"Error loading image {path}: {e}")
        return None

try:
    print(f"Trying to load images from: {ASSETS_PATH}")
    
    background_img = load_image(os.path.join(ASSETS_PATH, 'space_bg.jpg'))
    if background_img:
        print("Loaded background image")
        background_img = pygame.transform.scale(background_img, (WIDTH, HEIGHT))
    else:
        raise Exception("Failed to load background image")
    
    spaceship_img = load_image(os.path.join(ASSETS_PATH, 'spaceship.png'))
    if spaceship_img:
        print("Loaded spaceship image")
        spaceship_img = pygame.transform.scale(spaceship_img, (PLAYER_SIZE, PLAYER_SIZE))
    else:
        raise Exception("Failed to load spaceship image")
    
    asteroid_img = load_image(os.path.join(ASSETS_PATH, 'asteroid.png'))
    if asteroid_img:
        print("Loaded asteroid image")
        asteroid_img = pygame.transform.scale(asteroid_img, (ALIEN_SIZE, ALIEN_SIZE))
    else:
        raise Exception("Failed to load asteroid image")
    
    missile_img = None
    for ext in ['png', 'jpg', 'jpeg', 'bmp']:
        missile_img = load_image(os.path.join(ASSETS_PATH, f'missile.{ext}'))
        if missile_img:
            print(f"Loaded missile image as .{ext}")
            missile_img = pygame.transform.scale(missile_img, (BULLET_SIZE, BULLET_SIZE*2))
            break
    
    if not missile_img:
        print("Creating a simple missile image")
        missile_surface = pygame.Surface((BULLET_SIZE, BULLET_SIZE*2), pygame.SRCALPHA)
        pygame.draw.rect(missile_surface, BLUE, (0, 0, BULLET_SIZE, BULLET_SIZE*2))
        missile_img = missile_surface
    
    
    has_images = True
    print("Successfully loaded all images!")
except Exception as e:
    print(f"Warning: Could not load image files: {e}")
    print(f"Looking in path: {ASSETS_PATH}")
    has_images = False

print("Creating missile image programmatically...")
missile_img = pygame.Surface((BULLET_SIZE, BULLET_SIZE*2), pygame.SRCALPHA)
pygame.draw.polygon(missile_img, BLUE, [
    (BULLET_SIZE//2, 0),
    (0, BULLET_SIZE*2),
    (BULLET_SIZE, BULLET_SIZE*2)
])
pygame.draw.polygon(missile_img, YELLOW, [
    (BULLET_SIZE//4, BULLET_SIZE*2),
    (BULLET_SIZE//2, BULLET_SIZE*2 + BULLET_SIZE//2),
    (3*BULLET_SIZE//4, BULLET_SIZE*2)
])

PLAYER_BASE_SPEED = 5
BULLET_COOLDOWN = 200

font = pygame.font.Font(None, 36)

data = []
target = []
knn_model = KNeighborsClassifier(n_neighbors=3)

def adjust_difficulty(score, misses):
    if len(data) > 3:
        difficulty = knn_model.predict([[score, misses]])[0]
        return difficulty * 0.5
    return 1.0

Q_TABLE = {}
LEARNING_RATE = 0.1
DISCOUNT_FACTOR = 0.9
EXPLORATION_RATE = 0.3

def q_learning_difficulty_adjustment(score, misses, last_state=None, last_action=None, reward=None):
    
    current_state = (score // 5, min(misses, 3))
    
    if current_state not in Q_TABLE:
        Q_TABLE[current_state] = {"increase": 0, "decrease": 0, "maintain": 0}
    
    if last_state and last_action and reward is not None:
        max_future_q = max(Q_TABLE[current_state].values())
        current_q = Q_TABLE[last_state][last_action]
        new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT_FACTOR * max_future_q)
        Q_TABLE[last_state][last_action] = new_q
    
    if np.random.random() < EXPLORATION_RATE:
        action = np.random.choice(["increase", "decrease", "maintain"])
    else:
        action = max(Q_TABLE[current_state], key=Q_TABLE[current_state].get)
    
    if action == "increase":
        return 1.2, current_state, action
    elif action == "decrease":
        return 0.8, current_state, action
    else:
        return 1.0, current_state, action

class Node:
    def __init__(self, score):
        self.score = score
        self.next = None

class ScoreLinkedList:
    def __init__(self):
        self.head = None
    
    def append(self, score):
        new_node = Node(score)
        if not self.head:
            self.head = new_node
        else:
            current = self.head
            while current.next:
                current = current.next
            current.next = new_node
    
    def to_list(self):
        scores = []
        current = self.head
        while current:
            scores.append(current.score)
            current = current.next
        return scores
    
    def quick_sort(self, scores):
        if len(scores) <= 1:
            return scores
        pivot = scores[len(scores) // 2]
        left = [x for x in scores if x > pivot]
        middle = [x for x in scores if x == pivot]
        right = [x for x in scores if x < pivot]
        return self.quick_sort(left) + middle + self.quick_sort(right)

class GeneticAlgorithm:
    def __init__(self, population_size=10):
        self.population_size = population_size
        self.population = self.initialize_population()
        self.generation = 1
        self.best_specimen = None
    
    def initialize_population(self):
        population = []
        for _ in range(self.population_size):
            specimen = {
                "speed": random.uniform(0.8, 1.5),
                "size": random.uniform(0.7, 1.3),
                "dodge_ability": random.uniform(0, 1),
                "fitness": 0
            }
            population.append(specimen)
        return population
    
    def evaluate_fitness(self, specimen, survival_time, passed_bottom):
        fitness = survival_time
        if passed_bottom:
            fitness *= 2
        specimen["fitness"] = fitness
        
        if not self.best_specimen or fitness > self.best_specimen["fitness"]:
            self.best_specimen = specimen.copy()
    
    def crossover(self, parent1, parent2):
        child = {}
        for attribute in ["speed", "size", "dodge_ability"]:
            if random.random() < 0.5:
                child[attribute] = parent1[attribute]
            else:
                child[attribute] = parent2[attribute]
        
        if random.random() < 0.2:
            attribute = random.choice(["speed", "size", "dodge_ability"])
            mutation = random.uniform(-0.2, 0.2)
            child[attribute] += mutation
            child[attribute] = max(0, min(1, child[attribute]))
        
        child["fitness"] = 0
        return child
    
    def next_generation(self):
        sorted_population = sorted(self.population, key=lambda x: x["fitness"], reverse=True)
        
        elites = sorted_population[:2]
        
        new_population = elites.copy()
        
        while len(new_population) < self.population_size:
            parent1 = random.choice(elites)
            parent2 = random.choice(sorted_population[:5])
            child = self.crossover(parent1, parent2)
            new_population.append(child)
        
        self.population = new_population
        self.generation += 1

DIFFICULTY_EASY = {"alien_speed": 1.0, "bullet_limit": 5, "dodge_chance": 0.3}
DIFFICULTY_MEDIUM = {"alien_speed": 1.5, "bullet_limit": 4, "dodge_chance": 0.5}
DIFFICULTY_HARD = {"alien_speed": 2.0, "bullet_limit": 3, "dodge_chance": 0.7}
current_difficulty = DIFFICULTY_MEDIUM

def select_difficulty():
    global current_difficulty
    
    screen.fill(BLACK)
    title = font.render("SELECT DIFFICULTY", True, WHITE)
    title_rect = title.get_rect(center=(WIDTH // 2, 100))
    
    easy_text = font.render("EASY", True, GREEN)
    easy_rect = easy_text.get_rect(center=(WIDTH // 2, 250))
    
    medium_text = font.render("MEDIUM", True, YELLOW)
    medium_rect = medium_text.get_rect(center=(WIDTH // 2, 350))
    
    hard_text = font.render("HARD", True, RED)
    hard_rect = hard_text.get_rect(center=(WIDTH // 2, 450))
    
    screen.blit(title, title_rect)
    screen.blit(easy_text, easy_rect)
    screen.blit(medium_text, medium_rect)
    screen.blit(hard_text, hard_rect)
    
    pygame.display.update()
    
    waiting = True
    while waiting:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.MOUSEBUTTONDOWN:
                if easy_rect.collidepoint(event.pos):
                    current_difficulty = DIFFICULTY_EASY
                    waiting = False
                elif medium_rect.collidepoint(event.pos):
                    current_difficulty = DIFFICULTY_MEDIUM
                    waiting = False
                elif hard_rect.collidepoint(event.pos):
                    current_difficulty = DIFFICULTY_HARD
                    waiting = False

def save_score(score, misses):
    scores_list = ScoreLinkedList()
    try:
        with open("highscores.txt", "r") as file:
            for line in file.readlines():
                scores_list.append(int(line.strip()))
    except FileNotFoundError:
        pass
    
    scores_list.append(score)
    sorted_scores = scores_list.quick_sort(scores_list.to_list())[:10]
    
    with open("highscores.txt", "w") as file:
        for s in sorted_scores:
            file.write(f"{s}\n")
    
    data.append([score, misses])
    target.append(int(score / 10))
    knn_model.fit(data, target)

def load_scores():
    scores_list = ScoreLinkedList()
    try:
        with open("highscores.txt", "r") as file:
            for line in file.readlines():
                scores_list.append(int(line.strip()))
    except FileNotFoundError:
        return []
    return scores_list.quick_sort(scores_list.to_list())

def spawn_alien():
    return pygame.Rect(random.randint(0, WIDTH - ALIEN_SIZE), random.randint(0, HEIGHT // 2), ALIEN_SIZE, ALIEN_SIZE)

def display_scores():
    scores = load_scores()
    screen.fill(BLACK)
    y_offset = 100
    for idx, score in enumerate(scores):
        text = font.render(f"{idx+1}. Score: {score}", True, WHITE)
        screen.blit(text, (WIDTH // 3, y_offset))
        y_offset += 30
    pygame.display.update()
    pygame.time.delay(3000)

def game_over_screen(score):
    screen.fill(BLACK)
    text = font.render(f"Game Over! Score: {score}", True, WHITE)
    screen.blit(text, (WIDTH // 3, HEIGHT // 3))
    pygame.display.update()
    pygame.time.delay(2000)
    display_scores()
    
    restart_text = font.render("Press 'R' to Restart or any other key to Exit", True, WHITE)
    restart_rect = restart_text.get_rect(center=(WIDTH // 2, HEIGHT - 50))
    screen.blit(restart_text, restart_rect)
    pygame.display.update()
    
    waiting = True
    while waiting:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    return True
                else:
                    return False

def show_instructions():
    screen.fill(BLACK)
    
    instructions = [
        "Use LEFT and RIGHT arrow keys to move your spaceship.",
        "Hold SHIFT for faster movement or CTRL for precise control.",
        "Press SPACE to fire bullets.",
        "Destroy the alien invaders before they reach the bottom.",
        "If an alien reaches the bottom, the game is over.",
        "Press any key to continue!"
    ]
    
    y_offset = 150
    for line in instructions:
        text = font.render(line, True, WHITE)
        text_rect = text.get_rect(center=(WIDTH // 2, y_offset))
        screen.blit(text, text_rect)
        y_offset += 30
    
    pygame.display.update()
    
    waiting = True
    while waiting:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                waiting = False

def smart_alien_movement(alien, player_x, score):
    dx = 0
    dy = 1
    
    if score < 5:
        
        pass
    elif 5 <= score < 15:
        
        if alien.x < player_x:
            dx = 0.3
        else:
            dx = -0.3
    else:
        
        if alien.x < player_x:
            dx = 0.5
        else:
            dx = -0.5
    
    return {"dx": dx, "dy": dy}

def main():
    global player_x, bullets, alien, ALIEN_SPEED, misses, speed_multiplier
    
    genetic_algo = GeneticAlgorithm()
    current_specimen = genetic_algo.population[0]
    last_state = None
    last_action = None
    
    
    show_instructions()
    select_difficulty()
    
    restart = True
    while restart:
        player_x = WIDTH // 2
        bullets = []
        alien = spawn_alien()
        alien_start_time = pygame.time.get_ticks()
        misses = 0
        speed_multiplier = 1.0
        running = True
        score = 0
        clock = pygame.time.Clock()
        last_bullet_time = 0
        
        while running:
            if has_images:
                screen.blit(background_img, (0, 0))
            else:
                screen.fill(BLACK)
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
            
            keys = pygame.key.get_pressed()
            
            current_speed = PLAYER_BASE_SPEED
            if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
                current_speed = PLAYER_BASE_SPEED * 1.8
            elif keys[pygame.K_LCTRL] or keys[pygame.K_RCTRL]:
                current_speed = PLAYER_BASE_SPEED * 0.6
                
            if keys[pygame.K_LEFT] and player_x > 0:
                player_x -= current_speed
            if keys[pygame.K_RIGHT] and player_x < WIDTH - PLAYER_SIZE:
                player_x += current_speed
                
            current_time = pygame.time.get_ticks()
            if keys[pygame.K_SPACE] and current_time - last_bullet_time > BULLET_COOLDOWN:
                if len(bullets) < current_difficulty["bullet_limit"]:
                    bullet = pygame.Rect(player_x + PLAYER_SIZE // 2 - BULLET_SIZE // 2, player_y, BULLET_SIZE, BULLET_SIZE * 2)
                    bullets.append(bullet)
                    last_bullet_time = current_time
            
            
            
            bullets = [b for b in bullets if b.y >= 0 and b.width > 0 and b.height > 0]
            for bullet in bullets:
                bullet.y -= BULLET_SPEED
            
            reward = 0.1
            difficulty_factor, current_state, action = q_learning_difficulty_adjustment(
                score, misses, last_state, last_action, reward
            )
            last_state = current_state
            last_action = action
            
            movement = smart_alien_movement(alien, player_x, score)
            alien.x += movement["dx"]
            
            base_speed = current_difficulty["alien_speed"]
            alien.y += int(base_speed * current_specimen["speed"] * difficulty_factor * movement["dy"])
            
            alien.x = max(0, min(WIDTH - ALIEN_SIZE, alien.x))
            
            if alien.y > HEIGHT:
                genetic_algo.evaluate_fitness(
                    current_specimen, 
                    (pygame.time.get_ticks() - alien_start_time) / 1000, 
                    True
                )
                misses += 1
                if misses >= 1:
                    running = False
                alien = spawn_alien()
                alien_start_time = pygame.time.get_ticks()
                speed_multiplier *= 1.05
                current_specimen = random.choice(genetic_algo.population)
            
            for bullet in bullets[:]:
                
                dodge_chance = current_specimen["dodge_ability"] * current_difficulty["dodge_chance"]
                if (bullet.y - alien.y < HEIGHT * 0.3 and
                    random.random() < dodge_chance):
                    
                    dodge_direction = 1 if bullet.x < alien.x + ALIEN_SIZE/2 else -1
                    alien.x += dodge_direction * 3
                    alien.x = max(0, min(WIDTH - ALIEN_SIZE, alien.x))
                
                if bullet.colliderect(alien):
                    genetic_algo.evaluate_fitness(
                        current_specimen, 
                        (pygame.time.get_ticks() - alien_start_time) / 1000, 
                        False
                    )
                    bullets.remove(bullet)
                    alien = spawn_alien()
                    alien_start_time = pygame.time.get_ticks()
                    score += 1
                    current_specimen = random.choice(genetic_algo.population)
                    q_learning_difficulty_adjustment(score, misses, last_state, last_action, 1.0)
            
            if has_images:
                
                screen.blit(spaceship_img, (player_x, player_y))
                
                if has_images and missile_img:
                    
                    for bullet in bullets:
                        screen.blit(missile_img, (bullet.x, bullet.y))
                else:
                    
                    for bullet in bullets:
                        pygame.draw.rect(screen, BLUE, bullet)
                
                alien_size = int(ALIEN_SIZE * current_specimen["size"])
                scaled_asteroid = pygame.transform.scale(asteroid_img, (alien_size, alien_size))
                screen.blit(scaled_asteroid, 
                           (alien.x + (ALIEN_SIZE - alien_size)//2, 
                            alien.y + (ALIEN_SIZE - alien_size)//2))
            else:
                
                pygame.draw.polygon(screen, GREEN, [(player_x + PLAYER_SIZE // 2, player_y), 
                                                  (player_x, player_y + PLAYER_SIZE), 
                                                  (player_x + PLAYER_SIZE, player_y + PLAYER_SIZE)])
                for bullet in bullets:
                    pygame.draw.rect(screen, BLUE, bullet)
                
                alien_size = int(ALIEN_SIZE * current_specimen["size"])
                alien_display = pygame.Rect(
                    alien.x + (ALIEN_SIZE - alien_size)//2, 
                    alien.y + (ALIEN_SIZE - alien_size)//2, 
                    alien_size, alien_size
                )
                pygame.draw.rect(screen, BLUE, alien_display)
            
            text = font.render(f"Score: {score}", True, WHITE)
            screen.blit(text, (10, 10))
            
            if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
                speed_text = font.render("Speed: FAST", True, RED)
            elif keys[pygame.K_LCTRL] or keys[pygame.K_RCTRL]:
                speed_text = font.render("Speed: SLOW", True, GREEN)
            else:
                speed_text = font.render("Speed: NORMAL", True, WHITE)
            screen.blit(speed_text, (WIDTH - 200, 10))
            
            ai_text = font.render(f"Generation: {genetic_algo.generation}", True, WHITE)
            screen.blit(ai_text, (10, 40))
            
            pygame.display.update()
            clock.tick(60)
        
        genetic_algo.next_generation()
        
        save_score(score, misses)
        restart = game_over_screen(score)
        # Save best specimen genome
        with open(\"best_specimen.pkl\", \"wb\") as f:
            pickle.dump(genetic_algo.best_specimen, f)
    
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()