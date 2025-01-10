import sys

import neat.nn
import pygame
import os
import random

pygame.init()

SCREEN_HEIGHT = 600
SCREEN_WIDTH = 1200
WHITE = (255, 255, 255)
GROUND_HEIGHT = 350

screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("DinoAI")

clock = pygame.time.Clock()
FPS = 30

BASE_PATH = os.path.dirname(__file__)
CACTUS_PATH = os.path.join(BASE_PATH, "Assets", "Cactus")
DINO_PATH = os.path.join(BASE_PATH, "Assets", "Dino")
OTHER_PATH = os.path.join(BASE_PATH, "Assets", "Other")

DINO_RUN1 = pygame.image.load(os.path.join(DINO_PATH, "DinoRun1.png"))
DINO_RUN2 = pygame.image.load(os.path.join(DINO_PATH, "DinoRun2.png"))
DINO_JUMP = pygame.image.load(os.path.join(DINO_PATH, "DinoJump.png"))
TRACK = pygame.image.load(os.path.join(OTHER_PATH, "Track.png"))

LARGE_CACTUS = [
    pygame.image.load(os.path.join(CACTUS_PATH, "LargeCactus1.png")),
    pygame.image.load(os.path.join(CACTUS_PATH, "LargeCactus2.png")),
    pygame.image.load(os.path.join(CACTUS_PATH, "LargeCactus3.png"))
]
SMALL_CACTUS = [
    pygame.image.load(os.path.join(CACTUS_PATH, "SmallCactus1.png")),
    pygame.image.load(os.path.join(CACTUS_PATH, "SmallCactus2.png")),
    pygame.image.load(os.path.join(CACTUS_PATH, "SmallCactus3.png"))
]

points = 0
game_speed = 20
obstacles = []
dinosaurs = []
ge = []
nets = []


class Dino:
    X_POS = 88
    Y_POS = GROUND_HEIGHT - 78
    JUMP_VEL = 9

    def __init__(self):
        self.run_img = [DINO_RUN1, DINO_RUN2]
        self.jump_img = DINO_JUMP
        self.dino_rect = self.run_img[0].get_rect()
        self.dino_rect.x = self.X_POS
        self.dino_rect.y = self.Y_POS
        self.is_jumping = False
        self.jump_vel = self.JUMP_VEL
        self.step_index = 0

    def update(self):
        if self.is_jumping:
            self.jump()
        else:
            self.run()

    def run(self):
        self.dino_rect.y = self.Y_POS
        self.step_index += 1
        if self.step_index >= 10:
            self.step_index = 0

    def jump(self):
        self.dino_rect.y -= self.jump_vel * 4
        self.jump_vel -= 0.8
        if self.jump_vel < -self.JUMP_VEL:
            self.is_jumping = False
            self.jump_vel = self.JUMP_VEL

    def draw(self, screen):
        if self.is_jumping:
            screen.blit(self.jump_img, (self.dino_rect.x, self.dino_rect.y))
        else:
            screen.blit(self.run_img[self.step_index // 5 % 2], (self.dino_rect.x, self.dino_rect.y))


class Obstacle:
    def __init__(self, image, type):
        self.image = image
        self.type = type
        self.rect = self.image[self.type].get_rect()
        self.rect.x = SCREEN_WIDTH

        if self.image == LARGE_CACTUS:
            self.rect.y = GROUND_HEIGHT - self.rect.height + 10
        elif self.image == SMALL_CACTUS:
            self.rect.y = GROUND_HEIGHT - self.rect.height + 10

    def update(self):
        global points, game_speed
        self.rect.x -= game_speed
        if self.rect.x < -self.rect.width:
            obstacles.pop(0)

    def draw(self, screen):
        screen.blit(self.image[self.type], self.rect)


class Ground:
    def __init__(self):
        self.image = TRACK
        self.x = 0
        self.y = GROUND_HEIGHT

    def update(self):
        global game_speed
        self.x -= game_speed
        if self.x < -SCREEN_WIDTH:
            self.x = 0

    def draw(self, screen):
        screen.blit(self.image, (self.x, self.y))
        screen.blit(self.image, (self.x + SCREEN_WIDTH, self.y))


def display_score():
    global points, game_speed
    points += 1
    if points % 100 == 0:
        game_speed += 2
    font = pygame.font.Font('freesansbold.ttf', 20)
    text = font.render(f"Speed: {game_speed} | Points: {points}", True, (0, 0, 0))
    screen.blit(text, (50, 50))

def remove(index):
    dinosaurs.pop(index)
    ge.pop(index)
    nets.pop(index)

def display_dino_info(generation, alive_count):
    font = pygame.font.Font('freesansbold.ttf', 20)
    # Dinosaurs Count
    text_alive = font.render(f"Dinosaurs Alive: {alive_count}", True, (0,0,0))
    screen.blit(text_alive, (50, SCREEN_HEIGHT -50))
    # Generation
    text_generation = font.render(f"Generation: {generation}", True, (0,0,0))
    screen.blit(text_generation, (50, SCREEN_HEIGHT - 80))

def eval_genomes(genomes, config):
    global game_speed, points, obstacles, dinosaurs, ge, nets
    clock = pygame.time.Clock()

    # Ground Object
    ground = Ground()
    obstacles = []
    dinosaurs = []
    ge = []
    nets = []

    for genome_id, genome in genomes:
        dinosaurs.append(Dino())
        nets.append(neat.nn.FeedForwardNetwork.create(genome, config))
        genome.fitness = 0
        ge.append(genome)

    run = True
    while run:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
        screen.fill(WHITE)

        # Ground is updating and drawing
        ground.update()
        ground.draw(screen)

        if len(obstacles) == 0:
            if random.randint(0, 1) == 0:
                obstacles.append(Obstacle(SMALL_CACTUS, random.randint(0, 2)))
            else:
                obstacles.append(Obstacle(LARGE_CACTUS, random.randint(0, 2)))
        for obstacle in obstacles:
            obstacle.update()
            obstacle.draw(screen)

        for i in range(len(dinosaurs) - 1, -1, -1):
            dinosaur = dinosaurs[i]

            # Calculate activation only if there is an obstacle
            if len(obstacles) > 0:
                output = nets[i].activate((dinosaur.dino_rect.y,
                                           abs(dinosaur.dino_rect.x - obstacles[0].rect.x)))
                if output[0] > 0.5 and not dinosaur.is_jumping and dinosaur.dino_rect.y == dinosaur.Y_POS:
                    dinosaur.is_jumping = True

            # Collision control
            if len(obstacles) > 0 and dinosaur.dino_rect.colliderect(obstacles[0].rect):
                ge[i].fitness -= 1
                remove(i)
                continue

            ge[i].fitness += 1
            dinosaur.update()
            dinosaur.draw(screen)

        # If all the dinosaurs are dead, it's game over.
        if len(dinosaurs) == 0:
            run = False  # End the loop

        display_score()
        generation = pop.generation + 1  # Current generation
        display_dino_info(generation, len(dinosaurs))
        pygame.display.update()
        clock.tick(30)


def run(config_path):
    global pop

    config = neat.config.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path
    )
    pop = neat.Population(config)
    pop.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)
    # NEAT loop
    pop.run(eval_genomes, 50)

if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config.txt')
    run(config_path)