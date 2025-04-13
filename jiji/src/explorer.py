"""
Maze Explorer module that implements automated maze solving.
"""

import time
import pygame
from typing import Tuple, List, Optional, Deque
from collections import deque
from .constants import BLUE, WHITE, CELL_SIZE, WINDOW_SIZE
from queue import PriorityQueue
import random
import heapq
import numpy as np

class Explorer:
    def __init__(self, maze, visualize: bool = False):
        self.maze = maze
        self.x, self.y = maze.start_pos
        self.direction = (1, 0)  # Start facing right
        self.moves = []
        self.start_time = None
        self.end_time = None
        self.visualize = visualize
        self.move_history = deque(maxlen=3)  # Keep track of last 3 moves
        self.backtracking = False
        self.backtrack_path = []
        self.backtrack_count = 0  # Count number of backtrack operations
        if visualize:
            pygame.init()
            self.screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
            pygame.display.set_caption("Maze Explorer - Automated Solving")
            self.clock = pygame.time.Clock()

    def turn_right(self):
        """Turn 90 degrees to the right."""
        x, y = self.direction
        self.direction = (-y, x)

    def turn_left(self):
        """Turn 90 degrees to the left."""
        x, y = self.direction
        self.direction = (y, -x)

    def can_move_forward(self) -> bool:
        """Check if we can move forward in the current direction."""
        dx, dy = self.direction
        new_x, new_y = self.x + dx, self.y + dy
        return (0 <= new_x < self.maze.width and 
                0 <= new_y < self.maze.height and 
                self.maze.grid[new_y][new_x] == 0)

    def move_forward(self):
        """Move forward in the current direction."""
        dx, dy = self.direction
        self.x += dx
        self.y += dy
        current_move = (self.x, self.y)
        self.moves.append(current_move)
        self.move_history.append(current_move)
        if self.visualize:
            self.draw_state()

    def is_stuck(self) -> bool:
        """Check if the explorer is stuck in a loop."""
        if len(self.move_history) < 3:
            return False
        # Check if the last 3 moves are the same
        return (self.move_history[0] == self.move_history[1] == self.move_history[2])

    def backtrack(self) -> bool:
        """Backtrack to the last position where we had multiple choices."""
        if not self.backtrack_path:
            # If we don't have a backtrack path, find one
            self.backtrack_path = self.find_backtrack_path()
        
        if self.backtrack_path:
            # Move to the next position in the backtrack path
            next_pos = self.backtrack_path.pop()
            self.x, self.y = next_pos
            self.backtrack_count += 1
            if self.visualize:
                self.draw_state()
            return True
        return False

    def find_backtrack_path(self) -> List[Tuple[int, int]]:
        """Find a path back to a position with multiple choices."""
        # Start from current position and go backwards through moves
        path = []
        current_pos = (self.x, self.y)
        visited = set()
        
        # Look for a position where we had multiple choices
        for i in range(len(self.moves) - 1, -1, -1):
            pos = self.moves[i]
            if pos in visited:
                continue
            visited.add(pos)
            path.append(pos)
            
            # Check if this position had multiple choices
            choices = self.count_available_choices(pos)
            if choices > 1:
                return path[::-1]  # Return reversed path
        
        return path[::-1]  # Return reversed path if no better position found

    def count_available_choices(self, pos: Tuple[int, int]) -> int:
        """Count the number of available moves from a position."""
        x, y = pos
        choices = 0
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            new_x, new_y = x + dx, y + dy
            if (0 <= new_x < self.maze.width and 
                0 <= new_y < self.maze.height and 
                self.maze.grid[new_y][new_x] == 0):
                choices += 1
        return choices

    def draw_state(self):
        """Draw the current state of the maze and explorer."""
        self.screen.fill(WHITE)
        
        # Draw maze
        for y in range(self.maze.height):
            for x in range(self.maze.width):
                if self.maze.grid[y][x] == 1:
                    pygame.draw.rect(self.screen, (0, 0, 0),
                                   (x * CELL_SIZE, y * CELL_SIZE,
                                    CELL_SIZE, CELL_SIZE))
        
        # Draw start and end points
        pygame.draw.rect(self.screen, (0, 255, 0),
                        (self.maze.start_pos[0] * CELL_SIZE,
                         self.maze.start_pos[1] * CELL_SIZE,
                         CELL_SIZE, CELL_SIZE))
        pygame.draw.rect(self.screen, (255, 0, 0),
                        (self.maze.end_pos[0] * CELL_SIZE,
                         self.maze.end_pos[1] * CELL_SIZE,
                         CELL_SIZE, CELL_SIZE))
        
        # Draw explorer
        pygame.draw.rect(self.screen, BLUE,
                        (self.x * CELL_SIZE, self.y * CELL_SIZE,
                         CELL_SIZE, CELL_SIZE))
        
        pygame.display.flip()
        self.clock.tick(30)  # Control visualization speed

    def print_statistics(self, time_taken: float):
        """Print detailed statistics about the exploration."""
        print("\n=== Maze Exploration Statistics ===")
        print(f"Total time taken: {time_taken:.2f} seconds")
        print(f"Total moves made: {len(self.moves)}")
        print(f"Number of backtrack operations: {self.backtrack_count}")
        print(f"Average moves per second: {len(self.moves)/time_taken:.2f}")
        print("==================================\n")

    # solve the maze using right hand
    def solve_right_hand(self) -> Tuple[float, List[Tuple[int, int]]]:
        """
        Solve the maze using the right-hand rule algorithm with backtracking.
        Returns the time taken and the list of moves made.
        """
        self.start_time = time.time()
        
        # Keep track of visited positions to detect loops
        visited = set()
        visited.add((self.x, self.y))
        
        if self.visualize:
            self.draw_state()
        
        while (self.x, self.y) != self.maze.end_pos:
            if self.is_stuck():
                # If stuck, try backtracking
                if not self.backtrack():
                    # If backtracking fails, try a different direction
                    self.turn_left()
                    self.turn_left()  # Turn around
                    self.move_forward()
                self.backtracking = True
            else:
                self.backtracking = False
                # Try to turn right first
                self.turn_right()
                if self.can_move_forward():
                    self.move_forward()
                    visited.add((self.x, self.y))
                else:
                    # If we can't move right, try forward
                    self.turn_left()
                    if self.can_move_forward():
                        self.move_forward()
                        visited.add((self.x, self.y))
                    else:
                        # If we can't move forward, try left
                        self.turn_left()
                        if self.can_move_forward():
                            self.move_forward()
                            visited.add((self.x, self.y))
                        else:
                            # If we can't move left, turn around
                            self.turn_left()
                            self.move_forward()
                            visited.add((self.x, self.y))

        # end time
        self.end_time = time.time()
        time_taken = self.end_time - self.start_time
        
        if self.visualize:
            # Show final state for a few seconds
            pygame.time.wait(2000)
            pygame.quit()
        
        # Print detailed statistics
        #self.print_statistics(time_taken)
            
        return time_taken, self.moves, self.backtrack_count

    # solve the maze using flood fill
    def solve_flood_fill(self) -> Tuple[float, List[Tuple[int, int]]]:
        """
        Solve the maze using the flood fill algorithm.
        Returns the time taken and the list of moves made.
        """
    
        self.start_time = time.time()
        
        # get the starting and ending postitions
        start = self.maze.start_pos
        end = self.maze.end_pos

        # get a queue with all the moves made
        queue = deque([start])

        # create a set of visited nodes
        visited = set()
        visited.add(start)
        came_from = {start: None}
    
        if self.visualize:
            self.draw_state()
    
        while queue:
            current = queue.popleft()
            self.x, self.y = current
    
            if self.visualize:
                self.draw_state()

            # stop if we reached the end
            if current == end:
                break
    
            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                neighbor = (current[0] + dx, current[1] + dy)
                x, y = neighbor
                if (0 <= x < self.maze.width and 0 <= y < self.maze.height and 
                    self.maze.grid[y][x] == 0 and neighbor not in visited):
                    queue.append(neighbor)
                    visited.add(neighbor)
                    came_from[neighbor] = current
    
        # Reconstruct path
        path = []
        current = end
        while current:
            path.append(current)
            current = came_from[current]
        path.reverse()
    
        self.moves = path
        
        # end time
        self.end_time = time.time()
    
        if self.visualize:
            pygame.time.wait(2000)
            pygame.quit()

        # set backtrack count to 0 becasue the algorithm doesnt backtrack
        self.backtrack_count = 0
        time_taken = self.end_time - self.start_time
        #self.print_statistics(time_taken)
        return time_taken, self.moves, self.backtrack_count

    # solve the maze using a star
    def solve_astar(self) -> Tuple[float, List[Tuple[int, int]], int, List[Tuple[int, int]]]:
        """
        Solve the maze using the A* algorithm
        Returns time taken, moves, backtrack count (always 0), and path
        """
        # start time
        self.start_time = time.time()

        # get the start and end positions
        start = self.maze.start_pos
        end = self.maze.end_pos

        # priority queue to store nodes to visit 
        open_set = []
        # stores the f score and the node
        heapq.heappush(open_set, (0, start))
        # stores the best parent for each node
        came_from = {}
        # store the cost from start to each node
        g_score = {start: 0}
    
        while open_set:
            # get the node with the lowest f score
            _, current = heapq.heappop(open_set)

            # reached the end so break
            if current == end:
                break

            # check all directions (neighbors)
            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                neighbor = (current[0] + dx, current[1] + dy)
                x, y = neighbor
                
                # ensure its not a wall or out of bounds
                if (0 <= x < self.maze.width and 0 <= y < self.maze.height and self.maze.grid[y][x] == 0):
                    # update the cost to get to the neighbor
                    tentative_g = g_score[current] + 1

                    # update the score if its better than the old score
                    if neighbor not in g_score or tentative_g < g_score[neighbor]:
                        came_from[neighbor] = current
                        g_score[neighbor] = tentative_g
                        # use manhattan distance to update f score
                        f_score = tentative_g + abs(neighbor[0] - end[0]) + abs(neighbor[1] - end[1])
                        heapq.heappush(open_set, (f_score, neighbor))
                        
        # reconstruct the path from end to start then reverse it
        path = []
        current = end
        while current in came_from:
            path.append(current)
            current = came_from[current]
        path.append(start)
        path.reverse()

        # update the moves
        self.moves = path

        # set backtrack count to 0 becasue the algorithm doesnt backtrack
        self.backtrack_count = 0
        
        # end time
        self.end_time = time.time()
        time_taken = self.end_time - self.start_time
        #self.print_statistics(self.end_time - self.start_time)

        # return the results
        return time_taken, self.moves, self.backtrack_count

    # helper function for GA
    def evaluate_path(self, start, end, genome):
        """
        A helper function used by the GA solver that evaluates the fitness of a genome by simulating moves through the maze
        """
        pos = start
        visited = [pos]
        for dx, dy in genome:
            next_pos = (pos[0]+dx, pos[1]+dy)
            if 0 <= next_pos[0] < self.maze.width and 0 <= next_pos[1] < self.maze.height and self.maze.grid[next_pos[1]][next_pos[0]] == 0:
                visited.append(next_pos)
                pos = next_pos
                if pos == end:
                    break
        score = -abs(pos[0]-end[0]) - abs(pos[1]-end[1])
        return score, visited
        
    # solve the maze using genetic algorithm
    def solve_ga(self) -> Tuple[float, List[Tuple[int, int]], int, List[Tuple[int, int]]]:
        """
        Solve the maze using a Genetic Algorithm.
        Evolves a population of random path directions toward the goal.
        """
        # start time
        self.start_time = time.time()

        # get the start and endign positions
        start = self.maze.start_pos
        end = self.maze.end_pos        

        # generate a random population of 50 individuals
        population = [random_genome() for _ in range(50)]

        # run for 50 generations
        for _ in range(50):
            # for every genome in the population, evaluate its fitness and store it in a list then sort the list in descending order
            scored = sorted([(self.evaluate_path(start, end, g), g) for g in population], key=lambda x: x[0][0], reverse=True) # explain what this does
            # select the top 10 best individuals
            population = [g for (score, g) in scored[:10]]

            # fill the rest of the population
            while len(population) < 50:
                # randomly select 2 individuals as parents and cross them over to create offspring
                p1, p2 = random.sample(population[:10], 2)
                # select a random crossover point and combine their genes
                crossover = random.randint(1, len(p1)-1)
                child = p1[:crossover] + p2[crossover:]
                # apply utation to the child with a 0.1 mutation rate
                if random.random() < 0.1:
                    child[random.randint(0, len(child)-1)] = random.choice([(1,0), (-1,0), (0,1), (0,-1)])
                # add the new offspring to the population
                population.append(child)

        # get the best path
        score, best_path = max([self.evaluate_path(start, end, g) for g in population], key=lambda x: x[0])
        # update the moves
        self.moves = best_path

        # set backtrack count to 0 becasue the algorithm doesnt backtrack
        self.backtrack_count = 0
        
        # end time
        self.end_time = time.time()
        time_taken = self.end_time - self.start_time
        #self.print_statistics(self.end_time - self.start_time)

        # return the results
        return time_taken, self.moves, self.backtrack_count

    # solve the maze using ant colony optitization
    def solve_aco(self) -> Tuple[float, List[Tuple[int, int]], int, List[Tuple[int, int]]]:
        """
        Solve the maze using Ant Colony Optimization.
        Ants explore probabilistically, guided by pheromone and heuristic.
        """
        self.start_time = time.time()
        start = self.maze.start_pos
        end = self.maze.end_pos
        pheromone = np.ones((self.maze.height, self.maze.width))
        alpha, beta, decay = 1, 2, 0.05
    
        best_path = []
        for _ in range(50):
            paths = [construct_path() for _ in range(20)]
            pheromone *= (1 - decay)
            for path in paths:
                if path and path[-1] == end:
                    reward = 1 / len(path)
                    for x, y in path:
                        pheromone[y][x] += reward
            paths = [p for p in paths if p and p[-1] == end]
            if paths:
                best_path = min(paths, key=len)
    
        self.moves = best_path

        # set backtrack count to 0 becasue the algorithm doesnt backtrack
        self.backtrack_count = 0

        # end time
        self.end_time = time.time()
        time_taken = self.end_time - self.start_time
        #self.print_statistics(self.end_time - self.start_time)
        return time_taken, self.moves, self.backtrack_count

    # solve the maze using particle swarm optimization
    def solve_pso(self) -> Tuple[float, List[Tuple[int, int]], int, List[Tuple[int, int]]]:
        """
        Solve the maze using Particle Swarm Optimization.
        Each path (particle) learns from its own and the swarm’s best experience.
        """
        self.start_time = time.time()
        start = self.maze.start_pos
        end = self.maze.end_pos
        
        swarm = [random_path() for _ in range(30)]
        personal_best = swarm[:]
        personal_best_scores = [fitness(p)[0] for p in swarm]
        global_best = personal_best[personal_best_scores.index(max(personal_best_scores))]
    
        for _ in range(50):
            for i in range(30):
                new_path = []
                for j in range(len(swarm[i])):
                    inertia = swarm[i][j]
                    cognitive = random.choice(personal_best[i])
                    social = random.choice(global_best)
                    new_path.append(random.choice([inertia, cognitive, social]))
                swarm[i] = new_path
                new_score, _ = fitness(new_path)
                if new_score > personal_best_scores[i]:
                    personal_best[i] = new_path
                    personal_best_scores[i] = new_score
            global_best = personal_best[personal_best_scores.index(max(personal_best_scores))]
    
        _, best_path = fitness(global_best)
        self.moves = best_path

        # set backtrack count to 0 becasue the algorithm doesnt backtrack
        self.backtrack_count = 0

        # end time
        self.end_time = time.time()
        time_taken = self.end_time - self.start_time
        #self.print_statistics(self.end_time - self.start_time)
        return time_taken, self.moves, self.backtrack_count


# helper function for PSO
def random_path(length=100):
    return [random.choice([(1,0), (-1,0), (0,1), (0,-1)]) for _ in range(length)]

# helper function for PSO
def simulate_path(path):
    pos = start
    visited = [pos]
    for dx, dy in path:
        next_pos = (pos[0]+dx, pos[1]+dy)
        if 0 <= next_pos[0] < self.maze.width and 0 <= next_pos[1] < self.maze.height and self.maze.grid[next_pos[1]][next_pos[0]] == 0:
            pos = next_pos
            visited.append(pos)
            if pos == end:
                break
    return visited

# helper function for PSO
def fitness(path):
    visited = simulate_path(path)
    last = visited[-1]
    return - (abs(last[0] - end[0]) + abs(last[1] - end[1])), visited

# helper function for ACO
def construct_path():
    path = [start]
    visited = set(path)
    pos = start
    while pos != end:
        neighbors = []
        probs = []
        for dx, dy in [(1,0), (-1,0), (0,1), (0,-1)]:
            next_pos = (pos[0]+dx, pos[1]+dy)
            if 0<=next_pos[0]<self.maze.width and 0<=next_pos[1]<self.maze.height and self.maze.grid[next_pos[1]][next_pos[0]]==0 and next_pos not in visited:
                neighbors.append(next_pos)
                pher = pheromone[next_pos[1]][next_pos[0]] ** alpha
                heur = 1 / (abs(next_pos[0] - end[0]) + abs(next_pos[1]-end[1]) + 1) ** beta
                probs.append(pher * heur)
        if not neighbors:
            break
        probs = np.array(probs)
        probs /= probs.sum()
        choice = np.random.choice(len(neighbors), p=probs)
        pos = neighbors[choice]
        path.append(pos)
        visited.add(pos)
    return path

# helper function for GA
def random_genome(length=100):
    """
    A helper function used by the GA solver that generates a random sequence of directions
    this corresponds to the moves made by the explorer (by default 100 moves)
    """
    return [random.choice([(1,0), (-1,0), (0,1), (0,-1)]) for _ in range(length)]

