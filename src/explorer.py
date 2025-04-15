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
import math
from typing import Tuple, List

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
        Solve the maze using the right-hand rule algorithm.
        
        This strategy follows the wall on the explorer’s right side to navigate through the maze.
        It may take longer routes and is not guaranteed to find the shortest path. Backtracking
        is implemented to recover from loops or dead ends. Returns the total time taken,
        the list of moves made, and the number of backtrack operations.
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

    # solve the maze using bfs
    def solve_bfs(self) -> Tuple[float, List[Tuple[int, int]]]:
        """
        Solve the maze using the Breadth-First Search (BFS) algorithm.
        
        BFS explores the maze level by level, guaranteeing the shortest path in unweighted mazes.
        It uses a queue to explore neighbors and a came_from dictionary to reconstruct the path.
        Returns the total time taken, the shortest path found, and a backtrack count of 0.
        """
        # start time
        self.start_time = time.time()
        
        # get the starting and ending postitions
        start = self.maze.start_pos
        end = self.maze.end_pos

        # get a queue with all the moves made
        queue = deque([start])

        # create a set of visited nodes
        visited = set()
        visited.add(start)
        # keep track of parent node for path recunstruction
        came_from = {start: None}
    
        if self.visualize:
            self.draw_state()

        # iterate while the queue is not empty
        while queue:
            current = queue.popleft()
            self.x, self.y = current
    
            if self.visualize:
                self.draw_state()

            # stop if we reached the end
            if current == end:
                break

            # check every neighboring node
            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                neighbor = (current[0] + dx, current[1] + dy)
                x, y = neighbor
                # ensure its a valid path
                if (0 <= x < self.maze.width and 0 <= y < self.maze.height and self.maze.grid[y][x] == 0 and neighbor not in visited):
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
        Solve the maze using the A* (A-Star) search algorithm.
        
        A* uses a priority queue and a heuristic (Euclidean distance) to guide its search
        toward the goal efficiently. It guarantees the shortest path if the heuristic is admissible.
        Returns the total time taken, the optimal path, a backtrack count of 0, and the move list.
        """
        # start time
        self.start_time = time.time()

        # get the start and end positions
        start = self.maze.start_pos
        end = self.maze.end_pos

        # priority queue to store nodes to visit 
        open_set = []
        # stores the f score and the node using a min heap
        heapq.heappush(open_set, (0, start))
        # stores the best parent for each node (to reconstruct the path at the end)
        came_from = {}
        # store the cost from start to each node
        g_score = {start: 0}

        if self.visualize:
            self.draw_state()
            
        while open_set:
            # get the node with the lowest f score
            score, current = heapq.heappop(open_set)

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
                        #f_score = tentative_g + abs(neighbor[0] - end[0]) + abs(neighbor[1] - end[1])
                        f_score = tentative_g + math.sqrt((neighbor[0] - end[0])**2 + (neighbor[1] - end[1])**2)
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

        if self.visualize:
            pygame.time.wait(2000)
            pygame.quit()
        
        # set backtrack count to 0 becasue the algorithm doesnt backtrack
        self.backtrack_count = 0
        
        # end time
        self.end_time = time.time()
        time_taken = self.end_time - self.start_time
        #self.print_statistics(self.end_time - self.start_time)

        # return the results
        return time_taken, self.moves, self.backtrack_count

    # solve using depth first search
    def solve_dfs(self) -> Tuple[float, List[Tuple[int, int]], int]:
        """
        Solve the maze using the Depth-First Search (DFS) algorithm.
        
        DFS explores as far as possible along each branch before backtracking. It is memory-efficient
        and fast but does not guarantee the shortest path. Multiple choice points are tracked to
        approximate the number of backtracks. Returns the time taken, the discovered path,
        and the number of backtracks.
        """
        # start time
        self.start_time = time.time()
    
        start = self.maze.start_pos
        end = self.maze.end_pos

        # use stack for lifo
        stack = [start] 
        visited = set()
        visited.add(start)
        # keep track of parent to reconstruct path later
        came_from = {start: None} 
    
        self.backtrack_count = 0
    
        if self.visualize:
            self.draw_state()
    
        while stack:
            current = stack.pop()
            self.x, self.y = current
    
            if self.visualize:
                self.draw_state()

            # stop if we reached the end
            if current == end:
                break
    
            neighbors = []
            # explore all neighboring cells
            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                neighbor = (current[0] + dx, current[1] + dy)
                x, y = neighbor
                # ensure its a valid path and not visited yet
                if (0 <= x < self.maze.width and 0 <= y < self.maze.height and self.maze.grid[y][x] == 0 and neighbor not in visited):
                    neighbors.append(neighbor)

            # backtrack when theres multiple options
            if len(neighbors) > 1:
                self.backtrack_count += 1
    
            for neighbor in neighbors:
                stack.append(neighbor)
                visited.add(neighbor)
                came_from[neighbor] = current
    
        # reconstruct the path
        path = []
        current = end
        while current:
            path.append(current)
            current = came_from.get(current)
        path.reverse()
    
        self.moves = path
        self.end_time = time.time()
    
        if self.visualize:
            import pygame
            pygame.time.wait(2000)
            pygame.quit()
    
        time_taken = self.end_time - self.start_time
        return time_taken, self.moves, self.backtrack_count
