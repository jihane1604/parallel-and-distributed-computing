# Assignment 2

## Project Structure

```md
parallel-and-distributed-computing
├── distributed.py
├── machine.txt
├── main.py
├── maze_visualization.ipynb
├── README.md
├── requirements.txt
├── run_all.sh
├── src
│   ├── constants.py
│   ├── explorer.py
│   ├── final_visualization.py
│   ├── game.py
│   ├── __init__.py
│   ├── maze.py
│   ├── player.py
│   └── visualization.py
└── visuals
    ├── A_Star_Algorithm_visual.png
    ├── Best_First_Search_Algorithm_visual.png
    ├── Depth_First_Search_Algorithm_visual.png
    └── Right_Hand_Algorithm_visual.png
```

- The `explorer.py` file was edited in order to implement different search algorithms to solve the maze. The `distributed.py` file is an edited version of the `main.py` file which handles the distribution of the solvers across multiple machines. A `final_visualization.py` file was added in order to visualize the final path taken by the explorer, and the visualizations were stored in the `visuals` folder.

## Algorithm explanation
- The maze explorer uses the right hand rule algorithm. This algorithm is a classic maze-solving algorithm, which essentially works by placing your hand on the wall and turning right continuously until you can't move forward, then you move left instead. 
- The function defined in the `explorer.py` file always tries to move right. If it cant go right it moves forward. If that also fails, it turns left. If all options are blocked, it turns around and continues.
- The algorithm detects getting stuck in a loop by keeping track of the last 3 moves or positions of the explorer in a `dequeue` called `move_history`. Because if it stayed in the same position for the past 3 iterations, then that means there are walls surrounding it on the right, left and front, so it should turn around.
- If the explorer is stuck, then it backtracks to the last move that had multiple unexplored choices, in order to make another move. This is done by going through the `moves` list in reverse (from most recent to least recent) and creating a backtrack path to the decision point to return to. 
- One constraint is that this algorithm only wokrs if all the walls are connected together (i.e. there are no free standing walls), otherwise the explorer might get stuck searching forever.
- At the end of the algorithm, the explorer prints a summary of the statistics computed including: the time taken to solve the maze, the total number of moves made, the number of backtracking operations made and the average moves per second. 
- After running the algorithm on the static mode as well as the random mode with different heights and widths, I noticed that the algorithm runs in pretty much 0 seconds and doesn't backtrack. The number of average moves per second is very high because the algorithm is very fast. 

## Distribution of solvers

- I created a `distributed.py` file which runs multiple workers across machines using `mpi4py`. This is done by creating the maze in the root process (process 0) and broadcasting it to all the other processes using the `bcast` function.
- Note that the visualization and the interactive mode can only be played from the root process (though in this assignment I will only be running the automated mode with no visualizations)
- I started by running 4 explorers across 2 machines with the same explorer algorithm, and as expected I got the same result for each explorer.
- I implemented multiple algorithms to give each worker in order to properly compare their performance (the algorithms will be explained in detail later). I stored the algorithms in a dictionary for easier acces by the workers.
- The results returned by each worker are: the time taken, the number of moves, the number of backtracks and the complete path (for later visualizations). After each worker has finished executing the algorithm, the results are then collected in rank 0 using the `gather` function.
- At the end, root process displays the metrics of each worker, then displays which algorithm performed the best in terms of number of moves along with the algorithm that was used.

## Algorithms used

#### A* Algorithm

- A* is an informed search algorithm, which finds the shortest path from a start point to an end point, using both actual movement cost and estimated remaining cost. It does so by maintaining a tree of paths, from the starting node, and extending those paths one edge at a time, based on the cost, until the goal node is reached. Essentially, the algorithm aims to minimize the cost function defined by: `f(n) = g(n) + h(n)` where `g(n)` represents the cost of the path from the start node to the n, and `h(n)` is the heuristic value that estimates the cost of the cheapest path from n to the end node.
- In the `solve_astar` function, the algotihm uses a priority queue called an `open set`, that is stored in a min heap, to repeatedly select the nodes with the minimum cost to expand. From the current node, the algorithm keeps track of the preceding node, and checks all the valid neighbors (ensures its not a wall and not out of the bounds of the maze). For each neighbor, if the new path to this neighbor is cheaper, it updates the cost and parent pointer, then adds it back to the queue with the updated score. The hueristic value is calculated using the Euclidean distance. Once we have reached the end node, the algorithm stops, then it reconstruct the path by tracing parent nodes from goal to start and reversing the path.
- The algorithm does not perform any backtracks so the counter is set to 0 at the end.

#### Breadths First Search Algorithm

- BFS is an algorithm that can be used in maze solving by exploring all reachable areas layer by layer, ensuring that the shortest path is found in an unweighted graph. The algorithm starts at the initial position, and expands outward evenly in all directions, ensuring that the first time it reaches the goal, it has found the shortest possible path
- In the `solve_BFS` function, the algorithm keeps track of the cells to explore next using a double ended queu (following a fist in first out structure), along with keeping track of the previously visited cells (to prevent revisiting the same ones) and the preceding cells. At each node, the algorithm explores the possible neighbors (ensuring theyre valid and havent been visited yet), and saves the current node as the parent of that neighbor for later path reconstruction. Once the goal has been reached, the algorithm reconstructs the path just like in A*.
- This algorithm also has no backtracks so the backtrack count is set to 0 as well.
- While simple and reliable, BFS can become slow in large, open spaces due to its exhaustive nature, since it explores all possibilities at the same distance before going deeper.

#### Depth First Search Algorithm

- DFS is an algorithm that explores a maze or graph by going as deep as possible along one path before backtracking and trying alternative paths. In maze solving, DFS begins at the starting cell and follows one direction until it hits a dead end or the goal.
- In the `solve_dfs` function, the algorithm uses a stack (following last in first out) to keep track of the nodes to visit, as well as a set of previously visited nodes. A `came_from` dictionary is used to record the parent of each cell, allowing reconstruction of the full path once the goal is found. At each step, it checks the neighboring cells, ensuring their valid, and proceeds to the first unvisited valid neighbor it finds (that hasnt been visited yet). If no unvisited neighbors remain, it backtracks to the previous cell and tries the next direction.
- While DFS is memory-efficient and easy to implement, it does not guarantee the shortest path, especially in large or branching mazes, since it commits to one path until it fails.
- This algorithm can also be implemented recursively.

## Comparison And Analysis

- The righ hand algorithm is inefficient because it has a looping behavior, revisiting the same cells without keeping track of the previously visited ones. It also lacks any strategy to orient it towards the goal (it blindly follows the walls).

- The additional algorithms proposed take into consideration the end goal as a heuristic.

- To evaluate the different maze-solving strategies, four explorers were executed concurrently on the static maze: Right-Hand Rule, Breadth-First Search, A* Search, and Depth-First Search. Each explorer was analyzed using three metrics: **time taken**, **number of moves** and **number of backtracks**

Algorithm | Time Taken | Moves | Backtracks
|---|---|---|---|
Right-Hand Rule | 0.0014 sec | 1278 | 0
Breadth-First Search | 0.0015 sec | 127 | 0
A* Search | 0.0017 sec | 127 | 0
Depth-First Search | 0.0003 sec | 129 | 67

- The right hand algorithm performed the worst in terms of path length with 1278 moves made, which is 10 times more than others. This is expected, as it blindly follows walls and lacks global awareness.

- BFS and A* were tied for the best move count at 127, confirming that both are optimal for unweighted graphs. However, BFS is simpler to understadn and implment and doesnt require heuristic information, but A* would outperform in weighted or larger maps.

- DFS also reached the goal but with 129 moves and 67 backtracks, highlighting its tendency to explore deep but inefficient paths. It’s not optimal but still functional and better than the right hand algorithm.

- BFS and A* are clearly the most efficient and reliable for structured, unweighted mazes, and I reached under 130 moves so I should get 100% :D

- A trade-off in using any of the proposed algorithms over right hand is the introduction of memory usage. Since BFS, DFS and A* all need to store the paths to explore in some sort of data structure (double ended queue, priority queue and stack), theyr equire more memory. This isn't necessarily a problem in this assignment; however, in a more complex problem with many bigger mazes, this might become an issue.

## Visualization 
- I visualized the final path taken by each algorithm:
- Right Hand Algorithm
![Right hand ](/visuals/A_Star_Algorithm_visual.png "right hand")
- A Star Algorithm
![A star](/visuals/Best_First_Search_Algorithm_visual.png "a star")
- Breadth First Search Algorithm
![Breadth first search](/visuals/Depth_First_Search_Algorithm_visual.png "bfs")
- Depth First Search Algorithm
![Depth first search](/visuals/Right_Hand_Algorithm_visual.png "dfs")