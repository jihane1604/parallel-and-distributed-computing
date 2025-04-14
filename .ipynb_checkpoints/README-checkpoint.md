# Assignment 2

## Questions

### Algorithm explanation
- The maze explorer uses the right hand rule algorithm. This algorithm is a classic maze-solving algorithm, which essentially works by placing your hand on the wall and turning right continuously until you can't move forward, then you move left instead. 
- The function defined in the `explorer.py` file always tries to move right. If it cant go right it moves forward. If that also fails, it turns left. If all options are blocked, it turns around and continues.
- The algorithm detects getting stuck in a loop by keeping track of the last 3 moves or positions of the explorer in a `dequeue` called `move_history`. Because if it stayed in the same position for the past 3 iterations, then that means there are walls surrounding it on the right, left and front, so it should turn around.
- If the explorer is stuck, then it backtracks to the last move that had multiple unexplored choices, in order to make another move. This is done by going through the `moves` list in reverse (from most recent to least recent) and creating a backtrack path to the decision point to return to. 
- One constraint is that this algorithm only wokrs if all the walls are connected together (i.e. there are no free standing walls), otherwise the explorer might get stuck searching forever.
- At the end of the algorithm, the explorer prints a summary of the statistics computed including: the time taken to solve the maze, the total number of moves made, the number of backtracking operations made and the average moves per second. 
- After running the algorithm on the static mode as well as the random mode with different heights and widths, I noticed that the algorithm runs in pretty much 0 seconds and doesn't backtrack. The number of average moves per second is very high because the algorithm is very fast. 

### Distribution of solvers

- I created a `distributed.py` file which runs multiple workers across machines using `mpi4py`. This is done by creating the maze in the root process (process 0) and broadcasting it to all the other processes using the `bcast` function.
- Note that the visualization and the interactive mode can only be played from the root process (though in this assignment I will only be running the automated mode with no visualizations)
- I started by running 4 explorers across 2 machines with the same explorer algorithm, and as expected I got the same result for each explorer.
- I implemented multiple algorithms to give each worker in order to properly compare their performance (the algorithms will be explained in detail later). I stored the algorithms in a dictionary for easier acces by the workers.
- The results returned by each worker are: the time taken, the number of moves, the number of backtracks and the complete path (for later visualizations). After each worker has finished executing the algorithm, the results are then collected in rank 0 using the `gather` function.
- At the end, root process displays the metrics of each worker, then displays which algorithm performed the best in terms of number of moves along with the algorithm that was used.

### Algorithms used

#### Flood Fill Algorithm

#### A* Algorithm

- A* is an informed search algorithm, which finds the shortest path from a start point to an end point, using both actual movement cost and estimated remaining cost. It does so by maintaining a tree of paths, from the starting node, and extending those paths one edge at a time, based on the cost, until the goal node is reached. Essentially, the algorithm aims to minimize the cost function defined by: `f(n) = g(n) + h(n)` where `g(n)` represents the cost of the path from the start node to the n, and `h(n)` is the heuristic value that estimates the cost of the cheapest path from n to the end node.
- In the `solve_astar` function, the algotihm uses a priority queue called an `open set`, that is stored in a min heap, to repeatedly select the nodes with the minimum cost to expand. From the current node, the algorithm keeps track of the preceding node, and checks all the valid neighbors (ensures its not a wall and not out of the bounds of the maze). For each neighbor, if the new path to this neighbor is cheaper, it updates the cost and parent pointer, then adds it back to the queue with the updated score. The hueristic value is calculated using the Euclidean distance. Once we have reached the end node, the algorithm stops, then it reconstruct the path by tracing parent nodes from goal to start.

### Question 3 (10 points)
Analyze and compare the performance of different maze explorers on the static maze. Your analysis should:

1. Run multiple explorers (at least 4 ) simultaneously on the static maze
2. Collect and compare the following metrics for each explorer:
   - Total time taken to solve the maze
   - Number of moves made
   - *Optional*:
     - Number of backtrack operations

3. What do you notice regarding the performance of the explorers? Explain the results and the observations you made.

- I used 3 different algorithms along with the right hand: flood fill, A*

### Question 4 (20 points)
Based on your analysis from Question 3, propose and implement enhancements to the maze explorer to overcome its limitations. Your solution should:

1. Identify and explain the main limitations of the current explorer:

2. Propose specific improvements to the exploration algorithm:

3. Implement at least two of the proposed improvements:

Your answer should include:
1. A detailed explanation of the identified limitations
2. Documentation of your proposed improvements
3. The modified code with clear comments explaining the changes

### Question 5 (20 points)

Compare the performance of your enhanced explorer with the original:
   - Run both versions on the static maze
   - Collect and compare all relevant metrics
   - Create visualizations showing the improvements
   - Document the trade-offs of your enhancements
Your answer should include:
1. Performance comparison results and analysis
2. Discussion of any trade-offs or new limitations introduced