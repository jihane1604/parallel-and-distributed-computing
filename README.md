# Assignment 2

## Questions

### Question 1 (10 points)
- The maze explorer uses the right hand rule algorithm. This algorithm is a classic maze-solving algorithm, which essentially works by placing your hand on the wall and turning right continuously until you can't move forward, then you move left instead. 
- The function defined in the `explorer.py` file always tries to move right. If it cant go right it moves forward. If that also fails, it turns left. If all options are blocked, it turns around and continues.
- The algorithm detects getting stuck in a loop by keeping track of the last 3 moves or positions of the explorer in a `dequeue` called `move_history`. Because if it stayed in the same position for the past 3 iterations, then that means there are walls surrounding it on the right, left and front, so it should turn around.
- If the explorer is stuck, then it backtracks to the last move that had multiple unexplored choices, in order to make another move. This is done by going through the `moves` list in reverse (from most recent to least recent) and creating a backtrack path to the decision point to return to. 
- One constraint is that this algorithm only wokrs if all the wallas are connected together (i.e. there are no free standing walls), otherwise the explorer might get stuck searching forever.
- At the end of the algorithm, the explorer prints a summary of the statistics computed including: the time taken to solve the maze, the total number of moves made, the numbe rof backtracking operations made and the average moves per second. 
- After running the algorithm on the static mode as well as the random mode with different heights and widths, I noticed that the algorithm runs in pretty much 0 seconds and doesn't backtrack. The number of average moves per second is very high because the algorithm is very fast. 

### Question 2 (30 points)
Modify the main program to run multiple maze explorers simultaneously. This is because we want to find the best route out of the maze. Your solution should:
1. Allow running multiple explorers in parallel
2. Collect and compare statistics from all explorers
3. Display a summary of results showing which explorer performed best

*Hints*:
- To get 20 points, use use multiprocessing.
- To get 30 points, use MPI4Py on multiple machines.
- Use Celery and RabbitMQ to distribute the exploration tasks. You will get full marks plus a bonus.
- Implement a task queue system
- Do not visualize the exploration, just run it in parallel
- Store results for comparison

**To answer this question:** 
1. Study the current explorer implementation
2. Design a parallel execution system
3. Implement task distribution
4. Create a results comparison system

### Question 3 (10 points)
Analyze and compare the performance of different maze explorers on the static maze. Your analysis should:

1. Run multiple explorers (at least 4 ) simultaneously on the static maze
2. Collect and compare the following metrics for each explorer:
   - Total time taken to solve the maze
   - Number of moves made
   - *Optional*:
     - Number of backtrack operations

3. What do you notice regarding the performance of the explorers? Explain the results and the observations you made.

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

### Final points 6 (10 points)
1. Solve the static maze in 150 moves or less to get 10 points.
2. Solve the static maze in 135 moves or less to get 15 points.
3. Solve the static maze in 130 moves or less to get 100% in your assignment.

### Bonus points
1. Fastest solver to get top  10% routes (number of moves)
2. Finding a solution with no backtrack operations
3. Least number of moves.