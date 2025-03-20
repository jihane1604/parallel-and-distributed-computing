# Assignment 1 Part 2

Total sequential execution time for the normal dataset: 8.46 seconds -- distance 1224
Total sequential execution time for the extended dataset: 20.25 seconds -- distance -1000000 --> no path found

# Genetic Algorithm for TSP: Quick Overview

This program implements a genetic algorithm (GA) to solve a routing problem (e.g., the Traveling Salesman Problem) by evolving a population of candidate routes.

## Explanation of the `genetic_algorithm_trial` code:

- **Data Loading:**
  - Reads a distance matrix from a CSV file (`city_distances.csv`, `city_distances_extended.csv`) where each entry represents the distance between nodes, then turns it into a numpy array.
  
- **Population Initialization:**
  - Sets parameters: number of nodes, population size, number of tournaments, mutation rate, number of generations, stagnation limit and a large infeasability penealty.
  - Generates an initial population of 10,000 unique routes starting from node 0.

- **Main GA Loop (for each generation):**
  - Loop through the number of generations and do the following for each generation
  - **Fitness Calculation:** Calculate the fitness values for the each individual node in the population.
  - **Stagnation Check:** Pick the best fitness of the current population by using `np.max()` (because the `calculate_fitness` function returns a negative value), then compare it to the overall best fitness. Reset the stagnation counter if the current fitness is better, otherwise increment it. If no improvement is seen for 5 consecutive generations, regenerate the population (keeping the best individual).
  - **Selection:** Apply tournament selection via `select_in_tournament` to choose parent routes for the crossover phase.
  - **Crossover & Mutation:** 
    - Perform order crossover (ignoring the fixed starting node because we always start from node 0) to create offspring.
    - Mutate the offspring routes based on a defined mutation rate.
  - **Replacement:** Replace the worst individuals with the new mutated offspring.
  - **Uniqueness Maintenance:** Ensure the population remains unique by generating new individuals if necessary.

- **Final Output:**
  - Evaluates the final population to select and print the best route and its total distance.
  - Measures and prints the total execution time.

## Parallelizing and distribution the code 
- **Fitness Evaluation:** Each individual's fitness is computed by summing the distances along its route. Since the fitness of one route is independent of another, the population can be split among multiple processes/machines. Each process/machine computes fitness for its subset of routes in parallel, which reduces the overall computation time when the population size is large.
- **Tournament selection, crossover and mutation:** Atfer chunking the population into sevral (100) sub populations. For each one, 
- The generation loop cannot be parallelized or distributed because each generation depends on the previous one.
- **Multiprocessing pools:** As a first trial, `multiprocessing.pool()` was used with `starmap_async` in order to parallelize the task across multiple processors on a single machine.
- This resulted in a much slower execution time of *90.12 seconds* compared to the sequential time of *23.50 seconds*. We can assume that this is due to the process management overhead (just like in part one of the assignment).
- Speedup: 0.26
- Efficiency: 0.04
- **Distributed using MPI4PY:**

## Improvements of the genetic algorithm
- After further investigating the genetic algorithm, some of the parameters were increased
- **Population size:** A larger population increases genetic diversity, which can improve the chance of finding a good solution. However it increases the computational time. (15000)
- **Number of tournaments:** More tournaments can lead to a more robust selection process, ensuring that a diverse set of high-quality individuals are chosen. If set too low, the selection might not effectively differentiate among individuals, while too many may increase the computational overhead without proportional benefits. (10)
- **Tournament size:** A larger tournament size increases selection pressure because it raises the probability that the best individuals (with the highest fitness) are selected. However, excessive selection pressure can reduce diversity, leading to premature convergence on suboptimal solutions. (4)
- **Number of generations:** More generations provide the algorithm with more opportunities to evolve and improve the population, potentially yielding better solutions over time. However it also increases the runtime of the program. (3000) -- stopped changing after a while
- The best distance was 1050.0 (achieved starting from the 2255th generation)
- execution time = 237.48 seconds 