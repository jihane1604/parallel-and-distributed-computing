# Assignment 1 Part 2

Total sequential execution time fro the normal dataset: 8.11 seconds -- distance -1224
Total sequential execution time fro the extended dataset: 20.25 seconds -- distance -1000000

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

## Parallelizing the code 
- **a. Fitness Evaluation:** Each individual's fitness is computed by summing the distances along its route. Since the fitness of one route is independent of another, the population can be split among multiple processes/machines. Each process/machine computes fitness for its subset of routes in parallel, which reduces the overall computation time when the population size is large.
- As a first trial, `multiprocessing.pool()` was used with `starmap_async` in order to parallelize the task across multiple (6) processors on a single machine, which resulted in a much slower execution time of *100.92 seconds*. We can assume that this is due to the process management overhead (just like in part one of the assignment).
- 
- **b. Tournament Selection:** Each tournament (where a fixed number of individuals compete) is independent. Although in the fitness evaluation is the most computationally intensive part, the tournament selection can also be parallelized by running multiple tournaments simultaneously.
