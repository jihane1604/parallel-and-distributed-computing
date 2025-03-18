# Assignment 1 Part 2

Total sequential execution time fro the normal dataset: 22.69 seconds
Total sequential execution time fro the extended dataset: 59.44 seconds 

# Genetic Algorithm for TSP: Quick Overview

This program implements a genetic algorithm (GA) to solve a routing problem (e.g., the Traveling Salesman Problem) by evolving a population of candidate routes.

## Key Components

- **Data Loading:**
  - Reads a distance matrix from a CSV file (`city_distances.csv`, `city_distances_extended.csv`) where each entry represents the distance between nodes.
  
- **Population Initialization:**
  - Sets parameters: number of nodes, population size, number of tournaments, mutation rate, number of generations, stagnation limit and a large infeasability penealty.
  - Generates an initial population of 10,000 unique routes starting from node 0.

- **Fitness Evaluation:**
  - Uses `calculate_fitness` to compute the negative total distance of each route.
  - Returns a large negative penalty if any segment has an infeasible distance (10000).

- **Main GA Loop (for each generation):**
  - **Fitness Calculation:** Compute fitness values for the entire population.
  - **Stagnation Check:** If no improvement is seen for 5 consecutive generations, regenerate the population (keeping the best individual).
  - **Selection:** Apply tournament selection via `select_in_tournament` to choose parent routes.
  - **Crossover & Mutation:** 
    - Perform order crossover (ignoring the fixed starting node) to create offspring.
    - Mutate the offspring routes based on a defined mutation rate.
  - **Replacement:** Replace the worst individuals with the new mutated offspring.
  - **Uniqueness Maintenance:** Ensure the population remains unique by generating new individuals if necessary.
  - **Logging:** Print the best fitness value for each generation.

- **Final Output:**
  - Evaluates the final population to select and print the best route and its total distance.
  - Measures and prints the total execution time.

This concise structure demonstrates the core GA operations: initialization, fitness evaluation, selection, crossover, mutation, and replacement to iteratively improve route quality.

## Parallelizing the code 
- **a. Fitness Evaluation:** Each individual's fitness is computed by summing the distances along its route. Since the fitness of one route is independent of another, the population can be split among multiple machines. Each machine computes fitness for its subset of routes in parallel, which reduces the overall computation time when the population size is large.
- **b. Tournament Selection:** Each tournament (where a fixed number of individuals compete) is independent. Although in the fitness evaluation is the most computationally intensive part, the tournament selection can also be parallelized by running multiple tournaments simultaneously.
