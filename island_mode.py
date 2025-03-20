"""
This file runs the fitness calculation, tournament selection, offspring crossover, and offspring mutation and undergoes the evolutionary process
"""

import numpy as np
import pandas as pd
import time
import multiprocessing
from itertools import repeat
from src.genetic_algorithms_functions import (
    calculate_fitness,
    select_in_tournament,
    order_crossover,
    mutate,
    generate_unique_population,
    run_ga_on_subpopulation
)



def run_parallel():
    start_time = time.time()
    
    # Load the distance matrix.
    distance_matrix = pd.read_csv('data/city_distances_extended.csv').to_numpy()
    
    # Parameters.
    num_nodes = distance_matrix.shape[0]
    population_size = 15000
    tournament_size = 4         # individuals per tournament
    mutation_rate = 0.1
    num_generations = 300
    stagnation_limit = 5
    num_tournaments = 10        # tournaments per subpopulation (can adjust as needed)
    
    # Split the initial population into subpopulations.
    np.random.seed(42)  # For reproducibility
    initial_population = generate_unique_population(population_size, num_nodes)
    num_chunks = 6  # assuming 6 processors
    subpopulations = [list(chunk) for chunk in np.array_split(initial_population, num_chunks)]
    
    # Process each subpopulation in parallel for all generations.
    pool = multiprocessing.Pool(processes=num_chunks)
    results = pool.starmap_async(
        run_ga_on_subpopulation,
        [(subpop, distance_matrix, num_generations, stagnation_limit, tournament_size,
          mutation_rate, num_nodes, num_tournaments) for subpop in subpopulations]
    ).get()
    
    # 'results' now contains the best individual from each subpopulation.
    # Choose the overall best solution.
    best_fitness_values = np.array([-calculate_fitness(route, distance_matrix) for route in results])
    overall_best_idx = np.argmin(best_fitness_values)
    overall_best = results[overall_best_idx]
    
    end_time = time.time()
    
    print("Best solutions from subpopulations:")
    for idx, sol in enumerate(results):
        print(f" Subpopulation {idx}: Distance = {-calculate_fitness(sol, distance_matrix)}")
    print("\nOverall Best Solution:", overall_best)
    print("Total Distance:", -calculate_fitness(overall_best, distance_matrix))
    print("Total time taken:", end_time - start_time)
    
    pool.close()
    pool.join()
    return end_time - start_time

if __name__ == "__main__":
    run_parallel()
