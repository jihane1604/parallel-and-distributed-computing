import numpy as np
import pandas as pd
import time
import multiprocessing
from itertools import repeat
from src.genetic_algorithms_functions import (
    parallel_fitness_evaluation,
    calculate_fitness,
    select_in_tournament,
    order_crossover,
    mutate,
    generate_unique_population  # assumes this version accepts an optional past_routes argument
)

def process_subpopulation(chunk, distance_matrix, num_tournaments_chunk, tournament_size, mutation_rate, num_nodes):
    """
    Process a subpopulation chunk: calculate fitness, run tournament selection, 
    perform crossover and mutation, and ensure uniqueness.
    
    Parameters:
      - chunk: sub-list of individuals (routes)
      - distance_matrix: the distance matrix
      - num_tournaments_chunk: number of tournaments to run for this chunk
      - tournament_size: number of individuals in each tournament
      - mutation_rate: probability of mutation per offspring
      - num_nodes: total nodes (used when creating new individuals)
    
    Returns:
      - A new subpopulation (list of routes) updated for this generation.
    """
    # Calculate fitness for each individual in the chunk.
    fitness_values = np.array([-calculate_fitness(route, distance_matrix) for route in chunk])
    
    # Run tournament selection on this chunk.
    selected = select_in_tournament(chunk, fitness_values, num_tournaments_chunk, tournament_size)
    
    # Crossover: pair selected winners (if odd, the last one is skipped).
    offspring = []
    for i in range(0, len(selected) - 1, 2):
        parent1, parent2 = selected[i], selected[i+1]
        # Exclude the fixed starting node for crossover.
        child_route = order_crossover(parent1[1:], parent2[1:])
        offspring.append([0] + child_route)
    
    # Apply mutation to the offspring.
    mutated_offspring = [mutate(route, mutation_rate) for route in offspring]
    
    # Replacement: replace the worst individuals in the chunk with the mutated offspring.
    indices_to_replace = np.argsort(fitness_values)[::-1][:len(mutated_offspring)]
    for j, idx in enumerate(indices_to_replace):
        chunk[idx] = mutated_offspring[j]
    
    # Ensure the subpopulation is unique.
    unique_chunk = set(tuple(ind) for ind in chunk)
    while len(unique_chunk) < len(chunk):
        # Generate new individual(s) if needed (using num_nodes).
        new_individual = [0] + list(np.random.permutation(np.arange(1, num_nodes)))
        unique_chunk.add(tuple(new_individual))
    new_chunk = [list(ind) for ind in unique_chunk]
    
    return new_chunk

def run_parallel():
    start_time = time.time()
    
    # Load the distance matrix.
    distance_matrix = pd.read_csv('data/city_distances.csv').to_numpy()
    
    # Parameters.
    num_nodes = distance_matrix.shape[0]
    population_size = 15000
    # num_tournaments = 100      # (overall, used in sequential version)
    tournament_size = 10       # individuals per tournament
    mutation_rate = 0.1
    num_generations = 300
    stagnation_limit = 5      # generations without improvement before regeneration
    
    # Generate initial population: each route starts at node 0.
    np.random.seed(42)
    population = generate_unique_population(population_size, num_nodes)
    past_routes = population[:]  # history of routes
    
    best_overall_fitness = 1e6
    stagnation_counter = 0
    
    # We'll split the population into 6 chunks (one per processor).
    num_chunks = 100
    # For each chunk, we can run a fixed number of tournaments.
    num_tournaments_chunk = 100
    
    # Create a multiprocessing pool with 6 processes.
    pool = multiprocessing.Pool(processes=num_chunks)
    
    for generation in range(num_generations):
        # Evaluate overall fitness (sequentially) to check for stagnation.
        fitness_values = np.array([-calculate_fitness(route, distance_matrix) for route in population])
        # fitness_values = parallel_fitness_evaluation(population, distance_matrix, processes=6, chunk_size=100)
        current_best_fitness = np.min(fitness_values)
        if current_best_fitness < best_overall_fitness:
            best_overall_fitness = current_best_fitness
            stagnation_counter = 0
        else:
            stagnation_counter += 1
        
        # Regenerate population if stagnation is reached (preserving top 10 individuals).
        if stagnation_counter >= stagnation_limit:
            print(f"Regenerating population at generation {generation} due to stagnation")
            best_indices = np.argsort(fitness_values)[:10]
            best_individuals = [population[i] for i in best_indices]
            new_pop = generate_unique_population(population_size - 10, num_nodes)
            past_routes.extend(new_pop)
            population = new_pop + best_individuals
            stagnation_counter = 0
            continue
        
        # Split the population into chunks (subpopulations).
        # Using np.array_split preserves order, then convert to lists.
        chunks = [list(chunk) for chunk in np.array_split(population, num_chunks)]
        
        # Process each chunk in parallel.
        results = pool.starmap_async(
            process_subpopulation,
            [(chunk, distance_matrix, num_tournaments_chunk, tournament_size, mutation_rate, num_nodes)
             for chunk in chunks]
        ).get()
        
        # Combine the processed subpopulations back into one population.
        population = []
        for subpop in results:
            population.extend(subpop)
        
        # In case uniqueness checks reduced the subpopulation sizes,
        # ensure the overall population reaches the target size.
        if len(population) < population_size:
            additional = generate_unique_population(population_size - len(population), num_nodes, past_routes)
            population.extend(additional)
        
        print(f"Generation {generation}: Best fitness = {current_best_fitness}")
    
    # Final evaluation.
    fitness_values = np.array([-calculate_fitness(route, distance_matrix) for route in population])
    best_idx = np.argmin(fitness_values)
    best_solution = population[best_idx]
    
    end_time = time.time()
    
    print("Best Solution:", best_solution)
    print("Total Distance:", -calculate_fitness(best_solution, distance_matrix))
    print("Total time taken:", end_time - start_time)
    
    pool.close()
    pool.join()
    return end_time - start_time

if __name__ == "__main__":
    run_parallel()
