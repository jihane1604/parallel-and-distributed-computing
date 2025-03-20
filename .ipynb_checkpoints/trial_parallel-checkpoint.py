import numpy as np
import pandas as pd
from src.genetic_algorithms_functions import (
    calculate_fitness,
    select_in_tournament,
    order_crossover,
    mutate,
    generate_unique_population,
    parallel_fitness_evaluation
)
import time

def run_parallel():
    start_time = time.time()
    
    # Load the distance matrix from CSV.
    distance_matrix = pd.read_csv('data/city_distances.csv').to_numpy()
    
    # Parameters
    num_nodes = distance_matrix.shape[0]
    population_size = 15000
    num_tournaments = 6      # Number of tournaments to run
    tournament_size = 4
    mutation_rate = 0.1
    num_generations = 2000
    infeasible_penalty = 1e6 # Penalty for infeasible routes
    stagnation_limit = 5     # Generations without improvement before regeneration
    
    # Generate initial population: each individual starts at node 0.
    np.random.seed(42)  # For reproducibility
    population = generate_unique_population(population_size, num_nodes)
    
    # Initialize stagnation tracking.
    best_fitness = 1e6
    stagnation_counter = 0
    
    # Main GA loop.
    for generation in range(num_generations):
        # Evaluate fitness in parallel using chunking.
        calculate_fitness_values = parallel_fitness_evaluation(population, distance_matrix, processes=6, chunk_size=100)
        
        # Check for improvement (lower fitness is better).
        current_best_fitness = np.min(calculate_fitness_values)
        if current_best_fitness < best_fitness:
            best_fitness = current_best_fitness
            stagnation_counter = 0
        else:
            stagnation_counter += 1
    
        # If stagnation persists, regenerate population while keeping the best individual.
        if stagnation_counter >= stagnation_limit:
            print(f"Regenerating population at generation {generation} due to stagnation")
            best_individual = population[np.argmin(calculate_fitness_values)]
            population = generate_unique_population(population_size - 1, num_nodes)
            population.append(best_individual)
            stagnation_counter = 0
            continue  # Skip further operations this generation
    
        # Selection, crossover, and mutation.
        selected = select_in_tournament(population,
                                        calculate_fitness_values,
                                        num_tournaments,
                                        tournament_size)
        offspring = []
        # Process pairs of selected individuals.
        for i in range(0, len(selected), 2):
            parent1, parent2 = selected[i], selected[i + 1]
            # Perform order crossover on the routes (excluding the fixed starting node).
            route1 = order_crossover(parent1[1:], parent2[1:])
            offspring.append([0] + route1)
        mutated_offspring = [mutate(route, mutation_rate) for route in offspring]
    
        # Replacement: Replace the worst individuals with the new offspring.
        replace_indices = np.argsort(calculate_fitness_values)[::-1][:len(mutated_offspring)]
        for i, idx in enumerate(replace_indices):
            population[idx] = mutated_offspring[i]
    
        # Ensure population uniqueness.
        unique_population = set(tuple(ind) for ind in population)
        while len(unique_population) < population_size:
            individual = [0] + list(np.random.permutation(np.arange(1, num_nodes)))
            unique_population.add(tuple(individual))
        population = [list(individual) for individual in unique_population]
    
        print(f"Generation {generation}: Best fitness = {current_best_fitness}")
    
    # Final fitness evaluation and best solution selection.
    calculate_fitness_values = parallel_fitness_evaluation(population, distance_matrix, processes=6, chunk_size=100)
    best_idx = np.argmin(calculate_fitness_values)
    best_solution = population[best_idx]
    
    end_time = time.time()
    
    print("Best Solution:", best_solution)
    print("Total Distance:", -calculate_fitness(best_solution, distance_matrix))
    print("Total time taken:", end_time - start_time)
    
    return end_time - start_time

run_parallel()