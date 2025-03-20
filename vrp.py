import numpy as np
import pandas as pd
import time
from src.genetic_algorithms_functions import select_in_tournament, order_crossover, mutate, generate_unique_population

# Splitting Procedure and VRP Fitness Evaluation
def calculate_fitness_vrp(chromosome, distance_matrix, demands, capacity):
    """
    Given a chromosome (a permutation of client nodes), this function splits
    it optimally into trips such that the total cost is minimized. A trip is 
    feasible if the total demand of clients in that trip does not exceed 'capacity'.
    
    The cost of a trip is computed as:
        depot -> first client + sum(inter-client distances) + last client -> depot.
    
    This dynamic programming procedure computes the minimal total cost to service 
    all clients in order.
    
    Args:
        chromosome (list): Permutation of clients (integers 1..N).
        distance_matrix (numpy.ndarray): Matrix of distances (index 0 is depot).
        demands (list): List of demands for nodes (index 0 is depot, then clients 1..N).
        capacity (int): Vehicle capacity.
    
    Returns:
        float: The minimal total cost (fitness) of the VRP solution corresponding to the chromosome.
    """
    n = len(chromosome)
    # cost[i] will hold the minimum cost to serve the first i clients (i=0 means no client served)
    cost = [float('inf')] * (n + 1)
    cost[0] = 0

    # Dynamic programming: for each starting point i, try to extend a trip to j (i < j <= n)
    for i in range(n):
        total_demand = 0
        # Initialize trip cost
        trip_cost = 0
        for j in range(i, n):
            client = chromosome[j]
            total_demand += demands[client]
            # If adding this client violates capacity, break out (cannot extend trip further)
            if total_demand > capacity:
                break
            # Compute trip cost for clients chromosome[i] to chromosome[j]:
            if j == i:
                # Only one client in the trip: depot -> client -> depot
                trip_cost = distance_matrix[0, client] + distance_matrix[client, 0]
            else:
                # When adding client j, update the cost incrementally:
                # Remove the old return from previous client to depot, add cost from previous client to current client,
                # then add the new return from current client to depot.
                prev_client = chromosome[j-1]
                trip_cost = trip_cost - distance_matrix[prev_client, 0] \
                            + distance_matrix[prev_client, client] \
                            + distance_matrix[client, 0]
            # Update cost for serving the first (j+1) clients:
            if cost[i] + trip_cost < cost[j+1]:
                cost[j+1] = cost[i] + trip_cost
    return cost[n]

# mian VRP GA loop
def run_vrp_ga():
    """
    Run a genetic algorithm for the Vehicle Routing Problem (VRP) using the 
    splitting procedure to evaluate chromosomes. The chromosome is a permutation 
    of clients (1..num_clients) and the depot is assumed to be node 0.
    
    The GA uses tournament selection, order crossover, and mutation (with local search 
    as mutation operator suggested in the article, here simplified as a swap mutation).
    
    Returns:
        float: Total execution time in seconds.
    """
    start_time = time.time()
    
    # Load the distance matrix from file (assumed CSV format)
    # The distance matrix should include the depot (index 0) and clients (1..N)
    distance_matrix = pd.read_csv('data/city_distances.csv').to_numpy()
    
    num_nodes = distance_matrix.shape[0]
    num_clients = num_nodes - 1  # clients are nodes 1..N
    
    # Generate (or load) demands for each node. For simplicity, random integers (1 to 10) for clients.
    # Depot demand is 0.
    demands = [0] + list(np.random.randint(1, 11, size=num_clients))
    
    # Vehicle capacity (for instance, 50)
    capacity = 50
    
    # GA parameters
    population_size = 15000
    num_tournaments = 10      # number of tournaments per generation
    tournament_size = 4       # individuals per tournament
    mutation_rate = 0.1
    num_generations = 300
    stagnation_limit = 5      # generations without improvement before regeneration
    
    np.random.seed(42)  # For reproducibility
    
    # Generate initial population: each chromosome is a permutation of clients 1..num_clients
    population = generate_unique_population(population_size, num_clients)
    
    best_fitness = float('inf')
    stagnation_counter = 0
    
    for generation in range(num_generations):
        # Evaluate fitness of each chromosome using the splitting procedure
        fitness_values = np.array([calculate_fitness_vrp(ind, distance_matrix, demands, capacity) for ind in population])
        current_best_fitness = np.min(fitness_values)
        
        # Check for improvement
        if current_best_fitness < best_fitness:
            best_fitness = current_best_fitness
            stagnation_counter = 0
        else:
            stagnation_counter += 1
        
        # Regenerate population if stagnation limit is reached, preserving the top 10 best individuals
        if stagnation_counter >= stagnation_limit:
            print(f"Regenerating population at generation {generation} due to stagnation")
            best_indices = np.argsort(fitness_values)[:10]
            best_individuals = [population[i] for i in best_indices]
            population = generate_unique_population(population_size - 10, num_clients)
            population.extend(best_individuals)
            stagnation_counter = 0
            continue
        
        # Tournament selection
        selected = select_in_tournament(population, fitness_values, number_tournaments=num_tournaments, tournament_size=tournament_size)
        
        # Crossover: pair selected individuals (if odd, the last is skipped)
        offspring = []
        for i in range(0, len(selected) - 1, 2):
            parent1, parent2 = selected[i], selected[i+1]
            child = order_crossover(parent1, parent2)
            offspring.append(child)
        
        # Mutate the offspring
        mutated_offspring = [mutate(child, mutation_rate) for child in offspring]
        
        # Replacement: replace the worst individuals with the mutated offspring
        worst_indices = np.argsort(fitness_values)[-len(mutated_offspring):]
        for j, idx in enumerate(worst_indices):
            population[idx] = mutated_offspring[j]
        
        # Ensure population uniqueness
        population = list({tuple(ind): list(ind) for ind in population}.values())
        while len(population) < population_size:
            new_ind = list(np.random.permutation(np.arange(1, num_clients+1)))
            population.append(new_ind)
        
        if generation % 100 == 0:
            print(f"Generation {generation}: Best VRP cost = {current_best_fitness}")
    
    # Final evaluation
    fitness_values = np.array([calculate_fitness_vrp(ind, distance_matrix, demands, capacity) for ind in population])
    best_idx = np.argmin(fitness_values)
    best_solution = population[best_idx]
    best_cost = fitness_values[best_idx]
    
    end_time = time.time()
    print("Best VRP solution (chromosome):", best_solution)
    print("Best VRP total cost:", best_cost)
    print("Total time taken:", end_time - start_time)
    
    return end_time - start_time

if __name__ == "__main__":
    run_vrp_ga()
