import numpy as np
import multiprocessing
from itertools import repeat
from functools import partial
import concurrent.futures

def calculate_fitness(route,
                      distance_matrix):
    """
    calculate_fitness function: total distance traveled by the car.

    Parameters:
        - route (list): A list representing the order of nodes visited in the route.
        - distance_matrix (numpy.ndarray): A matrix of the distances between nodes.
            A 2D numpy array where the element at position [i, j] represents the distance between node i and node j.
    Returns:
        - float: The negative total distance traveled (negative because we want to minimize distance).
           Returns a large negative penalty if the route is infeasible.
    """
    total_distance = 0
    # iterate through consecutive node pairs in the route
    for i in range(len(route)):
        if i == len(route)-1:
            node1 = route[i]
            node2 = route[0]
        else:
            node1 = route[i]
            node2 = route[i+1]
        distance = distance_matrix[node1, node2]
        # if the distance is 100000, the route is infeasible: return a large negative penalty
        if distance == 100000:
            return -1e6
        total_distance += distance
    # return the negative total distance to make it a fitness value (minimizing distance)
    return -total_distance

def calculate_fitness_chunk(routes_chunk, distance_matrix):
    """
    Helper function: compute fitness for a chunk of routes.
    Returns a list of fitness values (negative of the calculated fitness).
    """
    # We return negative calculate_fitness so that lower total distances become lower numbers.
    return [-calculate_fitness(route, distance_matrix) for route in routes_chunk]

def parallel_fitness_evaluation(population, distance_matrix, chunk_size=100):
    """
    Evaluate fitness values in parallel using multiprocessing with chunking.
    
    Parameters:
        population (list): List of routes.
        distance_matrix (numpy.ndarray): The distance matrix.
        processes (int): Number of parallel processes.
        chunk_size (int): Number of routes per task.
        
    Returns:
        np.array: Array of fitness values (where lower is better).
    """
    # Split the population into chunks.
    chunks = [population[i:i + chunk_size] for i in range(0, len(population), chunk_size)]
    
    # Create a partial function binding the distance_matrix.
    # Note: This requires that the original function 'calculate_fitness_chunk'
    #       has a parameter named 'distance_matrix'.
    partial_func = partial(calculate_fitness_chunk, distance_matrix=distance_matrix)
    
    # with multiprocessing.Pool(processes = 12) as pool:
    #     # chunk_results = pool.starmap_async(calculate_fitness_chunk, zip(chunks, repeat(distance_matrix))).get()
    #     chunk_results = pool.map_async(partial_func, chunks).get()

    # Use ProcessPoolExecutor to process the chunks in parallel.
    with concurrent.futures.ProcessPoolExecutor(max_workers=12) as executor:
        # executor.map returns results in order corresponding to the input chunks.
        chunk_results = list(executor.map(partial_func, chunks))
        
    # Flatten the list of results.
    fitness_values = [fitness for chunk in chunk_results for fitness in chunk]
    return np.array(fitness_values)

def run_ga_on_subpopulation(subpop, distance_matrix, num_generations, stagnation_limit,
                             tournament_size, mutation_rate, num_nodes, num_tournaments):
    """
    Run the genetic algorithm on a given subpopulation for num_generations.
    If stagnation occurs, the subpopulation is regenerated while preserving the top 10 individuals.
    At the end, the best individual from the subpopulation is returned.
    """
    best_overall_fitness = 1e6
    stagnation_counter = 0
    for generation in range(num_generations):
        # Evaluate fitness for the subpopulation (lower is better)
        fitness_values = np.array([-calculate_fitness(route, distance_matrix) for route in subpop])
        current_best_fitness = np.min(fitness_values)
        if current_best_fitness < best_overall_fitness:
            best_overall_fitness = current_best_fitness
            stagnation_counter = 0
        else:
            stagnation_counter += 1

        # Regenerate subpopulation if stagnation is reached, preserving the top 10 individuals.
        if stagnation_counter >= stagnation_limit:
            best_indices = np.argsort(fitness_values)[:10]
            best_individuals = [subpop[i] for i in best_indices]
            new_subpop = generate_unique_population(len(subpop) - 10, num_nodes)
            subpop = new_subpop + best_individuals
            stagnation_counter = 0
            continue

        # Tournament Selection on this subpopulation.
        selected = select_in_tournament(subpop, fitness_values,
                                        number_tournaments=num_tournaments,
                                        tournament_size=tournament_size)

        # Crossover: pair the selected individuals (if odd, the last is skipped).
        offspring = []
        for i in range(0, len(selected) - 1, 2):
            parent1, parent2 = selected[i], selected[i + 1]
            # Exclude the fixed starting node for crossover.
            child_route = order_crossover(parent1[1:], parent2[1:])
            offspring.append([0] + child_route)

        # Mutate the offspring.
        mutated_offspring = [mutate(route, mutation_rate) for route in offspring]

        # Replacement: Replace the worst individuals with the mutated offspring.
        indices_to_replace = np.argsort(fitness_values)[::-1][:len(mutated_offspring)]
        for j, idx in enumerate(indices_to_replace):
            subpop[idx] = mutated_offspring[j]

        # Ensure uniqueness within the subpopulation.
        unique_subpop = set(tuple(ind) for ind in subpop)
        while len(unique_subpop) < len(subpop):
            new_ind = [0] + list(np.random.permutation(np.arange(1, num_nodes)))
            unique_subpop.add(tuple(new_ind))
        subpop = [list(ind) for ind in unique_subpop]
        # print(f"Generation {generation}: Best calculate_fitness = {current_best_fitness}")

    # After all generations, return the best individual from this subpopulation.
    fitness_values = np.array([-calculate_fitness(route, distance_matrix) for route in subpop])
    best_idx = np.argmin(fitness_values)
    best_individual = subpop[best_idx]
    return best_individual

def select_in_tournament(population,
                         scores,
                         number_tournaments=4,
                         tournament_size=3):
    """
    Tournament selection for genetic algorithm.

    Parameters:
        - population (list): The current population of routes.
        - scores (np.array): The calculate_fitness scores corresponding to each individual in the population.
        - number_tournaments (int): The number of the tournamnents to run in the population.
        - tournament_size (int): The number of individual to compete in the tournaments.

    Returns:
        - list: A list of selected individuals for crossover.
    """
    selected = []
    # run the specified number of tournaments
    for _ in range(number_tournaments):
        # randomly select indices for the tournament participants
        idx = np.random.choice(len(population), tournament_size, replace=False)
        # find the index of the individual with the highest fitness score among the selected
        best_idx = idx[np.argmax(scores[idx])]
        # append the best individual to the selected list
        selected.append(population[best_idx])
    return selected


def order_crossover(parent1, parent2):
    """
    Order crossover (OX) for permutations.

    Parameters:
        - parent1 (list): The first parent route.
        - parent2 (list): The second parent route.

    Returns:
        - list: The offspring route generated by the crossover.
    """
    size = len(parent1)
    start, end = sorted(np.random.choice(range(size), 2, replace=False))
    offspring = [None] * size
    offspring[start:end + 1] = parent1[start:end + 1]
    fill_values = [x for x in parent2 if x not in offspring[start:end + 1]]
    idx = 0
    for i in range(size):
        if offspring[i] is None:
            offspring[i] = fill_values[idx]
            idx += 1
    return offspring


def mutate(route,
           mutation_rate = 0.1):
    """
    Mutation operator: swap two nodes in the route.

    Parameters:
        - route (list): The route to mutate.
        - mutation_rate (float): The chance to mutate an individual.
    Returns:
        - list: The mutated route.
    """
    if np.random.rand() < mutation_rate:
        i, j = np.random.choice(len(route), 2, replace=False)
        route[i], route[j] = route[j], route[i]
    return route

def generate_unique_population(population_size, num_nodes, past_routes = []):
    """
    Generate a unique population of individuals for a genetic algorithm.

    Each individual in the population represents a route in a graph, where the first node is fixed (0) and the 
    remaining nodes are a permutation of the other nodes in the graph. This function ensures that all individuals
    in the population are unique.

    Parameters:
        - population_size (int): The desired size of the population.
        - num_nodes (int): The number of nodes in the graph, including the starting node.

    Returns:
        - list of lists: A list of unique individuals, where each individual is represented as a list of node indices.
    """
    past_routes_set = set(tuple(route) for route in past_routes)
    population = set()
    while len(population) < population_size:
        individual = [0] + list(np.random.permutation(np.arange(1, num_nodes)))
        tup = tuple(individual)
        # If the individual is already in past_routes, skip it.
        if tup in past_routes_set:
            continue
        population.add(tup)
    return [list(ind) for ind in population]
