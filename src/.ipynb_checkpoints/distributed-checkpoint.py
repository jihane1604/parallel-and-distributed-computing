from mpi4py import MPI
import numpy as np
import time

def calculate_fitness(route, distance_matrix):
    """
    Calculate the fitness value for a given route based on the total distance traveled.
    If any segment of the route is infeasible (distance equals 10000), return a large negative penalty.
    """
    total_distance = 0
    for i in range(len(route) - 1):
        node1 = route[i]
        node2 = route[i + 1]
        distance = distance_matrix[node1, node2]
        if distance == 10000:
            return -1e6  # Infeasible route penalty
        total_distance += distance
    return -total_distance

def select_in_tournament(population, scores, number_tournaments=4, tournament_size=3):
    """
    Perform tournament selection: for each tournament, randomly choose a subset of individuals and 
    select the one with the highest fitness score.
    """
    selected = []
    for _ in range(number_tournaments):
        idx = np.random.choice(len(population), tournament_size, replace=False)
        best_idx = idx[np.argmax(np.array(scores)[idx])]
        selected.append(population[best_idx])
    return selected

def parallel_fitness_evaluation(population, distance_matrix):
    """
    Distribute the population among MPI processes and compute fitness values in parallel.
    Then gather the results on the root process.
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    n = len(population)
    # Determine chunk sizes for even distribution.
    chunk_size = n // size
    remainder = n % size
    start = rank * chunk_size + min(rank, remainder)
    end = start + chunk_size + (1 if rank < remainder else 0)
    
    local_population = population[start:end]
    local_scores = [calculate_fitness(route, distance_matrix) for route in local_population]
    
    # Gather all local scores to the root process.
    all_scores = comm.gather(local_scores, root=0)
    
    if rank == 0:
        # Flatten the gathered list to maintain the original population order.
        scores = []
        for proc in range(size):
            scores.extend(all_scores[proc])
        return scores
    else:
        return None

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    # Example data: population of routes and a distance matrix.
    # In a real problem, these would be much larger.
    population = [
        [0, 1, 2, 3, 0],
        [0, 2, 1, 3, 0],
        [0, 3, 1, 2, 0],
        [0, 1, 3, 2, 0],
    ]
    distance_matrix = np.array([
        [0, 10, 15, 20],
        [10, 0, 35, 25],
        [15, 35, 0, 30],
        [20, 25, 30, 0]
    ])
    
    # Ensure all processes are synchronized before timing starts.
    comm.Barrier()
    start_time = time.time()
    
    # Parallel fitness evaluation
    scores = parallel_fitness_evaluation(population, distance_matrix)
    
    # Synchronize again after parallel work
    comm.Barrier()
    end_time = time.time()
    parallel_time = end_time - start_time

    if rank == 0:
        # For demonstration, compute the sequential fitness evaluation time.
        seq_start = time.time()
        seq_scores = [calculate_fitness(route, distance_matrix) for route in population]
        seq_end = time.time()
        sequential_time = seq_end - seq_start
        
        # Compute performance metrics: speedup and efficiency.
        speedup = sequential_time / parallel_time if parallel_time > 0 else float('inf')
        efficiency = speedup / comm.Get_size()
        
        # Perform tournament selection (done sequentially here for clarity).
        selected = select_in_tournament(population, scores, number_tournaments=4, tournament_size=3)
        
        print("Performance Metrics:")
        print("--------------------")
        print("Parallel Time:", parallel_time)
        print("Sequential Time:", sequential_time)
        print("Speedup:", speedup)
        print("Efficiency:", efficiency)
        print("\nFitness Scores:", scores)
        print("Selected Individuals for Crossover:", selected)

if __name__ == "__main__":
    main()
