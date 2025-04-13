from mpi4py import MPI
import numpy as np
import pandas as pd
import time
from genetic_algorithms_functions import (
    calculate_fitness,
    run_ga_on_subpopulation,
    generate_unique_population
)

def run_mpi():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Master process loads data and sets parameters
    if rank == 0:
        start_time = time.time()
        # Load the distance matrix (CSV file must include depot as row/column 0)
        distance_matrix = pd.read_csv('data/city_distances.csv').to_numpy()
        num_nodes = distance_matrix.shape[0]
        
        # GA parameters
        population_size = 15000
        tournament_size = 4         # individuals per tournament
        mutation_rate = 0.1
        num_generations = 3000
        stagnation_limit = 5
        num_tournaments = 10        # tournaments per subpopulation

        # Generate initial population and split it among processes
        np.random.seed(42)  # For reproducibility
        initial_population = generate_unique_population(population_size, num_nodes)
        # Split into 'size' subpopulations
        subpopulations = np.array_split(initial_population, size)
    else:
        # For non-master processes, initialize variables to None
        distance_matrix = None
        num_nodes = None
        tournament_size = None
        mutation_rate = None
        num_generations = None
        stagnation_limit = None
        num_tournaments = None
        subpopulations = None
        start_time = None

    # Broadcast parameters to all processes
    distance_matrix = comm.bcast(distance_matrix, root=0)
    num_nodes = comm.bcast(num_nodes, root=0)
    tournament_size = comm.bcast(tournament_size, root=0)
    mutation_rate = comm.bcast(mutation_rate, root=0)
    num_generations = comm.bcast(num_generations, root=0)
    stagnation_limit = comm.bcast(stagnation_limit, root=0)
    num_tournaments = comm.bcast(num_tournaments, root=0)

    # Scatter subpopulations to all processes.
    local_subpop = comm.scatter(subpopulations, root=0)

    # Each process runs GA on its assigned subpopulation.
    local_best = run_ga_on_subpopulation(
        local_subpop, 
        distance_matrix, 
        num_generations, 
        stagnation_limit, 
        tournament_size, 
        mutation_rate, 
        num_nodes, 
        num_tournaments
    )

    # Gather the best solution from each process to the master.
    all_best = comm.gather(local_best, root=0)

    if rank == 0:
        # Evaluate fitness for each best solution (lower is better, so we use negative cost).
        best_fitness_values = np.array([-calculate_fitness(sol, distance_matrix) for sol in all_best])
        overall_best_idx = np.argmin(best_fitness_values)
        overall_best = all_best[overall_best_idx]
        total_distance = -calculate_fitness(overall_best, distance_matrix)
        end_time = time.time()
        
        print("Best solutions from subpopulations:")
        for idx, sol in enumerate(all_best):
            print(f" Subpopulation {idx}: Distance = {-calculate_fitness(sol, distance_matrix)}")
        print("\nOverall Best Solution:", overall_best)
        print("Total Distance:", total_distance)
        print("Total time taken:", end_time - start_time)
        return end_time - start_time
    else:
        return None

if __name__ == "__main__":
    run_mpi()
