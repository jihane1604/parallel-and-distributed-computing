# Assignment 1 Part 2

## Genetic Algorithm for TSP: Quick Overview

This program implements a genetic algorithm (GA) to solve a routing problem (Traveling Salesman Problem) by evolving a population of candidate routes.

## Explanation of the `genetic_algorithm_trial` code:

- **Data Loading:**
  - Reads a distance matrix from a CSV file (`city_distances.csv`) where each entry represents the distance between nodes, then turns it into a numpy array.  
- **Population Initialization:**
  - Sets parameters: number of nodes, population size, number of tournaments, mutation rate, number of generations, stagnation limit and a large infeasability penealty.
  - Generates an initial population of 10,000 unique routes starting from node 0.
- **Main GA Loop (for each generation):**
  - Loop through the number of generations and do the following for each generation
  - **Fitness Calculation:** Calculate the fitness values for the each individual node in the population.
  - **Stagnation Check:** Pick the best fitness of the current population by using `np.min()` (and taking the negative of the `calculate_fitness` function), then compare it to the overall best fitness. Reset the stagnation counter if the current fitness is better, otherwise increment it. If no improvement is seen for 5 consecutive generations, regenerate the population (keeping the best individual).
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
- **Fitness Evaluation:** Each individual's fitness is computed by summing the distances along its route. Since the fitness of one route is independent of another, the population can be split among multiple processes. Each process computes fitness for its subset of routes in parallel, which reduces the overall computation time when the population size is large.
- **Genetic Operators:** Atfer chunking the population into sevral (100) sub populations. For each one, the tournament selection, crossover and mutation is performed then collected after each generation.
- **Multiprocessing pools `multiprocessing_ver.py` file:** As a first trial, `multiprocessing.pool()` was used with `starmap_async` in order to parallelize the task across multiple processors on a single machine.
- This resulted in a much slower execution time of *__90.12 seconds__* compared to the sequential time of *__23.50 seconds__* when running it for 300 generations. We can assume that this is due to the process management overhead (just like in part one of the assignment).
- I tried with the default number of processes, with 12 processes, with starmap, with map, with map_async, with apply, with apply_async and with ProcessPoolExecutor all with diffrent amount of workers/processes but each time it was slowert than the sequential version :( 
- Speedup: 0.26
- Efficiency: 0.04
- **Island genetic algorithm `island_mode.py` file:** The second approach was to perform the GA loop for each sub population and let it undergo the evolutionary process, the best solution from each sub population is collected and compared, with the overall best solution selected as the final outcome. This strategy, is called an island genetic algorithm; however, it doesnt guarantee to achieve the absolute optimal solution.
- This resulted in a much faster execution time of *__5.78 seconds__* compared to the sequential time of *__14.78 seconds__* when running for 300 generation.
- Speedup: 2.66
- Efficiency: 0.44
- When running this version for 30,000 generations, the best solution was 1050 wtih an execution time of *__391 seconds__*

## Distributing using MPI4PY
- **`distributed folder`:** The distribution of the program was done using the island mode approach, and was run on 3 machines. This folder contains the files that were copied over to the other machines.
- **Data Distribution:**
    - The master (rank 0) loads the distance matrix and sets GA parameters, other ranks initiate them to None.
    - The initial population is generated and split into as many chunks as there are MPI processes (using scatter).
- **Broadcasting:**
    - The parameters are broadcast to all the other machines using `comm.bcast()`.
- **Scattering:**
    - Each process receives one subpopulation using `comm.scatter()`.
- **Local Evolution:**
    - Each process runs the GA on its subpopulation using `run_ga_on_subpopulation()`, which runs for all generations.
- **Gathering and Final Selection:**
    - The best solution from each process is gathered using `comm.gather()`, and the master chooses the overall best solution based on fitness. This doesnt guarantee to have the best optimal solution as it changes with every run.
    - Best distance: 1041
    - Execution time: 15.21
- **Bash file:** The folder also contains a bash file to run, which copies the folder to the other machines and runs the python file.

## Running on AWS
- I created 3 instances on aws following the tutorial on d2l. Then I created a script `install_anaconda.sh` which installs anaconda and another script `mpi_installation.sh` to install  mpi4py. I initially had everything in one script but I was running into issues while creating the environment so i had to do that manually and separate each step into a different bash script. I copied these files to the other machines using `scp` and ran them on each. I clone the GitHub repo only in the first machine (host) and pulled the assignment branch, the copied the `distributed_folder` to my root folder, changed the machines.txt ips and ran it on all my machines. The bash scripts can be found in the aws folder.
- HOWEVER, I ran into some issues AGAIN this time when running the mpi4py across the machines i kept getting the error `Host key verification failed`
- ![Host key verification error.](/image/error1.png "Error.")
- I tried fixing it by running `ssh-keyscan -H public_ip >> ~/.ssh/known_hosts` in each machine with the public ip of all the hosts, but I still got the same error so I tried using the private ip address instead and I got disconnected from all the machines
- ![Disconnected error.](/image/error2.png "Error.")
- And this is what was displayed on aws idk what I did wrong but pls appreciate the effort :(
- ![Error on aws.](/image/error3.png "Error.")

## Improvements of the genetic algorithm
- After further investigating the genetic algorithm, some of the parameters were increased
- **Population size:** A larger population increases genetic diversity, which can improve the chance of finding a good solution. However it increases the computational time. (`population_size = 15000`)
- **Number of tournaments:** More tournaments can lead to a more robust selection process, ensuring that a diverse set of high-quality individuals are chosen. If set too low, the selection might not effectively differentiate among individuals, while too many may increase the computational overhead without benefits. (`num_tournaments = 10`)
- **Tournament size:** A larger tournament size increases selection pressure because it raises the probability that the best individuals (with the highest fitness) are selected. However, excessive selection pressure can reduce diversity, leading to premature convergence on suboptimal solutions. (`tournament_size = 4`)
- **Number of generations:** More generations provide the algorithm with more opportunities to evolve and improve the population, potentially yielding better solutions over time. However it also increases the runtime of the program. (`num_generations = 3000`)
- **Stagnation:** The stagnation rate was also changed by keeping the top 10 best individuals instead of only the top 1.
- The best distance was 1050
- Execution time = *__237.48 seconds__*
- **Possible approach**: Another way to improve the algorithm is to keep track of the previously generated nodes using a list or a set, and when generating a new population, ensure that the individuals generated do not already exist. This ensures the absolute uniqueness of each individual in the population throughout the generations; however, when i tried implementing it, it was introducing a lot of overhead without necessarily arriving to a better minimum path (ran for 3000 generations and arrived to the same minimum path) so i decided to not keep this solution.

- **Performance metrics:**
    - Sequential time: 236.59 seconds
    - Parallel time: 34.57 seconds
    - Speedup: 6.84
    - Efficiency: 1.14
- We can conclude that the enhancements improved the efficiency of the paralllization probably due to the fact that the sequential version is slower due to the enhancements (more tournaments and higher population)
 
## Extended city dataset
- The `city_distances_extended.csv` was ran using both the parallel and the sequential versions of the algorithm for 30000 generations with the enhancements but no path was found (all paths were 1000000 indicating the absence of a feasable path).
- I am not sure how to fix this.

## Adding more cars to the problem
- By adding more cars, the problem becomes what is called a Vehicle Routing Problem (an optimization problem). An approach discussed by Ochelska-Mierzejewska et al. (2021) and Prins (2003) will be explained here (yes i read a whole research paper for this pls give me a bonus) which consists of:
- **Chromosome Representation:** A VRP solution is represented as a string of integers. *0* represents the depot, numbers from *1* to *n* (number of deliveries) represent delivery points, and numbers from *n+1* to *m-1* may represent vehicles. A solution is constructed modularly, with blocks for each vehicle containing the vehicle's identifier (optional for the first vehicle), followed by *0*, the sequence of customers visited, and another *0* to denote the return to the depot.
- **Initial Population:** An initial set of potential solutions (a population of chromosomes) is created
- **Fitness Evaluation:** Using an auxiliary structure (splitting procedure) whichy uses dynamic programming that “splits” the chromosome into feasible trips. For every possible split (every subsequence of clients), the procedure checks if the total demand is within the vehicle’s capacity and computes the corresponding trip cost. Then it determines the minimal total cost to serve all clients. 
- **Genetic Operators:** Generate unique populations, select parents via tournament selection, combine them with order crossover, and introduce variability through mutation.
- **Population Management:** The algorithm maintains a population of solutions (chromosomes). It includes mechanisms to prevent stagnation and to ensure solution uniqueness.
- **Overall Process:** The GA evolves the population over many generations. At each generation, the fitness of every chromosome is evaluated using the splitting procedure. Selection, crossover, and mutation then produce new candidate solutions. Finally, the best solution is reported.
- References
    - Prins, C. (2003). A simple and effective evolutionary algorithm for the vehicle routing problem. Computers & Operations Research, 31(12), 1985–2002. https://doi.org/10.1016/s0305-0548(03)00158-8
    - Ochelska-Mierzejewska, J., Poniszewska-Marańda, A., & Marańda, W. (2021). Selected genetic algorithms for vehicle routing problem solving. Electronics, 10(24), 3147. https://doi.org/10.3390/electronics10243147
- **Implementation in `vrp.py` file:**
    - **calculate_fitness_vrp:** Splits a given chromosome (a permutation of client nodes) into a sequence of feasible trips. It uses dynamic programming to compute the minimum total cost (including depot-to-client and client-to-depot distances) while ensuring each trip satisfies the vehicle capacity.
    - **generate_unique_population_vrp:** Generates an initial population of unique chromosomes (random permutations of clients). It ensures that no duplicate solutions are produced and can avoid routes from past generations if provided.
    - **run_vrp_ga:** The main function that runs the genetic algorithm for the VRP. It loads data, initializes parameters and the population, and then iteratively performs fitness evaluation, selection, crossover, mutation, and replacement (with stagnation checks) to evolve solutions. Finally, it outputs the best solution found. This was ran for 300 generation with the same parameters as the enhanced version of the original GA file.
    - **Execution time:** 793.94 seconds
    - **Best path:** 1056