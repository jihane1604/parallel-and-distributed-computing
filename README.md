# Lab 6: Distributed Computing using mpi4py

## Objectives
- Develop distributed computing programs that leverage multiple machines.
- Use mpi4py for parallel processing in Python.
- Implement two exercises: a square computation program and a virus spread simulation.

## Tasks
1. **Square Computation Program:**
   - Created a function `square` to compute squares of integers from 1 to n in parallel using mpi4py.
   - The root process gathers partial results, prints the total size of the results array, the last computed square, and the execution time.
   - Modified the program to compute squares up to 1e8.
   - **Bonus:** Determine the highest square computed within a 300-second time limit.
2. **Virus Spread Simulation:**
   - Initialized the MPI environment and set up communication among processes.
   - Defined simulation parameters (population size, spread chance, and random vaccination rate per process).
   - Implemented a `spread_virus` function to simulate infection dynamics.
   - Simulated the virus spread over multiple time steps with inter-process data exchange.
   - Calculated and printed the infection rate for each process.
3. **Performance Analysis:**
   - Measured execution times and analyzed scalability.
   - Observed the effects of distributing tasks across multiple machines.

## Results
- **Square Computation:**
  - Final array size: 
  - Last square computed: 
  - Execution Time: 
  - **Bonus Result:** 
- **Virus Spread Simulation:**
  - Infection Rates per Process: 
