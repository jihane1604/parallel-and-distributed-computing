# Lab 2: First Parallel Programs

## Objectives
- Build the first parallel programs using Python.
- Implement parallelism with threads and processes.
- Measure and compare performance against a sequential implementation.
- Compute performance metrics such as speedup, efficiency, and theoretical speedups using Amdahl’s and Gustaffson’s Laws.

## Tasks
1. **Sequential Implementation:**
   - Created a function to randomly generate and join 1000 characters.
   - Created a function to randomly generate and sum 1000 numbers.
   - Timed the execution of both functions.
2. **Threading Implementation:**
   - Created threads for each function and measured the execution time.
   - (Advanced) Created two threads per function.
3. **Multiprocessing Implementation:**
   - Created processes for each function and measured the execution time.
   - (Advanced) Created two processes per function.
4. **Performance Analysis:**
   - Calculated speedup and efficiency.
   - Applied Amdahl’s and Gustaffson’s Laws to estimate expected speedups.
5. **Discussion:**
   - Compared the performance of sequential, threaded, and multiprocessed approaches.

## Results
- **Sequential Execution Time:** 5.37 seconds
- **Threaded Execution Time:** 5.43 seconds
- **Threading Speedup:** 0.98
- **Threading Efficiency:** 0.24
- **Multiprocessing Execution Time:** 1.68
- **Multiprocessing Speeup:** 3.18
- **Multiprocessing Efficiency:** 0.79

## Conclusion
Using multithreading doesn't necessairily help with speedup time but more with dividing tasks amongst threads. For time efficency, its better to use multiprocessing.
