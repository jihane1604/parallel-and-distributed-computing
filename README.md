# Lab 3 Part 1: Data Parallel Model

## Objectives
- Develop a data-parallel model using Python.
- Implement parallel summation using both threading and multiprocessing.
- Compare the execution times and compute performance metrics.

## Tasks
1. **Sequential Summation:**
   - Wrote a Python program to compute the sum of numbers from 1 to a large number.
   - Measured execution time and recorded the sum.
2. **Threading Parallelization:**
   - Divided the summation task into equal parts and computed each partial sum in a separate thread.
   - Measured the execution time and combined partial results.
3. **Multiprocessing Parallelization:**
   - Divided the summation task into equal parts for separate processes.
   - Measured the execution time and aggregated the results.
4. **Performance Analysis:**
   - Calculated speedup, efficiency, and applied Amdahl’s and Gustaffson’s Laws.
5. **Discussion:**
   - Discussed performance differences and challenges encountered when parallelizing the summation.

## Results for summing n = 1000000000
- **Sequential Sum and Time:** 500000000500000000, 33.11 seconds
- **Threaded Sum and Time:** 500000000500000000, 32.14 seconds
- **Threading Speedup:** 1.03
- **Threading Efficiency:** 0.25
- **Multiprocessing Sum and Time:** 500000000500000000, 8.32 seconds
- **Multiprocessing Speedup:** 3.97
- **Multiprocessing Efficiency:** 0.99

## Conclusion
Using multithreading doesn't necessairily help with speedup time but more with dividing tasks amongst threads. For time efficency, its better to use multiprocessing.
