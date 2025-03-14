# Lab 3 Part 2: Parallelizing Machine Learning Parameter Search

## Objectives
- Enhance a machine learning model training process by parallelizing parameter search.
- Implement parallelism using both threading and multiprocessing.
- Analyze and compare the performance improvements achieved.

## Tasks
1. **Data Preparation and Setup:**
   - Downloaded and extracted the housing prices dataset.
   - Copied the required files (data and Jupyter notebook) to the repository.
2. **Sequential Parameter Search:**
   - Ran the machine learning program in a sequential mode to seek the best parameters.
   - Measured the execution time.
3. **Parallel Implementation:**
   - Modified the program to perform parameter search in parallel using the threading module.
   - Modified the program to perform parameter search in parallel using the multiprocessing module.
   - Measured the execution time for both approaches.
4. **Performance Analysis:**
   - Compared the execution times and calculated performance metrics.
5. **Discussion:**
   - Discussed the trade-offs between threading and multiprocessing for this ML task.
   - Reflected on the challenges and benefits encountered.

## Results
- **Sequential Execution Time:** 63.55 seconds
- **Threaded Execution Time:** 25.69 seconds
- **Threading Speedup:** 2.47
- **Threading Efficiency:** 0.41
- **Multiprocessing Execution Time:** 19.52 seconds
- **Multiprocessing Speedup:** 3.25
- **Multiprocessing Efficency:**  0.54

## Conclusion

