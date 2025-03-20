# Assignment 1 Part 1

This repository contains a set of Python programs developed as part of the DSAI 3202 course on parallel and distributed computing. The assignment explores various multiprocessing techniques and process synchronization using semaphores.

---

## Overview

The assignment has two major parts:

1. **Square Program**  
   In this part, we compare different approaches for computing the square of numbers in a range (from 0 to 10⁶ or 10⁷) and measure their execution times. The approaches include:
   - **Sequential execution** using a simple for loop.
   - **Multiprocessing by spawning a process per task.**
   - **Multiprocessing Pool methods:**  
     - Synchronous `map()`
     - Asynchronous `map_async()`
     - Synchronous `apply()`
     - Asynchronous `apply_async()`
   - **ProcessPoolExecutor** from the `concurrent.futures` module.

   **Results n = 6:**
   - **Sequential execution time** 0.10 seconds
   - **Multiprocessing loop execution time** -- seconds -- an error occurs
   - **Apply execution time:** 168.69 seconds
   - **Apply_async execution time:** 170.04 seconds
   - **Map execution time:** 0.17 seconds
   - **Map_async execution time:** 0.18 seconds
   - **ProcessPoolExecutor execution time:** Before chunking: 111.28 seconds - After chunking: 0.31 seconds

   **Results n = 7:**
   - **Sequential execution time** 1.15 seconds
   - **Multiprocessing loop execution time** -- seconds -- an error occurs
   - **Apply execution time:** 1749.47 seconds
   - **Apply_async execution time:** 1760.59 seconds
   - **Map execution time:** 1.56 seconds
   - **Map_async execution time:** 1.55 seconds
   - **ProcessPoolExecutor execution time:** Before chunking: 1116.72 seconds - After chunking: 3.20 seconds

   **Observations (for n = 6 and n = 7):**
   - The **sequential execution** is extremely fast for trivial computations like squaring numbers.
   - The **apply/apply_async** approaches incur a high overhead, leading to much slower performance.
   - The **map/map_async** methods in the multiprocessing pool show a good performance because they eliminate process management overhead; however they don't make the execution time any faster than running it sequentially.
   - The **ProcessPoolExecutor** method shows a very slow execution due to overhead from processing each small task individually without chunking. After applying chunking, the excution time was dramatically reduced because chunking reduces the overhead involved in dispatching and managing tasks in the `ProcessPoolExecutor`

3. **Process Synchronization with Semaphores**  
   This part demonstrates how to manage access to a limited resource (simulated database connections) using a semaphore.
   - A **ConnectionPool** class is implemented, which uses:
     - A shared list (via `multiprocessing.Manager`) to hold connection identifiers.
     - A **semaphore** to restrict concurrent access to the available connections.
     - A **lock** to protect modifications to the shared list.
   - The `access_database` function simulates a process that:
     - Waits for a connection.
     - Acquires it and simulates a database operation (by sleeping for a random duration).
     - Releases the connection.
   - When more processes than available connections try to access the pool, the semaphore forces the extra processes to wait until a connection is released. This prevents race conditions and ensures safe access to the shared resource.

---

## Project Structure

- **`src/utils.py`**  
  Contains the `square()` function which computes the square of a given integer.

- **`src/sequential.py`**  
  Implements the `run_seq()` function that:
  - Generates a list of numbers (0 to 10^n - 1).
  - Computes their squares sequentially.
  - Measures and prints the execution time and the last computed result.

- **`src/processes.py`**  
  Contains multiple functions demonstrating different multiprocessing strategies:
  - `run_loop_processes()`: Spawns a separate process for each number.
  - `run_pool_map()` and `run_pool_map_async()`: Use a multiprocessing Pool with synchronous and asynchronous map.
  - `run_pool_apply()` and `run_pool_apply_async()`: Use a multiprocessing Pool with synchronous and asynchronous apply.
  - `run_executor()`: Uses `ProcessPoolExecutor` from the `concurrent.futures` module.
  
  Additionally, it includes a helper function `worker()` that computes the square and puts the result in a queue.

- **`main.py`**  
  Acts as the entry point for testing the different implementations. It imports functions from both `sequential.py` and `processes.py`, runs them with `n = 6`, and prints their respective execution times. Sample results (for n = 6 and n = 7) are provided in the assignment instructions.

- **`semaphore_testing.py`**  
  Implements the `ConnectionPool` class and the `access_database()` function to demonstrate process synchronization with semaphores. It shows how a limited pool of connections (e.g., 3 connections) is managed when more processes (e.g., 6) attempt to access the pool concurrently.

---

## Discussion and Conclusions

- **Square Computations:**
    - **Sequential vs. Multiprocessing:**
        For simple operations like squaring a number, the sequential approach is very efficient. Multiprocessing introduces overhead due to process creation and interprocess communication.
 
    - **Multiprocessing loop:**
        The error `OSError: [Errno 24] Too many open files` occurs because were creating a new process for every number, which opens many file descriptors simultaneously, quickly exceeding the operating system’s limit on open files.

    - **Multiprocessing Pool Efficiency:**
        The Pool’s `map()` and `map_async()` methods significantly reduce the overhead by reusing a fixed number of worker processes. On the other hand, `apply()` and `apply_async()` are much slower because they submit tasks one at a time.

    - **ProcessPoolExecutor**:
        When you call `executor.map()` without specifying a chunk size, it typically sends one task at a time to the worker processes. Since each task (computing the square) is very trivial, the accumulated overhead of task scheduling, serialization and deserialization and interprocess communicatoin takes up all the execution time.
        In contrast, the `multiprocessing.Pool.map()` and `map_async()` methods automatically calculate an efficient chunk size, grouping many numbers together in each task submission. This reduces the overhead by paying off the cost of interprocess communication over many tasks.
        When using chunking, instead of incurring the overhead for each individual task, the overhead is incurred once per batch, meaning that fewer, larger data transfers occur between processes, and fewer tasks overall.
        In conclusion, because the ProcessPoolExecutor processes each tiny task individually, its overhead becomes very significant relative to the fast execution of the `square()` function, leading to much longer execution times. Chunking transforms that issue by efficiently handling the computational work by each worker process.

- **Semaphore and Connection Pool:**

    - When more processes request connections than are available, the semaphore blocks the extra processes until a connection is freed. This ensures that only a limited number of processes (equal to the number of connections) access the resource concurrently.

    - The use of a shared list (managed via `multiprocessing.Manager`) and a lock guarantees that the connection pool is modified safely, avoiding race conditions.

    - This pattern is useful in real-world applications to manage scarce resources (like database connections) in a concurrent environment.
