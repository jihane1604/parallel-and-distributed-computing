from genetic_algorithm_trial import run_seq
from island_mode import run_parallel

seq_time = run_seq()
parallel_time = run_parallel()

speedup = seq_time / parallel_time
efficiency = speedup / 6

print(f"Sequential time: {seq_time}")
print(f"Parallel time: {parallel_time}")

print(f"Speedup: {speedup}")
print(f"Efficiency: {efficiency}")