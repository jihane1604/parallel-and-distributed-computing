from src.multiprocessing_results import process_time, process_glcm_time
from src.sequential_results import seq_time, seq_glcm_time

total_seq_time = seq_time + seq_glcm_time
total_process_time = process_time + process_glcm_time

process_speedup = total_seq_time / total_process_time
process_efficiency = process_speedup / 6 # num processors = 6
process_amdhal = 1 / ((1-0.2)+(0.2/6))
process_gustafson = 6 + 0.8 *(1-6)

print(f"Multiprocessing speedup: {process_speedup} \n\tEfficiency: {process_efficiency} \n\tAmdhals: {process_amdhal} \n\tGustafson: {process_gustafson}")
