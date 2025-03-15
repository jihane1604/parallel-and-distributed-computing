from src.sequential import run_seq
from src.processes import run_loop_processes, run_pool_map, run_pool_map_async, run_pool_apply, run_pool_apply_async, run_executor

seq_time = run_seq(7)
# loop_process_time = run_loop_processes(6)
apply_time = run_pool_apply(7)
apply_async_time = run_pool_apply_async(7)
map_time = run_pool_map(7)
map_async_time = run_pool_map_async(7)
executor_time = run_executor(7)


print(f"Sequential execution time: {seq_time}\n")
# print(f"Process looping execution time: {loop_process_time}\n")
print(f"Pool apply execution time: {apply_time}\n")
print(f"Pool apply async execution time: {apply_async_time}\n")
print(f"Pool map execution time: {map_time}\n")
print(f"Pool map async execution time: {map_async_time}\n")
print(f"Process pool executor execution time: {executor_time}\n")