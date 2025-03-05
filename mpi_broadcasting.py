from mpi4py import MPI
import socket
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
if rank == 0:
    # Process 0 sends a non-blocking message to Process 1
    data_to_send = "Hello from Process 0"
else:
    data_to_send = None

data = comm.bcast(data_to_send, root = 0)
# Process 0 can perform other work here while the send operation completes
print("Process 0 sent data")

if rank != 0:
    # Process 1 sets up a non-blocking receive from Process 0
    
    hostname = socket.gethostname()
    ip = socket.gethostbyname(hostname)
    print(f"Process {rank} received data on machine {ip}\nthe data received:", data)
    # data_received = request.wait()  # Wait for the non-blocking receive to complete