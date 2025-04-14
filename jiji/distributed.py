"""
Main entry point for the distributed maze runner game using MPI.
"""

import argparse
import time
from mpi4py import MPI
from src.explorer import Explorer
from src.maze import create_maze
from src.game import run_game
from src.final_visualization import visualize_path

def main():
    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    #workers = {0: Explorer.solve_flood_fill, 1: Explorer.solve_right_hand}

    parser = argparse.ArgumentParser(description="Distributed Maze Runner Game with MPI")
    parser.add_argument("--type", choices=["random", "static"], default="random",
                        help="Type of maze to generate (random or static)")
    parser.add_argument("--width", type=int, default=30,
                        help="Width of the maze (default: 30, ignored for static mazes)")
    parser.add_argument("--height", type=int, default=30,
                        help="Height of the maze (default: 30, ignored for static mazes)")
    parser.add_argument("--auto", action="store_true",
                        help="Run automated maze exploration")
    parser.add_argument("--visualize", action="store_true",
                        help="Visualize the automated exploration in real-time (only rank 0)")
    
    args = parser.parse_args()

    # Rank 0 generates and broadcasts the maze
    if rank == 0:
        maze = create_maze(args.width, args.height, args.type)
    else:
        maze = None

    # Broadcast maze to all processes
    maze = comm.bcast(maze, root=0)

    if args.auto:
        # only rank 0 will have visualization if asked
        visualize = args.visualize if rank == 0 else False

        # each rank runs its own instance of the explorer
        explorer = Explorer(maze, visualize=visualize)
        start_time = time.time()
        # define a different function for each worker
        worker_strategies = {
            0: explorer.solve_right_hand,
            1: explorer.solve_flood_fill,
            2: explorer.solve_astar,
            #3: explorer.solve_aco,
            #4: explorer.solve_ga,
            #5: explorer.solve_pso
        }
        # map each function to its name
        function_names = {
            explorer.solve_right_hand: "Right Hand",
            explorer.solve_flood_fill: "Flood Fill",
            explorer.solve_astar: "A Star",
            #explorer.solve_aco: "Ant Colony"
        }

        # default to right hand if the rank is not in the dictionary
        strategy = worker_strategies.get(rank, explorer.solve_right_hand)
        
        time_taken, moves, backtracks = strategy()
        end_time = time.time()

        # visualize the path taken
        visualize_path(moves, function_names[strategy])

        # gather results at rank 0
        results = comm.gather((rank, time_taken, len(moves), moves), root=0)

        if rank == 0:
            print("\n=== MPI Maze Explorer Results ===")
            best = {"worker": None, "moves": float('inf'), "path": None}
            for r, t, m, p in results:
                print(f"Worker {r}: Time {t:.2f} seconds || Moves: {m} || Backtracks: {backtracks}")
                if m < best["moves"]:
                    best["worker"] = r
                    best["moves"] = m
                    best["path"] = p
            print("Note: Width and height arguments were ignored for the static maze" if args.type == "static" else "")
            print
            print("==================================\n")
            print(f"The best performing worker is: {best['worker']} using the {function_names[worker_strategies[best['worker']]]}, solved in: {best['moves']} moves \n")
            
    else:
        if rank == 0:
            # Only rank 0 can run the interactive game
            run_game(maze_type=args.type, width=args.width, height=args.height)
        else:
            print(f"Rank {rank}: Interactive mode is disabled in non-root processes.")

if __name__ == "__main__":
    main()
