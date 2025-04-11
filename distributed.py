"""
Main entry point for the distributed maze runner game using MPI.
"""

import argparse
import time
from mpi4py import MPI
from src.explorer import Explorer
from src.maze import create_maze
from src.game import run_game

def main():
    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

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
        # Disable visualization for all ranks except 0
        visualize = args.visualize if rank == 0 else False

        # Each rank runs its own instance of the explorer
        explorer = Explorer(maze, visualize=visualize)
        start_time = time.time()
        time_taken, moves = explorer.solve_right_hand()
        end_time = time.time()

        # Gather results at rank 0
        results = comm.gather((rank, time_taken, len(moves)), root=0)

        if rank == 0:
            print("\n=== MPI Maze Explorer Results ===")
            for r, t, m in results:
                print(f"Rank {r}: Solved in {t:.2f} seconds, Moves = {m}")
            print("Note: Width and height arguments were ignored for the static maze" if args.type == "static" else "")
            print("==================================\n")
    else:
        if rank == 0:
            # Only rank 0 can run the interactive game
            run_game(maze_type=args.type, width=args.width, height=args.height)
        else:
            print(f"Rank {rank}: Interactive mode is disabled in non-root processes.")

if __name__ == "__main__":
    main()
