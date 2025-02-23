from src.utils import initialize_display
from src.threads import run_threads
import time

# initialize the display
initialize_display()

# run the threads
run_threads()

while True:
    time.sleep(1)