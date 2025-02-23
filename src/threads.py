import threading
from src.utils import simulate_sensor, process_temperatures, update_display
import time

def run_threads():
    """
    Starts and manages multiple threads for sensor simulation, temperature processing, and display updates.
    The function initializes three sensor threads, a temperature processing thread, and a display update thread.
    It keeps the main thread alive to ensure continuous execution of the background threads.
    """
    
    # Start sensor simulation threads
    sensor_threads = []
    for i in range(3):
        thread = threading.Thread(target=simulate_sensor, args=(i,), daemon=True)
        sensor_threads.append(thread)
        thread.start()

    # Start temperature processing thread
    processing_thread = threading.Thread(target=process_temperatures, daemon=True)
    processing_thread.start()

    # Start the display update thread
    display_thread = threading.Thread(target=update_display, daemon=True)
    display_thread.start()

    # Keep the main thread alive to let other threads run
    while True:
        time.sleep(1)
