import random
import time
import threading
import queue

# Global dictionaries
latest_temperatures = {}
temperature_averages = {}
temperature_queue = queue.Queue()

# Synchronization: Lock and Condition
lock = threading.RLock()
condition = threading.Condition(lock)

# Function to simulate the sensor (Part 3.a)
def simulate_sensor():
    while True:
        temp = random.randint(15, 40)
        latest_temperatures["latest"] = temp
        temperature_queue.put(temp)
        time.sleep(1)

# Function to process temperature readings (Part 3.b)
def process_temperatures():
    while True:
        with lock:
            if not temperature_queue.empty():
                temps = list(temperature_queue.queue)  # Get all items in the queue
                avg_temp = sum(temps) / len(temps)
                temperature_averages["average"] = avg_temp
            time.sleep(1)

def initialize_display():
    print("Current temperatures:")
    print("Latest Temperatures: Sensor 1: --°C Sensor 2: --°C Sensor 3: --°C", flush = True)
    print("Sensor 1 Average: --°C", flush = True)
    print("Sensor 2 Average: --°C", flush = True)
    print("Sensor 3 Average: --°C", flush = True)

def update_display():
    while True:
        with lock:
            latest_temp = latest_temperatures.get("latest", "--")
            avg_temp = temperature_averages.get("average", "--")
            
        print(f"\rLatest Temperatures: Sensor 1: {latest_temp}°C Sensor 2: {latest_temp}°C Sensor 3: {latest_temp}°C", end="", flush = True)
        print(f"\rSensor 1 Average: {avg_temp}°C", end="", flush=True)
        print(f"\rSensor 2 Average: {avg_temp}°C", end="", flush=True)
        print(f"\rSensor 3 Average: {avg_temp}°C", end="", flush=True)
        time.sleep(5)  # Update every 5 seconds

# Threading to run both functions concurrently (Part 3.c)
if __name__ == "__main__":
    initialize_display()
    # Creating threads
    sensor_thread = threading.Thread(target=simulate_sensor, daemon=True)
    processor_thread = threading.Thread(target=process_temperatures, daemon=True)
    display_thread = threading.Thread(target=update_display, daemon=True)

    # Start the threads
    sensor_thread.start()
    processor_thread.start()
    display_thread.start()

    # Main thread waits here, but the others continue running in the background
    while True:
        time.sleep(1)  # Main program can continue doing other work
