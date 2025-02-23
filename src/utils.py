import random
import time
from queue import Queue
import threading
import os

# Global dictionaries
latest_temperatures = {"Sensor 0": None, "Sensor 1": None, "Sensor 2": None}
temperature_averages = {"Sensor 0": None, "Sensor 1": None, "Sensor 2": None}
temperature_queues = {"Sensor 0": Queue(), "Sensor 1": Queue(), "Sensor 2": Queue()}

# Synchronization Lock 
lock = threading.RLock()

def simulate_sensor(sensor_id):
    """
    Simulates a temperature sensor that generates random temperature readings between 15 and 40 degrees Celsius.
    Updates the latest temperature for the corresponding sensor and stores values in a queue for further processing.
    
    Args:
        sensor_id (int): The identifier for the sensor (0, 1, or 2).
    """
    
    # gloabl variable 
    global latest_temperatures
    
    while True:
        # generate random temperature
        temperature = random.randint(15, 40)

        # update the latest temperature and temperature queues
        with lock:
            latest_temperatures[f"Sensor {sensor_id}"] = temperature
            
        with lock:
            temperature_queues[f"Sensor {sensor_id}"].put(temperature)
        
        time.sleep(1)

def process_temperatures():
    """
    Continuously calculates and updates the average temperature for each sensor.
    Retrieves all values from the respective sensor queue and computes the average.
    """
    
    # gloabl variable 
    global temperature_averages
    
    while True:
        with lock:
            # Calculate the average temperature for each sensor
            for sensor_id in range(3):
                sensor_key = f"Sensor {sensor_id}"
                
                if not temperature_queues[sensor_key].empty():
                    # Get all items in the queue
                    temps = list(temperature_queues[sensor_key].queue)
                    
                    avg_temp = sum(temps) / len(temps)
                    temperature_averages[sensor_key] = avg_temp
        
        time.sleep(1)

def initialize_display():
    """
    Initializes the display with placeholder values before actual sensor data is processed.
    Prints a formatted output showing the latest and average temperatures as "--°C".
    """
    
    print("Current temperatures:")
    print("Latest Temperatures: ", end="")
    for i in range(3):
        print(f"Sensor {i}: --°C", end=" ")
    print("\n")
    for i in range(3):
        print(f"Sensor {i} Average: --°C")

def update_display():
    """
    Periodically clears the terminal screen and displays the latest and average temperature readings for each sensor.
    Retrieves values from the global dictionaries and updates the display every second.
    """
    
    while True:
        # clear screen 
        os.system('clear')
        print("Current temperatures:")
        print("Latest Temperatures: ", end="")
        for i in range(3):
            print(f"Sensor {i}: {latest_temperatures[f'Sensor {i}']}°C", end=" ")
        print("\n")
        for i in range(3):
            print(f"Sensor {i} Average: {temperature_averages[f'Sensor {i}']: .2f}°C")
        
        time.sleep(5)
