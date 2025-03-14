# Lab 4: Temperature Monitoring System

## Objectives
- Simulate temperature readings from multiple sensors.
- Calculate and display average temperatures in real-time.
- Implement thread synchronization using locks and conditions.

## Tasks
1. **Sensor Simulation:**
   - Developed a `simulate_sensor` function to generate random temperature readings (using `random.randint(15, 40)`).
   - Updated a global dictionary (`latest_temperatures`) with sensor readings every second.
2. **Data Processing:**
   - Implemented `process_temperatures` to calculate the average temperature from values in a queue.
   - Updated a global dictionary (`temperature_averages`) with the calculated averages.
3. **Threading and Display:**
   - Created threads to run sensor simulation and temperature processing concurrently.
   - Implemented an `initialize_display` function to set up the console output.
   - Developed `update_display` to refresh the displayed temperatures without clearing the console.
4. **Synchronization:**
   - Used RLock and Condition to manage data access and synchronization.
