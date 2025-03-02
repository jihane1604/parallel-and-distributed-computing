import glob
import cv2
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
import time
from tqdm import tqdm
import os
from src.utils import open_images
from src.utils import filter_single_image

# function to filter images
def filter_images_process(images):
    """
    """
    processed_images = []

    with ProcessPoolExecutor(max_workers = 6) as executor:
        results = list(tqdm(executor.map(process_single_image, images, chunksize = 5), total=len(images)))
    return results

# function to process images 
def process_images_parallel(images_list, tumor_presence):
    with Pool(processes = 6) as pool:
        glcm_features_list = pool.map_async(process_single_image, [(filtered_image, tumor_presence) for filtered_image in images_list])
        result = glcm_features_list.get()
    return result
    
# main program
def main:
    # open the yes and no images
    yes_images, no_images = open_images()
    
    # run the processes
    start_time = time.time()
    yes_inputs = process_images_process(yes_images)
    no_inputs = process_images_process(no_images)
    end_time = time.time()
    
    process_filtering_time = end_time - start_time
    print(f"Processing execution time: {process_filtering_time} seconds")
    
    # Save execution time to a file
    script_directory = os.path.dirname(os.path.abspath(__file__))
    execution_time_path = os.path.join(script_directory, "processing_execution_time.txt")
    
    with open(execution_time_path, "w") as file:
        file.write(f"Execution Time: {process_filtering_time} seconds\n")
    print("saved filtering excecution time")

    # Process the 'yes' and 'no' image lists
    start_time = time.time()
    yes_glcm_features = process_images_parallel(yes_inputs, 1)
    no_glcm_features = process_images_parallel(no_inputs, 0)
    end_time = time.t()
    
    # Combine the features into a single list
    all_glcm_features = yes_glcm_features + no_glcm_features
    
    # Convert the list of dictionaries to a pandas DataFrame
    dataframe = pd.DataFrame(all_glcm_features)
    
    # Print the first few rows of the DataFrame
    print(dataframe.shape)
    
    # Shuffle the DataFrame
    shuffled_dataframe = dataframe.sample(frac=1).reset_index(drop=True)
    
    # Print the first few rows of the shuffled DataFrame
    print(shuffled_dataframe.head())

main()