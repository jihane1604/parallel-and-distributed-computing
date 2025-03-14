import multiprocessing
from concurrent.futures import ProcessPoolExecutor
import time
from tqdm import tqdm
import os
from utils import open_images, filter_single_image_parallel, process_single_image, train_model
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
from sklearn.svm import SVC


# function to filter images
def filter_images_process(images):
    """
    Apply parallel image filtering to a list of images using multiple processes.
    
    This function utilizes a ProcessPoolExecutor to apply the
    'filter_single_image_parallel' function to each image in the input list.
    It displays a progress bar using tqdm and returns a list of filtered images.
    
    Args:
        images (list): A list of image objects (e.g., NumPy arrays) to be filtered.
    
    Returns:
        list: A list of dictionaries, each containing the filtered results for an image.
    """

    with ProcessPoolExecutor(max_workers = 6) as executor:
        results = list(tqdm(executor.map(filter_single_image_parallel, images), total=len(images)))
    return results

# function to process images 
def process_images_parallel(images_list, tumor_presence):
    """
    Compute GLCM features for a list of filtered images in parallel.
    
    This function uses a multiprocessing pool to apply the 'process_single_image'
    function to each filtered image along with a tumor presence label.
    It returns a list of dictionaries containing the extracted GLCM features.
    
    Args:
        images_list (list): A list of filtered image dictionaries.
        tumor_presence (int or bool): A label indicating the presence (e.g., 1) or absence (e.g., 0) of a tumor.
    
    Returns:
        list: A list of dictionaries with GLCM features extracted from each image.
    """
    with multiprocessing.Pool(processes = 6) as pool:
        glcm_features_list = pool.map_async(process_single_image, [(filtered_image, tumor_presence) for filtered_image in images_list])
        result = glcm_features_list.get()
    return result

def run_filtering():
    """
    Apply image filtering to both 'yes' (tumor) and 'no' (non-tumor) images.
    
    This function performs the following steps:
      - Opens images using the 'open_images' function.
      - Applies parallel filtering to the 'yes' and 'no' image sets using 'filter_images_process'.
      - Measures and prints the total execution time for filtering.
      - Saves the filtering execution time to a file named 'multiprocessing_results.py'.
    
    Returns:
        tuple: Two lists containing filtered image dictionaries for 'yes' and 'no' images, respectively.
    """
    # open the yes and no images
    yes_images, no_images = open_images()
    print("applying filters to images")
    # run the processes
    start_time = time.time()
    yes_inputs = filter_images_process(yes_images)
    no_inputs = filter_images_process(no_images)
    end_time = time.time()
    
    process_filtering_time = end_time - start_time
    print(f"Processing execution time: {process_filtering_time} seconds")
    
    # Save execution time to a file
    script_directory = os.path.dirname(os.path.abspath(__file__))
    execution_time_path = os.path.join(script_directory, "multiprocessing_results.py")
    
    with open(execution_time_path, "w") as file:
        file.write(f"process_time = {process_filtering_time}\n")
    print("saved filtering excecution time")

    return yes_inputs, no_inputs

def run_glcm():
    """
    Compute and process GLCM features for filtered images and return a shuffled DataFrame.
    
    This function:
      - Calls 'run_filtering' to obtain filtered 'yes' and 'no' images.
      - Processes these images to extract GLCM features using 'process_images_parallel'.
      - Measures and prints the execution time for GLCM processing.
      - Appends the GLCM processing time to 'multiprocessing_results.py'.
      - Combines the features into a single pandas DataFrame and shuffles it.
    
    Returns:
        pandas.DataFrame: A shuffled DataFrame containing GLCM features and tumor labels.
    """
    yes_inputs, no_inputs = run_filtering()
    print("processing glcm")
    # Process the 'yes' and 'no' image lists
    start_time = time.time()
    yes_glcm_features = process_images_parallel(yes_inputs, 1)
    no_glcm_features = process_images_parallel(no_inputs, 0)
    end_time = time.time()
    
    process_glcm_time = end_time - start_time
    print(f"GLCM processing execution time: {process_glcm_time} seconds")
    
    script_directory = os.path.dirname(os.path.abspath(__file__))
    glcm_time_path = os.path.join(script_directory, "multiprocessing_results.py")
    
    with open(glcm_time_path, "a") as file:
        file.write(f"process_glcm_time = {process_glcm_time}\n")
    print("saved glcm excecution time")
    
    # Combine the features into a single list
    all_glcm_features = yes_glcm_features + no_glcm_features
    
    # Convert the list of dictionaries to a pandas DataFrame
    dataframe = pd.DataFrame(all_glcm_features)
    
    # Shuffle the DataFrame
    shuffled_dataframe = dataframe.sample(frac=1).reset_index(drop=True)

    return shuffled_dataframe

# Train models in parallel using multiprocessing
def train_models_parallel(models, X_train, X_test, y_train, y_test):
    """
    Train and evaluate multiple machine learning models in parallel.
    
    This function builds a list of arguments for each model using the provided training and test data.
    It then uses a multiprocessing pool to train and evaluate each model in parallel via the 'train_model' function.
    The function prints performance metrics for each model and saves the results to 'multiprocessing_results.py'.
    
    Args:
        models (dict): A dictionary mapping model names (str) to model instances.
        X_train (array-like): Training feature data.
        X_test (array-like): Testing feature data.
        y_train (array-like): Training labels.
        y_test (array-like): Testing labels.
    """
    args_list = [(name, model, X_train, X_test, y_train, y_test) for name, model in models.items()]
    with multiprocessing.Pool(processes=3) as pool:  # Adjust the number of processes as needed
        results = pool.map_async(train_model, args_list).get()
    
    # Print results
    for model_name, accuracy, cm, f1, precision, recall in results:
        print(f"{model_name}:\n\tAccuracy = {accuracy:.4f}\n\tConfusion Matrix: \n {cm} \n\tPrecision = {precision} \n\tRecall = {recall} \n\tF1 Score = {f1}")
    
        # Save results to a file
        script_directory = os.path.dirname(os.path.abspath(__file__))
        results_path = os.path.join(script_directory, "multiprocessing_results.py")
        
        with open(results_path, "a") as file:
            file.write(f"{model_name}_accuracy={accuracy} \n{model_name}_f1={f1} \n{model_name}_precision={precision} \n{model_name}_recall={recall} \n{model_name}_cm={cm.tolist()}\n")
        print("saved")

# main program
def main():
    """
    Main entry point of the image processing and model training pipeline.
    
    This function performs the following:
      - Executes the GLCM feature extraction pipeline by calling 'run_glcm' to obtain a shuffled DataFrame.
      - Splits the DataFrame into training and testing sets.
      - Defines a set of machine learning models.
      - Trains and evaluates the models in parallel using 'train_models_parallel'.
    """
    shuffled_dataframe = run_glcm()
    
    X = shuffled_dataframe.drop(['Tumor'], axis=1)
    y = shuffled_dataframe['Tumor']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    # Define models
    models = {"RandomForest": RandomForestClassifier(), "SVM": SVC(), "LogisticRegression": LogisticRegression()}
    train_models_parallel(models, X_train, X_test, y_train, y_test)


# run main
main()