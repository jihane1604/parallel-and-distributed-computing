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
    """

    with ProcessPoolExecutor(max_workers = 6) as executor:
        results = list(tqdm(executor.map(filter_single_image_parallel, images), total=len(images)))
    return results

# function to process images 
def process_images_parallel(images_list, tumor_presence):
    with multiprocessing.Pool(processes = 6) as pool:
        glcm_features_list = pool.map_async(process_single_image, [(filtered_image, tumor_presence) for filtered_image in images_list])
        result = glcm_features_list.get()
    return result

def run_filtering():
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
    shuffled_dataframe = run_glcm()
    
    X = shuffled_dataframe.drop(['Tumor'], axis=1)
    y = shuffled_dataframe['Tumor']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    # Define models
    models = {"RandomForest": RandomForestClassifier(), "SVM": SVC(), "LogisticRegression": LogisticRegression()}
    train_models_parallel(models, X_train, X_test, y_train, y_test)


# run main
main()