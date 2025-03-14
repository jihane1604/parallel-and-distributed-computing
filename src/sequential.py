import glob
import cv2
from utils import open_images, filter_single_image, process_single_image, train_model, compute_glcm_features
import time
from tqdm import tqdm
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
from sklearn.svm import SVC


def filter_images(images):
    """
    Apply image filtering sequentially to a list of images.

    This function iterates through the list of images, applying the
    'filter_single_image' function to each image. A progress bar is displayed
    using tqdm to track the processing progress.

    Args:
        images (list): A list of image objects (e.g., NumPy arrays) to be filtered.

    Returns:
        list: A list of dictionaries, each containing the filtered versions of an image.
              The dictionary keys correspond to different filter names.
    """
    processed_images = []
    for image in tqdm(images):
        filtered_images = filter_single_image(image)
        processed_images.append(filtered_images)
    return processed_images

def run_filtering():
    """
    Open and filter images sequentially from the dataset, and record the execution time.

    This function performs the following steps:
      - Opens images from the dataset using the 'open_images' function.
      - Applies the 'filter_images' function to both 'yes' (tumor) and 'no' (non-tumor) images.
      - Measures and prints the execution time for the filtering process.
      - Saves the execution time to a file named 'sequential_results.py'.

    Returns:
        tuple: Two lists containing the filtered image dictionaries for 'yes' images and 'no' images, respectively.
    """
    # open the yes and no images
    yes_images, no_images = open_images()
    print("applying filters to images")
    # run the processes
    start_time = time.time()
    yes_inputs = filter_images(yes_images)
    no_inputs = filter_images(no_images)
    end_time = time.time()
    
    seq_filtering_time = end_time - start_time
    print(f"Sequential execution time: {seq_filtering_time} seconds")
    
    # Save execution time to a file
    script_directory = os.path.dirname(os.path.abspath(__file__))
    execution_time_path = os.path.join(script_directory, "sequential_results.py")
    
    with open(execution_time_path, "w") as file:
        file.write(f"seq_time = {seq_filtering_time}\n")
    print("saved filtering excecution time")

    return yes_inputs, no_inputs

def process_images(images_list, tumor_presence):
    """
    Compute GLCM features for a list of filtered images and label them with tumor presence.

    For each image in the input list, this function iterates over its filtered versions,
    computes the Gray-Level Co-occurrence Matrix (GLCM) features using 'compute_glcm_features',
    and aggregates the results into a dictionary. The tumor presence label is added to the dictionary.

    Args:
        images_list (list): A list of dictionaries where each dictionary contains the filtered outputs for an image.
        tumor_presence (int or bool): Label indicating tumor presence (e.g., 1 for tumor, 0 for non-tumor).

    Returns:
        list: A list of dictionaries, each containing the computed GLCM features and the tumor label.
    """
    # Apply all filters to each image and compute GLCM features
    glcm_features_list = []
    for filtered_images in images_list:
        glcm_features = {}
        for key, image in filtered_images.items():
            glcm_features.update(compute_glcm_features(image, key))
        glcm_features['Tumor'] = tumor_presence
        glcm_features_list.append(glcm_features)
    return glcm_features_list

def run_glcm():
    """
    Process filtered images to compute GLCM features and return a shuffled DataFrame.

    This function performs the following:
      - Calls 'run_filtering' to obtain filtered 'yes' and 'no' images.
      - Processes each set using 'process_images' to compute GLCM features.
      - Measures and prints the execution time for GLCM processing.
      - Appends the GLCM processing time to 'sequential_results.py'.
      - Combines the features into a single pandas DataFrame and shuffles it.

    Returns:
        pandas.DataFrame: A shuffled DataFrame containing the computed GLCM features and tumor labels.
    """
    yes_inputs, no_inputs = run_filtering()
    print("processing glcm")
    # Process the 'yes' and 'no' image lists
    start_time = time.time()
    yes_glcm_features = process_images(yes_inputs, 1)
    no_glcm_features = process_images(no_inputs, 0)
    end_time = time.time()
    
    seq_glcm_time = end_time - start_time
    print(f"GLCM processing execution time: {seq_glcm_time} seconds")
    
    script_directory = os.path.dirname(os.path.abspath(__file__))
    glcm_time_path = os.path.join(script_directory, "sequential_results.py")
    
    with open(glcm_time_path, "a") as file:
        file.write(f"seq_glcm_time = {seq_glcm_time}\n")
    print("saved glcm excecution time")
    
    # Combine the features into a single list
    all_glcm_features = yes_glcm_features + no_glcm_features
    
    # Convert the list of dictionaries to a pandas DataFrame
    dataframe = pd.DataFrame(all_glcm_features)
    
    # Shuffle the DataFrame
    shuffled_dataframe = dataframe.sample(frac=1).reset_index(drop=True)

    return shuffled_dataframe

def train_models(models, X_train, X_test, y_train, y_test):
    """
    Train and evaluate machine learning models sequentially, printing and saving performance metrics.

    For each model provided in the 'models' dictionary, this function trains the model using the training data,
    evaluates it on the test data via the 'train_model' function, and prints the accuracy, confusion matrix,
    precision, recall, and F1 score. The results are also saved to a file named 'sequential_results.py'.

    Args:
        models (dict): A dictionary mapping model names (str) to model instances.
        X_train (array-like): Training features.
        X_test (array-like): Testing features.
        y_train (array-like): Training labels.
        y_test (array-like): Testing labels.
    """
    for name, model in models.items():
        model_name, accuracy, cm, f1, precision, recall = train_model((name, model, X_train, X_test, y_train, y_test))
        
        print(f"{model_name}:\n\tAccuracy = {accuracy:.4f}\n\tConfusion Matrix: \n {cm} \n\tPrecision = {precision} \n\tRecall = {recall} \n\tF1 Score = {f1}")
    
        # Save results to a file
        script_directory = os.path.dirname(os.path.abspath(__file__))
        results_path = os.path.join(script_directory, "sequential_results.py")
        
        with open(results_path, "a") as file:
            file.write(f"{model_name}_accuracy={accuracy} \n{model_name}_f1={f1} \n{model_name}_precision={precision} \n{model_name}_recall={recall} \n{model_name}_cm={cm.tolist()}\n")
        print("saved")
# main program
def main():
    """
    Main entry point for the image processing and model training pipeline.

    This function performs the following steps:
      - Executes the GLCM feature extraction pipeline by calling 'run_glcm' to obtain a shuffled DataFrame.
      - Splits the DataFrame into training and testing sets.
      - Defines machine learning models.
      - Trains and evaluates the models by calling 'train_models'.

    Returns:
        None
    """
    shuffled_dataframe = run_glcm()
    
    X = shuffled_dataframe.drop(['Tumor'], axis=1)
    y = shuffled_dataframe['Tumor']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    # Define models
    models = {"RandomForest": RandomForestClassifier(), "SVM": SVC(), "LogisticRegression": LogisticRegression()}
    train_models(models, X_train, X_test, y_train, y_test)


# run main
main()
