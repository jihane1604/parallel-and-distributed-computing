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
    processed_images = []
    for image in tqdm(images):
        filtered_images = filter_single_image(image)
        processed_images.append(filtered_images)
    return processed_images

def run_filtering():
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
    for name, model in models.items():
        model_name, accuracy, cm, f1, precision, recall = train_model((name, model, X_train, X_test, y_train, y_test))
        
        print(f"{model_name}:\n\tAccuracy = {accuracy:.4f}\n\tConfusion Matrix: \n {cm} \n\tPrecision = {precision} \n\tRecall = {recall} \n\tF1 Score = {f1}")
    
        # Save results to a file
        script_directory = os.path.dirname(os.path.abspath(__file__))
        results_path = os.path.join(script_directory, "sequential_results.py")
        
        with open(results_path, "a") as file:
            file.write(f"{model_name}_accuracy={accuracy} \n{model_name}_f1={f1} \n{model_name}_precision={precision} \n{model_name}_recall={recall} \n{model_name}_cm={cm.tolist()}")
        print("saved")
# main program
def main():
    shuffled_dataframe = run_glcm()
    
    X = shuffled_dataframe.drop(['Tumor'], axis=1)
    y = shuffled_dataframe['Tumor']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    # Define models
    models = {"RandomForest": RandomForestClassifier(), "SVM": SVC(), "LogisticRegression": LogisticRegression()}
    train_models(models, X_train, X_test, y_train, y_test)


# run main
main()
