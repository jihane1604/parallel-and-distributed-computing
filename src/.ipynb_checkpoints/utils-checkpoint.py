import glob
import cv2
from skimage.filters.rank import entropy
from skimage.morphology import disk
from scipy import ndimage as nd
from skimage.filters import sobel, gabor, hessian, prewitt
import numpy as np
import skimage.feature as feature
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import multiprocessing

# function to read the images
def read_images(images_path):
    """
    Reads all images from a specified path using OpenCV.

    Parameters:
        - images_path (str): The path to the directory containing the images.
    Returns:
        - images (list): A list of images read from the directory.
    """
    images = []
    for file_path in images_path:
        image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        if image is not None:
                images.append(image)
    return images

# function to open the images
def open_images():
    """
    Opens and loads images from the brain tumor dataset.

    The dataset is expected to be organized in a directory structure with two subdirectories:
    'yes' (images with tumors) and 'no' (images without tumors). This function reads the image file
    paths from these directories and loads them into memory using the `read_images` utility.

    Returns:
        tuple: A tuple containing two lists:
            - yes_images (list): List of images from the 'yes' directory.
            - no_images (list): List of images from the 'no' directory.
    """
    # Define the path to the dataset
    dataset_path = '../data/brain_tumor_dataset/'
    
    # List all image files in the 'yes' and 'no' directories
    yes_images = glob.glob(dataset_path + 'yes/*')
    no_images = glob.glob(dataset_path + 'no/*')
    
    yes_images = read_images(yes_images)
    no_images = read_images(no_images)

    return yes_images, no_images

# utility function to filter one image
def filter_single_image(image):
    """
    Applies a set of image filtering operations to a single image.

    This function takes an image and computes several filtered versions using different techniques:
    entropy filtering, Gaussian smoothing, Sobel edge detection, Gabor filtering, Hessian filtering, 
    and Prewitt filtering. The original image is also included in the output dictionary.

    Args:
        image (ndarray): The input image to be filtered.

    Returns:
        dict: A dictionary containing the filtered images with the following keys:
            - 'Original': The original input image.
            - 'Entropy': The entropy-filtered image.
            - 'Gaussian': The image after Gaussian filtering (sigma=1).
            - 'Sobel': The result of applying the Sobel edge detector.
            - 'Gabor': The Gabor filter response (the second output of the Gabor function).
            - 'Hessian': The result of applying the Hessian filter with a range of sigmas.
            - 'Prewitt': The result of the Prewitt edge detection.
    """
    return {
        'Original': image,
        'Entropy': entropy(image, disk(2)),
        'Gaussian': nd.gaussian_filter(image, sigma=1),
        'Sobel': sobel(image),
        'Gabor': gabor(image, frequency=0.9)[1],
        'Hessian': hessian(image, sigmas=range(1, 100, 1)),
        'Prewitt': prewitt(image)
    }

def apply_hessian(args):
    """
    Applies the Hessian filter to a subset of sigma values for a given image.

    This function is intended to be used in parallel processing. It extracts the image and sigma range parameters
    from the input tuple and computes the Hessian filter over the specified sigma range.

    Args:
        args (tuple): A tuple containing:
            - image (ndarray): The input image.
            - start (int): The starting value of sigma.
            - end (int): The ending value (exclusive) of sigma.

    Returns:
        ndarray: The Hessian-filtered image computed over the specified sigma range.
    """
    image, start, end = args
    return hessian(image, sigmas=range(start, end, 1))

def filter_single_image_parallel(image):
    """
    Applies multiple image filtering operations to a single image using parallel processing for the Hessian filter.

    This function filters an image by applying various filters including entropy, Gaussian, Sobel, Gabor, and Prewitt sequentially.
    The Hessian filter is computed in parallel over a range of sigma values (1 to 98) using multiple processes,
    and the results are averaged to obtain the final Hessian response.

    Args:
        image (ndarray): The input image to be filtered.

    Returns:
        dict: A dictionary containing the filtered images with the following keys:
            - 'Original': The original input image.
            - 'Entropy': The entropy-filtered image.
            - 'Gaussian': The image after Gaussian filtering (sigma=1).
            - 'Sobel': The result of applying the Sobel edge detector.
            - 'Gabor': The Gabor filter response (the second output of the Gabor function).
            - 'Hessian': The averaged result of applying the Hessian filter in parallel.
            - 'Prewitt': The result of the Prewitt edge detection.
    """
    args = [(image, i, i+1) for i in range(1,99)]
    # Start Hessian filtering in a separate process
    with multiprocessing.Pool(processes=6) as pool:  # Adjust number of processes based on your system
        hessian_result = pool.map_async(apply_hessian, args).get()
        
        # Apply other filters sequentially
        filtered_images = {
            'Original': image,
            'Entropy': entropy(image, disk(2)),
            'Gaussian': nd.gaussian_filter(image, sigma=1),
            'Sobel': sobel(image),
            'Gabor': gabor(image, frequency=0.9)[1],
            'Hessian': np.mean(hessian_result, axis = 0),
            'Prewitt': prewitt(image)
        }

    return filtered_images

# Function to compute GLCM features for an image
def compute_glcm_features(image, 
                                                    filter_name):
    """
    Computes GLCM (Gray Level Co-occurrence Matrix) features for an image.

    Parameters:
    - image: A 2D array representing the image. Should be in grayscale.
    - filter_name: A string representing the name of the filter applied to the image.

    Returns:
    - features: A dictionary containing the computed GLCM features. The keys are
        formatted as "{filter_name}_{feature_name}_{angle_index}", where "angle_index"
        corresponds to the index of the angle used for the GLCM calculation (1-based).
        The features include contrast, dissimilarity, homogeneity, energy, correlation,
        and ASM (Angular Second Moment) for each angle (0, π/4, π/2, 3π/4).

    Notes:
    - The image is first converted from float to uint8 format, as the graycomatrix
        function expects integer values.
    - The GLCM is computed using four angles (0, π/4, π/2, 3π/4) with a distance of 1.
    - The GLCM properties are computed and flattened into a 1D array to handle multiple
        angles. Each property value for each angle is stored as a separate key in the
        resulting dictionary.
    """
    # Convert the image from float to int
    image = (image * 255).astype(np.uint8)

    # Compute the GLCM
    graycom = feature.graycomatrix(image, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4], levels=256, symmetric=True, normed=True)

    # Compute GLCM properties
    features = {}
    for prop in ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']:
            values = feature.graycoprops(graycom, prop).flatten()
            for i, value in enumerate(values):
                    features[f'{filter_name}_{prop}_{i+1}'] = value
    return features

# utility function to process one image
def process_single_image(args):
    """
    Processes a single image by computing its GLCM (Gray-Level Co-occurrence Matrix) features.

    This function iterates over a dictionary of filtered images, computes GLCM features for each filter output,
    and aggregates the results into a single dictionary. It also attaches a tumor presence label to the results.

    Args:
        args (tuple): A tuple containing:
            - filtered_images (dict): A dictionary where keys are filter names and values are the corresponding filtered images.
            - tumor_presence (bool or int): A label indicating the presence (or absence) of a tumor in the image.

    Returns:
        dict: A dictionary containing GLCM features for each filter as well as the tumor presence label under the key 'Tumor'.
    """
    filtered_images, tumor_presence = args
    glcm_features = {}
    for key, image in filtered_images.items():
        glcm_features.update(compute_glcm_features(image, key))
    glcm_features['Tumor'] = tumor_presence
    return glcm_features

# Function to train and evaluate a model
def train_model(args):
    """
    Trains a machine learning model and evaluates its performance on a test dataset.

    This function fits the provided model using the training data, makes predictions on the test data,
    and calculates performance metrics including accuracy, confusion matrix, F1 score, precision, and recall.

    Args:
        args (tuple): A tuple containing:
            - name (str): The name or identifier for the model.
            - model: An instance of a machine learning model with a fit and predict method.
            - X_train (array-like): Training feature data.
            - X_test (array-like): Test feature data.
            - y_train (array-like): Training labels.
            - y_test (array-like): Test labels.

    Returns:
        tuple: A tuple containing:
            - name (str): The model name.
            - accuracy (float): The accuracy of the model on the test data.
            - cm (ndarray): The confusion matrix.
            - f1 (float): The F1 score (macro averaged).
            - precision (float): The precision score (macro averaged).
            - recall (float): The recall score (macro averaged).
    """
    name, model, X_train, X_test, y_train, y_test = args
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    return name, accuracy, cm, f1, precision, recall