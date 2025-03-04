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
    """Applies the Hessian filter in a separate process."""
    image, start, end = args
    return hessian(image, sigmas=range(start, end, 1))

def filter_single_image_parallel(image):
    """Applies multiple filters to an image, using multiprocessing for Hessian."""
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
    """
    filtered_images, tumor_presence = args
    glcm_features = {}
    for key, image in filtered_images.items():
        glcm_features.update(compute_glcm_features(image, key))
    glcm_features['Tumor'] = tumor_presence
    return glcm_features

# Function to train and evaluate a model
def train_model(args):
    name, model, X_train, X_test, y_train, y_test = args
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    return name, accuracy, cm, f1, precision, recall