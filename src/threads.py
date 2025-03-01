import glob
import cv2
from skimage.filters.rank import entropy
from skimage.morphology import disk
from scipy import ndimage as nd
from skimage.filters import sobel, gabor, hessian, prewitt
import threading
from concurrent.futures import ThreadPoolExecutor
import time
from tqdm import tqdm
import os

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


# utility function
def process_single_image(image):
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

# threading 
def process_images_thread(images):
    """
    """
    processed_images = []
    
    with ThreadPoolExecutor() as executor:
        results = list(tqdm(executor.map(process_single_image, images), total=len(images)))
    return results


# Define the path to the dataset
dataset_path = '../data/brain_tumor_dataset/'

# List all image files in the 'yes' and 'no' directories
yes_images = glob.glob(dataset_path + 'yes/*')
no_images = glob.glob(dataset_path + 'no/*')

yes_images = read_images(yes_images[:5])
no_images = read_images(no_images[:5])

# run the threads
start_time = time.time()
yes_inputs = process_images_thread(yes_images)
no_inputs = process_images_thread(no_images)
end_time = time.time()

execution_time = end_time - start_time
print(f"Threading execution time: {execution_time} seconds")

# Save execution time to a file
script_directory = os.path.dirname(os.path.abspath(__file__))
execution_time_path = os.path.join(script_directory, "threading_execution_time.txt")

with open(execution_time_path, "w") as file:
    file.write(f"Execution Time: {execution_time} seconds\n")

print(f"Execution time saved to {execution_time_path}")
