# Lab 4 Part 2: Image Processing for Tumor Detection

## Overview
This lab demonstrates the use of image processing techniques to detect tumors in medical images. The notebook guides you through preprocessing, segmentation, feature extraction, and tumor detection steps. The objective is to automate and improve the accuracy of tumor detection in medical imaging using Python.

## Objectives
- **Preprocessing:** Prepare and normalize the medical images for analysis.
- **Segmentation:** Separate the region of interest (tumor) from the background using techniques such as thresholding and morphological operations.
- **Feature Extraction:** Extract relevant features from the segmented tumor regions.
- **Tumor Detection:** Classify and detect the presence of tumors using the extracted features or a machine learning/deep learning model.
- **Evaluation:** Assess the performance of the detection method using accuracy, precision, recall, or other metrics.

## Tools and Libraries
- **Python:** The primary programming language.
- **Jupyter Notebook:** For interactive development and visualization.
- **Image Processing Libraries:** Such as OpenCV, scikit-image, or PIL.
- **Machine Learning Libraries (if applicable):** e.g., scikit-learn, TensorFlow, or PyTorch.
- **Visualization Tools:** Matplotlib or Seaborn for displaying results.

## Methodology
1. **Data Loading and Preprocessing:**
   - Import the dataset containing medical images.
   - Apply normalization and resizing to standardize the images.
2. **Image Segmentation:**
   - Implement segmentation techniques (e.g., thresholding, edge detection, or region-based methods) to isolate potential tumor regions.
   - Apply morphological operations to refine the segmentation.
3. **Feature Extraction and Tumor Detection:**
   - Extract features from the segmented regions.
   - Train and test a classifier (or use a pre-trained model) to distinguish between tumor and non-tumor areas.
4. **Evaluation and Visualization:**
   - Compute performance metrics such as accuracy, precision, recall, and F1-score.
   - Visualize segmentation outputs and detection results alongside the original images.

## Results
- **Best Metrics:**
  - Accuracy: (RandomForest Model) 0.76
  - Precision: (RandomForest Model) 0.75
  - Recall: (RandomForest Model) 0.73
  - F1-Score: (RandomForest Model) 0.74
- **Sequential Execution Time:** 10860.88
- **Multiprocessing Execution Time:** 1883.95
- **Multiprocessing Speedup:** 5.76
- **Multiprocessing Efficiency:** 0.96
- **Threading Execution Time:** 2432.47
- **Threading Speedup:** 4.46
- **Threading Efficiency:** 0.74
## Conclusion
Applying one of the filters was taking longer than all the other filters, so parallelizing it using pools made the execution a lot faster.


## How to Run
Run `main.py` file to get the results :3
