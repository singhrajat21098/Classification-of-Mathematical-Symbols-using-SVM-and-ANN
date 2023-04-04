
import numpy as np
import cv2
import os
from imutils import paths
from skimage import feature
import pandas as pd




def load_dataset(dataset_folder_path):
    """
    This function loads the dataset from the dataset folder. In the dataset folder, every subfolder name is the name of label and the subfolder contains the images of that label.
    """
    # initialize the list of data and labels
    data = []
    labels = []
    # grab the image paths
    imagePaths = sorted(list(paths.list_images(dataset_folder_path)))
    # loop over the input images
    for imagePath in imagePaths:
        # load the images
        image = cv2.imread(imagePath)
        # extract the label from the image path, then update the
        # label and data lists
        label = str(imagePath.split(os.path.sep)[-2]).split("//")[-1]
        data.append(image)
        labels.append(label)
    
    print("Number of Images: ", len(data))
    return data, labels



# Preprocessing the data

def preprocess_data(data, normalise = False):
    """
    This function preprocesses the data by converting the images into grayscale and binarizing the images.

    """
    # convert the images into grayscale
    gray = [cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in data]
    # binarize the images
    binary_images = [cv2.threshold(image,127,255,cv2.THRESH_BINARY)[1] for image in gray]

    #print("Threshold: ", threshold)
    return [binary_image/255 for binary_image in binary_images] if normalise else binary_images





def convert_to_HOG(data, orientations=9, pixels_per_cell=(5, 5), cells_per_block=(2, 2), transform_sqrt=True, block_norm="L2-Hys"):
    """
    This function converts the images into HOG features.
    """

    hog_images = []
    for img in data:
        # compute HOG features
        H = feature.hog(img, orientations=orientations, pixels_per_cell = pixels_per_cell,
            cells_per_block = cells_per_block, transform_sqrt = transform_sqrt, block_norm = block_norm)
        
        hog_images.append(H)
    
    return hog_images



def convert_to_LBP(data, numPoints = 24, radius = 8, eps=1e-7):
    """
    This function converts the images into LBP features.
    """
    lbp_images = []
    for img in data:
        # compute LBP features
        lbp = feature.local_binary_pattern(img, numPoints, radius, method="uniform")
        (hist, _) = np.histogram(lbp.ravel(),
            bins=np.arange(0, numPoints + 3),
            range=(0, numPoints + 2))
        
        # normalize the histogram
        hist = hist.astype("float")

        hist /= (hist.sum() + eps)
        
        lbp_images.append(hist)
    
    return lbp_images



def convert_2d_to_1d(data):
    """
    This function converts the 2D array into 1D array.
    """
    data_1d = []
    for img in data:
        img_1d = img.ravel()
        data_1d.append(img_1d)
    return data_1d


