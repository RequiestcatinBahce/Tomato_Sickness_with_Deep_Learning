import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import models, layers 
import os
import shutil
import pathlib

DATASET_PATH = pathlib.Path('C:\\Users\\...\\Leaves')

def load_image(DATASET_PATH):
    image = cv2.imread(DATASET_PATH, cv2.IMREAD_GRAYSCALE)
    blurred_image = cv2.GaussianBlur(image, (5, 5), 0)
    return blurred_image

def load_image_and_display_channels(DATASET_PATH):
    for image in DATASET_PATH.glob('*.jpg'):
        image = cv2.imread(DATASET_PATH, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        red_channel = image[:, :, 0]
        green_channel = image[:, :, 1]
        blue_channel = image[:, :, 2]
        nir_channel = image[:, :, 3] 
        
        red_channel = red_channel.astype(np.float32)
        green_channel = green_channel.astype(np.float32)
        blue_channel = blue_channel.astype(np.float32)
        nir_channel = nir_channel.astype(np.float32)

        red_color = np.zeros_like(image)
        red_color[:, :, 0] = red_channel

        green_color = np.zeros_like(image)
        green_color[:, :, 1] = green_channel

        blue_color = np.zeros_like(image)
        blue_color[:, :, 2] = blue_channel

        fig, axs = plt.subplots(1, 3, figsize=(15, 5))

        axs[0].imshow(red_color)
        axs[0].set_title('Red Channel')
        axs[0].axis('off')

        axs[1].imshow(green_color)
        axs[1].set_title('Green Channel')
        axs[1].axis('off')

        axs[2].imshow(blue_color)
        axs[2].set_title('Blue Channel')
        axs[2].axis('off')

        plt.show()
    
        bottom = (nir + red)
        bottom[bottom == 0] = 0.01
        ndvi = (nir - red) / bottom
        print("NDVI Shape:", ndvi.shape)
        print("NDVI Value at (0,0):", ndvi[0, 0])

        ndvi = cv2.normalize(ndvi, None, 0, 255, cv2.NORM_MINMAX)
        ndvi = ndvi.astype(np.uint8)
        return ndvi

  def classify_ndvi(ndvi):
    classification = np.zeros_like(ndvi, dtype = int)
    classification[(ndvi >= 0) & (ndvi < 0.33)] = 1
    classification[(ndvi >= 0.33) & (ndvi < 0.66)] = 2
    classification[(ndvi >= 0.66) & (ndvi <= 1)] = 3
    return classification

def classify_and_move_images(healthy_folder, moderately_healthy_folder, unhealthy_folder):
   healthy_folder = pathlib.Path('C:\\Users\\baska\\...\\healthy_folder')
   moderately_healthy_folder = pathlib.Path('C:\\Users\\...\\Leaves\\moderately_healthy_folder')
   unhealthy_folder = pathlib.Path('C:\\Users\\...\\Leaves\\unhealthy_folder')
    
   leaves_folder_path = pathlib.Path("plants")
   
   image_paths = [os.path.join(leaves_folder_path, file) for file in os.listdir(leaves_folder_path) if file.endswith(('.jpg'))]

   for image_path in image_paths:
        
        if  ndvi > 0.6: 
            shutil.move(image_path, os.path.join(healthy_folder, os.path.basename(image_path)))
        elif 0.33 <= ndvi <= 0.6:  
            shutil.move(image_path, os.path.join(moderately_healthy_folder, os.path.basename(image_path)))
        else:  
            shutil.move(image_path, os.path.join(unhealthy_folder, os.path.basename(image_path)))
