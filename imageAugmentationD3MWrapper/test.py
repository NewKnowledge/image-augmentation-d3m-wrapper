"""Temporary testing file to test some of the image processing functions of albumentations"""

import os
import numpy as np
import pandas as pd

from albumentations import VerticalFlip, Transpose, Compose
from PIL import Image

"""Global vars"""
IMAGE_BASE_DIR = '/home/snamjoshi/docker/datasets/seed_datasets_current/LL1_penn_fudan_pedestrian_MIN_METADATA/SCORE/dataset_SCORE/media/'
EXPORT_PATH = '/home/snamjoshi/Documents/git_repos/image-augmentation-d3m-wrapper/imageAugmentationD3MWrapper/temp/'
CSV_PATH = '/home/snamjoshi/docker/datasets/seed_datasets_current/LL1_tidy_terra_panicle_detection_MIN_METADATA/TRAIN/dataset_TRAIN/tables/learningData.csv'

"""Functions"""
# def augmentation():
#     return Compose([VerticalFlip(), Transpose()]

def _construct_augmented_image_table(self, results, export_path):
        original_filename = os.path.basename(results['filename'])
        augmented_filename = os.path.basename(os.listdir(export_path))

        if original_filename == augmented_filename:
            return results['filename'] == export_path

"""Process images"""
img_names = os.listdir(IMAGE_BASE_DIR)
img_absolute_paths = [IMAGE_BASE_DIR + x for x in img_names]
img_paths_subset = img_absolute_paths[0:5]
img_names_subset = img_names[0:5]
aug = augmentation()

for image_path in img_paths_subset:
    im = Image.open(image_path)
    im_augmented = aug(image = im)['image']
    img_name = os.path.basename(image_path)
    im_augmented.save(EXPORT_PATH + img_names)

"""Process table"""
results_table = pd.read_csv(CSV_PATH)

original_filename = results_table['image'].apply(lambda x: os.path.basename(x))
augmented_filename = results_table['image'].apply(lambda x: os.path.splitext(x)[0]) + '_augmented.jpg'

extracted_original_basename = original_filename.apply(lambda x:  os.path.splitext(x)[0])
extracted_augmented_basename = augmented_filename.apply(lambda x: os.path.splitext(x)[0].split('_augmented')[0])

# TODO: Need a way to address the fail condition here
for row in range(0, results_table.shape[0]):
    if extracted_original_basename.loc[row] == extracted_augmented_basename.loc[row]:
        results_table.loc[row, 'image'] = augmented_filename.loc[row]
