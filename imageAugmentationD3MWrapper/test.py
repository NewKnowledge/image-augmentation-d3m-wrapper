"""Temporary testing file to test some of the image processing functions of albumentations"""

import os
import numpy as np

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
            return results['filename'] = export_path

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
