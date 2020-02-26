## TODO: Test with bounding boxes
## TODO: Test in context of classification pipeline
## TODO: Test in context of object detection pipeline
## TODO: Research common transforms and set as hyperparameters

import os
import numpy as np
import pandas as pd
import sys

from albumentations import *   # TODO: Modify this once I have finalized the transforms I will be using.
from PIL import Image

from collections import OrderedDict

from d3m import container, utils
from d3m.container import DataFrame as d3m_DataFrame
from d3m.metadata import base as metadata_base, hyperparams, params
from d3m.primitive_interfaces.base import PrimitiveBase, CallResult
from d3m.primitive_interfaces.transformer import TransformerPrimitiveBase

Inputs = container.pandas.DataFrame
Outputs = container.pandas.DataFrame

class Hyperparams(hyperparams.Hyperparams):
    transform_group = hyperparams.Union(
        OrderedDict({
            'image_classification_option_1': hyperparams.Constant[str](
                default = 'image_classification_option_1',
                semantic_types = ['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
                description = 'Apply image classification transforms'
            ),
            'object_detection_option_1': hyperparams.Constant[str](
                default = 'object_detection_option_1',
                semantic_types = ['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
                description = 'Apply object detection transforms'
            )
        }),
        default = 'image_classification_option_1',
        semantic_types = ['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
        description = "Image augmentation transform group to apply to images."
    )

    data_path = hyperparams.Hyperparameter[str](
        default = './augmented_training_images',
        semantic_types = ['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description = "The path where the augmented images should be exported to."
    )

class Params(params.Params):
    pass

class ImageAugmentationPrimitive(TransformerPrimitiveBase[Inputs, Outputs, Hyperparams]):
    """
    Primitive that utilizes the albumentations library to augment input image data set
    before training. The base library can be found at:
    https://github.com/NewKnowledge/albumentations.

    The primitive accepts a Dataset consisting of image paths and (optionally)
    bounding boxes as inputs. It performs user-specified transforms on the images
    (and on bounding boxes if present), stores these new images in a hyperparameter
    write location, and updates the learning data CSV with new rows for the augmented
    images.

    Only a selection of spatial transforms were included from the base library (pixel-
    level transforms are omitted entirely). These transforms will affect all images and
    masks but only affects bounding boxes in some cases. Some sets of transforms
    commonly used together are also included with the capability of randomly selecting
    which gets applied according to a probability hyperparameter.
    """

    metadata = metadata_base.PrimitiveMetadata(
        {
        'id': 'c20dec51-b1c6-4662-a79d-6c0ca5f1838f',
        'version': '0.1.0',
        'name': 'image_augmentation',
        'python_path': 'd3m.primitives.data_augmentation.image_augmentation.image_augmentation',
        'keywords': ['image augmentation', 'transforms', 'digital image processing'],
        'source': {
            'name': 'Distil',
            'contact': 'mailto:sanjeev@yonder.co',
            'uris': [
                'https://github.com/NewKnowledge/image-augmentation-d3m-wrapper',
            ],
        },
       'installation': [
            {
                'type': 'PIP',
                'package_uri': 'git+https://github.com/NewKnowledge/image-augmentation-d3m-wrapper.git@{git_commit}#egg=imageAugmentationD3MWrapper'.format(
                    git_commit = utils.current_git_commit(os.path.dirname(__file__)),)
            },
        ],
        'algorithm_types': [metadata_base.PrimitiveAlgorithmType.RETINANET], #TODO: Fix, obviously
        'primitive_family': metadata_base.PrimitiveFamily.DATA_AUGMENTATION,
        }
    )

    def __init__(self, *, hyperparams: Hyperparams) -> None:
        super().__init__(hyperparams = hyperparams)
        self.classification_options = ['image_classification_option_1', 'image_classification_option_2']
        self.object_detection_options = ['object_detection_option_1', 'object_detection_option_2']

    # def get_params(self) -> Params:
    #     return self._params

    # def set_params(self, *, params: Params) -> None:
    #     self.params = params

    def _augmentation(self, bbox_params, transform_group):
        if transform_group == self.classification_options[0]:
            return Compose([
                RandomRotate90(),
                RandomGridShuffle(grid = (3,3)),
                Blur(blur_limit = (300, 300)),
                ShiftScaleRotate(shift_limit = 0.0625, scale_limit = 0.50, rotate_limit = 45, p = .75)
            ])

        if transform_group == self.classification_options[1]:
            return Compose([
                RandomRotate90(),
                Transpose(),
                ShiftScaleRotate(shift_limit = 0.0625, scale_limit = 0.50, rotate_limit = 45, p = .75)
            ])

        if transform_group == self.object_detection_options[0]:
            return Compose([
                RandomRotate90(),
                Transpose(),
                ShiftScaleRotate(shift_limit = 0.0625, scale_limit = 0.50, rotate_limit = 45, p = .75)
            ])

        if transform_group == self.object_detection_options[1]:
            return Compose([
                RandomRotate90(),
                Transpose(),
                ShiftScaleRotate(shift_limit = 0.0625, scale_limit = 0.50, rotate_limit = 45, p = .75)
            ])

    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> CallResult[Outputs]:
        # Import images paths from learningData.csv
        image_cols = inputs.metadata.get_columns_with_semantic_type('https://metadata.datadrivendiscovery.org/types/FileName')
        base_dir = [inputs.metadata.query((metadata_base.ALL_ELEMENTS, t))['location_base_uris'][0].replace('file:///', '/') for t in image_cols]
        image_paths = np.array([[os.path.join(base_dir, filename) for filename in inputs.iloc[:,col]] for base_dir, col in zip(base_dir, image_cols)]).flatten()
        image_paths = pd.Series(image_paths)

        # Import bounding box coordinates from learningData.csv if needed
        bbox_params = None
        if self.hyperparams['transform_group'] in self.object_detection_options:
            bounding_coords = inputs.bounding_box.str.split(',', expand = True)
            bounding_coords = bounding_coords.drop(bounding_coords.columns[[2, 5, 6, 7]], axis = 1)
            bounding_coords.columns = ['x1', 'y1', 'y2', 'x2']
            bounding_coords = bounding_coords[['x1', 'y1', 'x2', 'y2']]
            bbox_params = bounding_coords

        # Assemble into the Compose function
        aug = self._augmentation(bbox_params = bbox_params, transform_group = self.hyperparams['transform_group'])
        # Apply compose function to image set
        export_path = self.hyperparams['data_path']

        for image_path in image_paths[0:9]:   # TODO: Remove 0:9 later
            im = np.array(Image.open(image_path))
            im_augmented = Image.fromarray(aug(image = im)['image'])
            img_name = os.path.basename(image_path)
            #print('image_path', file = sys.__stdout__)
            #print(image_path, file = sys.__stdout__)
            #print('img_name:', file = sys.__stdout__)
            #print(img_name, file = sys.__stdout__)
            img_extension = os.path.splitext(img_name)[1]
            img_name = os.path.splitext(img_name)[0] + '_augmented' + img_extension
            #print('img_name:', file = sys.__stdout__)
            #print(img_name, file = sys.__stdout__)
            im_augmented.save(export_path + img_name)

        # Add rows to learning data with duplicate images and their new rows
        results = inputs.copy()

        original_filename = results['filename'].apply(lambda x: os.path.basename(x))
        original_basename = original_filename.apply(lambda x:  os.path.splitext(x)[0])
        augmented_filename = results['filename'].apply(lambda x: os.path.splitext(x)[0]) + '_augmented.jpg'
        augmented_basename = augmented_filename.apply(lambda x: os.path.splitext(x)[0].split('_augmented')[0])

        # TODO: Augmented filename needs export_path appeneded to it
        # TODO: Need a way to address the fail condition here
        for row in range(0, results.shape[0]):
            if original_basename.loc[row] == augmented_basename.loc[row]:
                results.loc[row, 'filename'] = augmented_filename.loc[row]

        ## TODO: Append to learning data
        results_df = pd.concat([inputs, results])

        ## TODO: Assign the result to the output data frame

        return CallResult(None)

# """ WORKSPACE """
#     center_crop = hyperparams.Union(
#         OrderedDict({
#             'height': hyperparams.Constant[int](
#                 semantic_types = ['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
#                 description = 'height of the crop'
#             ),
#             'width': hyperparams.Constant[int](
#                 semantic_types = ['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
#                 description = 'width of the crop'
#             )
#         })
#     )

# max_delta_step = hyperparams.Union[Union[int, None]](
#         configuration=OrderedDict(
#             limit=hyperparams.Bounded[int](
#                 lower=1,
#                 upper=None,
#                 default=1,
#                 description='Maximum delta step we allow each leaf output to be.'
#             ),
#             unlimited=hyperparams.Enumeration[int](
#                 values=[0],
#                 default=0,
#                 description='No constraint.',
#             ),
#         ),
#         default='unlimited',
#         description='Maximum delta step we allow.',
#         semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
#     )

# original_filename = os.path.basename(results['filename'])[0]
#         augmented_filename = os.path.basename(os.listdir(export_path))[0]

#         if original_filename == augmented_filename:
#             results['filename'][0] = export_path[0]

#         results.apply(lambda row: _construct_augmented_image_table(results[]))

# def _construct_augmented_image_table(self, results, export_path):
#         original_filename = os.path.basename(results['filename'])
#         augmented_filename = os.path.basename(os.listdir(export_path))

#         if original_filename == augmented_filename:
#             return results['filename'] = export_path

        # test_results = pd.DataFrame({
        #     'inputs_filename': inputs['filename'],
        #     'results_filename': results['filename'],
        #     'inputs_bounding_box': inputs['bounding_box'],
        #     'results_bounding_box': results['bounding_box']
        # })

        # test_results.to_csv('/image-augmentation/imageAugmentationD3MWrapper/temp/test_results.csv', index = False)
