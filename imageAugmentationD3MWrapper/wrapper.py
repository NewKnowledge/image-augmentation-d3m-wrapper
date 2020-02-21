import os
import numpy as np
import pandas as pd

from albumentations import *   # TODO: Modify this once I have finalized the transforms I will be using.
from PIL import Image

Inputs = container.pandas.DataFrame
Outputs = container.pandas.DataFrame

class Hyperparams(hyperparams.Hyperparams):
    center_crop = hyperparams.Union(
        OrderedDict({
            'height': hyperparams.Constant[int](
                semantic_types = ['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
                description = 'height of the crop'
            ),
            'width': hyperparams.Constant[int](
                semantic_types = ['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
                description = 'width of the crop'
            )
        }),
        default = False,
        semantic_types = ['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
        description = "Crop the central part of the input."
    )
    vertical_flip = hyperparams.Hyperparameter[bool](
        default = False,
        semantic_types = ['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description = "Flip the input vertically around the x-axis."
    )

    bounding_boxes = hyperparams.Hyperparameter[bool](
        default = False,
        semantic_types = ['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description = "Indicates whether or not the dataset has bounding boxes (an object detection task)"
    )

    data_path = hyperparams.Hyperparameter[str](
        default = './augmented_training_images',
        semantic_types = ['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description = "The path where the augmented images should be exported to."
    )

class Params(params.Params):
    pass

class ImageAugmentationPrimitive(PrimitiveBase[Inputs, Outputs, Params, Hyperparams]):
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
                '',
            ],
        },
       'installation': [
            {
                'type': 'PIP',
                'package_uri': ''
            },
        ],
        'algorithm_types': [metadata_base.PrimitiveAlgorithmType.IMAGE_AUGMENTATION],
        'primitive_family': metadata_base.PrimitiveFamily.DATA_AUGMENTATION,
        }
    )

    def __init__(self, *, hyperparams: Hyperparams) -> None:
        super().__init__(hyperparams = hyperparams)
        self.vertical_flip_params = None
        self.center_crop_params = None

    def get_params(self) -> Params:
        return self._params

    def set_params(self, *, params: Params) -> None:
        self.params = params

    def _get_params(self):
        if self.hyperparams['vertical_flip'] is True:
            self.vertical_flip_params = self.hyperparams['vertical_flip']

        if self.hyperparams['center_crop'] is True:
            self.center_crop_params = self.hyperparams['center_crop']

    def _transformations(self):
        pass

    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> CallResult[Outputs]:
        # Import images paths from learningData.csv
        image_cols = inputs.metadata.get_columns_with_semantic_type('https://metadata.datadrivendiscovery.org/types/FileName')
        base_dir = [inputs.metadata.query((metadata_base.ALL_ELEMENTS, t))['location_base_uris'][0].replace('file:///', '/') for t in image_cols]
        image_paths = np.array([[os.path.join(base_dir, filename) for filename in inputs.iloc[:,col]] for base_dir, col in zip(base_dir, image_cols)]).flatten()
        image_paths = pd.Series(image_paths)

        # Import bounding box coordinates from learningData.csv if specified
        bbox_params = None
        if self.hyperparams['bounding_boxes'] is True:
            bounding_coords = inputs.bounding_box.str.split(',', expand = True)
            bounding_coords = bounding_coords.drop(bounding_coords.columns[[2, 5, 6, 7]], axis = 1)
            bounding_coords.columns = ['x1', 'y1', 'y2', 'x2']
            bounding_coords = bounding_coords[['x1', 'y1', 'x2', 'y2']]
            bbox_params = bounding_coords

        # Parse hyperparams and assemble into the Compose function
        # Should maybe just have a function where if you pass it a hyperparameter, it will assemble the transform for you.
        _get_params()
        _augmentation(bbox_params = bbox_params)

        # apply compose function to image set
        # and export images to the correct folder. Probably just loop over each image in image_paths

        export_path = self.hyperparameters['data_path']   # Might need to be part of a function that does the exporting

        for image_path in image_paths:
            im = Image.open(image_path)
            im_augmented = _augmentation(image = im)['image']
            img_name = os.path.basename(image_path)
            im_augmented.save(export_path + img_name)

        # add rows to learning data with duplicate images and their new rows
        # return the resulting data frame to be based on to the next step in the pipeline
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
#                 # TODO: 1-10 instead?
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