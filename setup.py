from setuptools import setup

setup(
    name              = 'imageAugmentationD3MWrapper',
    version           = '0.1.0',
    description       = 'Primitive that utilizes the albumentations library to augment input image data set before training. ',
    author            = 'Sanjeev Namjoshi',
    author_email      = 'sanjeev@yonder.co',
    packages          = ['imageAugmentationD3MWrapper'],
    install_requires  = ['numpy>=1.15.4,<=1.17.3',
                         'albumentations @ git+https://github.com/NewKnowledge/albumentations@7a9dc691491fa0d691bbdf8603e5b8cd512f3281#egg=albumentations'],
    entry_points      = {
        'd3m.primitives': [
            'data_augmentation.image_augmentation.image_augmentation = imageAugmentationD3MWrapper:ImageAugmentationPrimitive'
        ],
    },
)

