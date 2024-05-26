from setuptools import setup, find_packages

setup(
    name='image_colorization',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'numpy==1.22.4',
        'Pillow==9.1.0',
        'pathlib==1.0.1',
        'tqdm==4.64.0',
        'matplotlib==3.5.2',
        'scikit-image==0.19.3',
        'torch==1.12.0',
        'torchvision==0.13.0',
        'torchsummary==1.5.1',
        'torchviz==0.0.2',
        'tensorboard==2.9.1',
        'opencv-python-headless==4.6.0.66',
        'torch_xla'  # For TPU support, version may depend on the TPU setup
    ],
    entry_points={
        'console_scripts': [
            'colorize-train=cli.train:main',
            'colorize-evaluate=cli.evaluate:main',
        ],
    },
)
