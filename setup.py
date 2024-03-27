from setuptools import find_packages
from setuptools import setup

with open("README.md", "r") as f:
    long_description = f.read()

setup(
    name='nam-interpret',
    version='0.0.0',
    description="Neural Additive Models for InterpretML",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="InterpretML Team",
    url="https://github.com/lemeln/nam",
    packages=find_packages(),
    install_requires=[
        "torch",
        "scikit-learn",
        "numpy",
        "pandas",
        "loguru",
        "tensorboard",
        "tqdm"
    ],
    classifiers=[
        "Programming Language :: Python :: 3.7",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
