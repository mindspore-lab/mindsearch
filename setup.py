# coding=utf-8
from setuptools import setup, find_packages

setup(
    name='mindsearch',
    version='0.0.1',
    package_dir={"": "src"},
    packages=find_packages("src"),
    install_requires=[
        'mindspore>=1.8.1',
        'tokenizers>=0.12.1',
        'numpy',
        'faiss',
        'onnx'
    ],
)
