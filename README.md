# MindSearch

## Introduction
MindSearch is an open source general search framework developed based on [MindSpore](https://www.mindspore.cn/en). 
It supports data preprocess, model training, model inference, index, and query service deployment of multiple models. 
MindSearch can solve problems such as comprehensiveness, ease-of-use, and fast construction, and provide users with an efficient search service platform.

## Major Features
- **Easy-to-use**: Friendly modular design for the overal search workflow, including data preprocess, model inference, query serving, etc.
- **State-of-art models**: MindSearch provides models of multiple languages, along with their pretrained weights.

## Installation
### Dependency
- mindspore >= 1.8.1
- tokenizers>=0.12.1
- numpy
- faiss
- onnx

To install the dependency, please run
```shell
pip install -r requirements.txt
```

## Get Started
### Quick Start Demo

See [examples](examples/retrieve_example.ipynb) in our code, this example shows how to use model for search. 

## Pre-Trained Models
We provide a list of pretrained models for search service, including Chinese and English.
- RetroMAE-base
- RetroMAE-pro
- RetroMAE-CN-base

### License

This project is released under the [Apache License 2.0](LICENSE.md).

# Acknowledgement
MindSearch is an open source project that welcome any contribution and feedback. We wish that MindSearch could serve the growing research community by providing a flexible as well as standardized platform to develop their own search service.

# Citation

If you find this project useful in your research, please consider citing:

```latex
@misc{ms_2022,
    author      = {Zheng Liu, Yingxia Shao},
    title       = {RetroMAE: Pre-training Retrieval-oriented Transformers via Masked Auto-Encoder},
    url         = {https://github.com/mindspore-ecosystem/mindsearch}
    year        = {2022}
}
```
