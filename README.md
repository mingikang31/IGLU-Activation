# IGLU: Integrated Gaussian Linear Unit 
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Accepted to the European Conference on Machine Learning and Principles and Practice of Knowledge Discovery in Databases **ECML-PKDD** 2026. 

## Paper 
- **IGLU: The Integrated Gaussian Linear Unit Activation**, Mingi Kang, Zai Yang, Jeova Farias, [[arXiv](https://arxiv.org/abs/2603.06861)] 

**Project Periods:** Fall 2025, Spring 2026 

## Overview
Through the introduction of transformer architectures, GELU and SiLU, a smooth, non-linear activation functions proved to show competitive advantages over ReLU in certain situations. This repository contains code for **IGLU**, a novel activation function, taking inspiration from GELU's Gaussian Error Phi Gating Function. Similar to GELU and SiLU, IGLU employs an arctan (and approximation) as the gating function with controllable parameter sigma. 

This repository explores image classification on CIFAR-10/100 datasets for vision models and WikiText103 dataset for language models for perplexity measurements. 

### Key Concepts
- **IGLU Activation Function**: Any standard neural network architecture can be adapted to use IGLU activation function in place of traditional activation functions. 
- **IGLU Approximation**: To reduce the computational burden of the arctan calculations, we provide *IGLU-Approximation* that is faster to compute and almost identical to the original IGLU formulation. 
- **Controlable Sigma Parameter**: Sigma parameter for *IGLU* and *IGLU-Approximate* allows for full control of model architecture. 
- **Vision Models**: This repository provides implementation of ResNet and Vision Transformer architectures for cross vision model evaluation. 
- **Language Models**: This repository provides implementation of GPT2 architecture for evaluating language models. 

### Implementation 
**GELU and SiLU Variation (`activation.py`)**: We are able to control the sharpness of the activation function controlled by a sigma parameter for both GELU and SiLU. 
- **`GELU_s`**: Modified GELU activation function to allow for controlable sigma function. 
- **`SiLU_s`**: Modified SiLU activation function to allow for controlable sigma function. 
**IGLU and IGLU Approximation (`activation.py`)**: The proposed activation function that generalizes ReLU, GELU, and SiLU. 
- **`ArcTan`**: ArcTan gating function used in `IGLU`. 
- **`ArcTan_Approx`**: ArcTan gating function used in `IGLU_Approx`.
- **`IGLU`**: IGLU activation function with ArcTan gating function. 
- **`IGLU_Approx`**: IGLU Approximation activation function with ArcTan Approximation gating function. 

## License

IGLU-Activation is released under the MIT License. Please see the [LICENSE](https://www.google.com/search?q=LICENSE) file for more information.

## Contributing

Contributions, issues, and question requests are welcome\!
Please reach out to:

  - **Mingi Kang** [mkang2@bowdoin.edu](mailto:mkang2@bowdoin.edu)
  - **Zai Yang** [zyang@bowdoin.edu](mailto:zyang@bowdoin.edu)
  - **Jeova Farias** [j.farias@bowdoin.edu](mailto:j.farias@bowdoin.edu)

