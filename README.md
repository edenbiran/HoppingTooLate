# Hopping Too Late: Exploring the Limitations of Large Language Models on Multi-Hop Queries

This repository contains the official code of the paper 
[Hopping Too Late: Exploring the Limitations of Large Language Models on Multi-Hop Queries](https://arxiv.org/abs/2406.12775).

## Setup

The benchmark creation and all experiments and evaluations were conducted in a Python 3.9 environment.
To clone the repository and set up the environment, please run the following commands:
```shell
git clone https://github.com/edenbiran/HoppingTooLate.git
cd HoppingTooLate
pip install -r requirements.txt
```

## Dataset

The dataset created for this work is available in `data/two_hop.csv`. 

Creating the dataset can be done using `src/create_dataset.py` and evaluating a model on the dataset on can be done using `src/evaluate_dataset.py`.

## Experiments

The experiments in the paper can be reproduced using the following scripts:
- `src/generate_entity_description.py` - The Patchscopes entity description experiments.
- `src/patch_activations.py` - The back-patching experiments.
- `src/project_sublayer.py` - The sublayer Projection experiments.
- `src/knockout_attention.py` - The attention knockout experiments.

## Results

Classifying the experiments results can be done using `src/classify_results.py` and analyzing the classified results can be done using `src/analyze_results.py`.

## Citation
```
@article{biran2024hopping,
  title={Hopping Too Late: Exploring the Limitations of Large Language Models on Multi-Hop Queries},
  author={Biran, Eden and Gottesman, Daniela and Yang, Sohee and Geva, Mor and Globerson, Amir},
  journal={arXiv preprint arXiv:2406.12775},
  year={2024}
}
```