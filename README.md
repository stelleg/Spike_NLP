# Spike_NLP
Use NLP to study the spike protein in SARS-CoV-2 virus. [Preprint](https://www.biorxiv.org/content/10.1101/2024.04.05.588133v1.full)

## Table of Contents
* [Purpose](https://github.com/hubin-keio/Spike_NLP?tab=readme-ov-file#purpose)

* [Installation](https://github.com/hubin-keio/Spike_NLP?tab=readme-ov-file#installation)

* [Usage](https://github.com/hubin-keio/Spike_NLP?tab=readme-ov-file#usage)

* [Citation](https://github.com/hubin-keio/Spike_NLP?tab=readme-ov-file#citation)

## Purpose
Understand the sequence to function, or genotype-phenotype, relationship of proteins by utilizing a language model-based approach. In particular, focusing on tailoring protein language models to predict protein mutation phenotypes, such as binding affinity or level of expression.

## Installation 
To match the packages found in our conda environment, spike_env, you can run `conda env create -f environment.yml`. After activating the conda environment, the pnlp module can be installed using `pip install -e .`. The pnlp module is necessary for running the models using the model runners. 

Other requirements:
- NVIDIA GPU

## Usage
We offer more in depth documentation located in the [notebooks](https://github.com/hubin-keio/Spike_NLP/tree/new_manuscript/notebooks) folder, which we recommend reading for further understanding before usage of the models. 

* I will update this with notebook details (WIP, 11/12/24) 

### Quickstart: Running the NLP BERT Model
Executing
  * There are 2 models for the NLP BERT model located in the `src/pnlp/runner` folder.
    * To run model runner ([link](https://github.com/hubin-keio/Spike_NLP/blob/new_manuscript/src/pnlp/runner/bert_mlm.py)): `python bert_mlm.py`
    * To run model runner initialized with ESM weights ([link](https://github.com/hubin-keio/Spike_NLP/blob/new_manuscript/src/pnlp/runner/bert_mlm-esm_init.py)): `python bert_mlm-esm_init.py`

### Available Models
* I will update this with model details, still making models (WIP, 11/12/24)

## Citation
