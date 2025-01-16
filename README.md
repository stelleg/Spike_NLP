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
To match the packages found in our conda environment, spike_env, you can run `conda env create -f environment.yml`. After activating the conda environment, the pnlp module can be installed using `pip install -e .`. The pnlp module is necessary for running the models using the model runners. Additionally, we have a `requirements.txt` available to use for venv if you do not want to use conda.  

Other requirements:
- NVIDIA GPU

## Documentation
We offer more in depth documentation located in the [notebooks](https://github.com/hubin-keio/Spike_NLP/tree/new_manuscript/notebooks) folder, which we recommend reading for further understanding before usage of the models. 

* Clustering notebooks
  - HDBSCAN clustering for betacov
  - HDBSCAN clustering for RBD
* Data processing notebooks
  - AlphaSeq
  - Betacov
  - DMS
  - RBD
* Model notebooks
  - Model results for bert_mlm models
  - Model results for finetuning/transfer learning models
  - Development notes for the bert_mlm models
  - DMS embedding notes for the pre-embedded models 
  - Measured vs predicted comparisons

### Quickstart: Running the NLP BERT Model
Executing
  * There are 2 models for the NLP BERT model located in the `src/pnlp/runner` folder.
    * To run model runner ([link](https://github.com/hubin-keio/Spike_NLP/blob/new_manuscript/src/pnlp/runner/bert_mlm.py)): `python bert_mlm.py`
    * To run model runner initialized with ESM weights ([link](https://github.com/hubin-keio/Spike_NLP/blob/new_manuscript/src/pnlp/runner/bert_mlm-esm_init.py)): `python bert_mlm-esm_init.py`

### Other Available Models
Model runners can be found [here](https://github.com/hubin-keio/Spike_NLP/tree/new_manuscript/src/pnlp/runner). We utilize these models for transfer learning of models using the DMS data. Models are ran similarly to those above, `python <model_runner_file>`.
* Binding & Expression (BE) models
  - Pre-embedded
    - gcn_BE-ESM AA embedded
    - gcn_BE-NLP embedded initialized w/ bert_mlm-esm_init
    - blstm_BE-ESM AA embedded
    - blstm_BE-NLP embedded initialized w/ bert_mlm-esm_init
    - fcn_BE-ESM CLS embedded
  - Non Pre-embedded
    - bert_BE initialized w/ bert_mlm-esm_init
    - bert-blstm_BE initialized w/ bert_mlm-esm_init
    - bert-gcn_BE initialized w/ bert_mlm-esm_init
    - esm-blstm_BE initialized w/ bert_mlm-esm_init
    - esm-blstm_BE 
    - esm-fcn_BE
    - esm-gcn_BE
* Binding or Expression models
  - Pre-embedded
    - gcn-ESM AA embedded
    - gcn-NLP embedded initialized w/ bert_mlm-esm_init
    - blstm-ESM AA embedded
    - blstm-NLP embedded initialized w/ bert_mlm-esm_init
    - fcn-ESM CLS embedded
  - Non Pre-embedded
    - bert-blstm initialized w/ bert_mlm-esm_init
    - bert-gcn initialized w/ bert_mlm-esm_init
    - esm-blstm initialized w/ bert_mlm-esm_init
    - esm-blstm 
    - esm-fcn
    - esm-gcn

## Citation
