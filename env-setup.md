# SetFitQuad Environment Setup

## Create and Activate Environment
```bash
conda create -n setfit python=3.9
conda activate setfit


------------
switch to setfit environment 
------------

## Install PyTorch
bash
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

## Install Required Packages
pip install setfit transformers datasets scikit-learn torchmetrics tqdm protobuf scipy spacy seaborn matplotlib GPUtil

## Install Spacy Models
set KMP_DUPLICATE_LIB_OK=TRUE
python -m spacy download en_core_web_lg
python -m spacy download en_core_web_sm
