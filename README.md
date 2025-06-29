# SetFitQuad
This repository provides the source code and setup instructions for our paper
**[SetFitQuad: A Few-Shot Framework for Aspect Sentiment Quad Prediction with Sampling Strategies]**.
*(Submitted to IEEE Access)*

## Task Summary
- SetFitQuad addresses the **Aspect Sentiment Quad Prediction (ASQP)** task.  
- Given a sentence, the goal is to extract sentiment quadruples of the form:  
  - **Aspect Term (AT)**  
  - **Aspect Category (AC)**  
  - **Opinion Term (OT)**  
  - **Sentiment Polarity (SP)**  

## Dataset
- We use the benchmark datasets **Rest15** and **Rest16**, available via Hugging Face:
-  [Hugging Face – fewshot-absaquad](https://huggingface.co/datasets/JaquanTW/fewshot-absaquad)

## Environment Setup & Requirements
```txt
setfit
transformers
datasets
scikit-learn
torchmetrics
tqdm
protobuf==3.20.*
scipy
spacy
seaborn
matplotlib
GPUtil
```

## Quick Start

- Follow `env-setup.md` to create and activate the SetFit environment.
- Install required dependencies:
  - **Create and activate the environment**  
    ```bash
    conda create -n setfit python=3.9
    conda activate setfit
    ```

------------
switch to setfit environment 
------------
  - **Install PyTorch**  
    ```bash
    conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
    ```
  - **Install required packages**  
    ```bash
    pip install setfit transformers datasets scikit-learn torchmetrics tqdm protobuf scipy spacy seaborn matplotlib GPUtil
    ```
  - **Install Spacy models**  
    ```bash
    set KMP_DUPLICATE_LIB_OK=TRUE
    python -m spacy download en_core_web_lg
    python -m spacy download en_core_web_sm
    ```

- Navigate to the experiment directory:
  ```bash
  cd ~/exp4
  python exp4_comparision_cs.py
