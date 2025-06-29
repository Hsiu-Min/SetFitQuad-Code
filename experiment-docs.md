# Experiment File Descriptions

## Baseline Comparisons
This study compares two main approaches:

1. **ASQP (Zhang et al., 2021)**
   - A generative model based on the T5 architecture.
   - F1 scores on the quadruple prediction task: 0.188 (Exact Match), 0.590 (Partial Match)

2. **SetFitABSA (Lai et al., 2023)**
   - A few-shot learning framework based on SetFit.
   - F1 score on the sentiment classification task: 0.774

## Experiment 1: Evaluation of Sample Selection Strategies (`exp1_samplestrategy.py`)
Evaluates different sample selection strategies, including:
- **Basic Strategies**: Random Sampling, Grid Sampling, Max-Min Distance, Density-based, Maximum Entropy, Cluster Sampling
- **Combined Strategies**: Lasso, Ridge, ElasticNet, Random Forest, Equal Proportion

## Experiment 2: Pre-trained Model Comparison
Evaluates pre-trained models using different sampling strategies:
- Maximum Entropy Sampling (`exp2_pretrainmodel_mes.py`)
- Cluster Sampling (`exp2_pretrainmodel_cs.py`) 
- Random Forest (`exp2_pretrainmodel_rf.py`)

Pre-trained models evaluated:
- all-MiniLM-L6-v2
- paraphrase-TinyBERT-L6-v2  
- all-mpnet-base-v2
- multi-qa-mpnet-base-cos-v1
- paraphrase-multilingual-mpnet-base-v2

## Experiment 3: Few-shot Learning Curve Analysis (`exp3_samplesize.py`)
Analyzes the impact of training sample sizes (1-200) on model performance using:
- **Model**: paraphrase-multilingual-mpnet-base-v2
- **Sampling Strategy**: Cluster Sampling
- **Evaluation**: Multiple cross-validation

## Experiment 4: Comprehensive Comparison (`exp4_comparision.py`)
Compares the performance of SetFitQuad with existing methods:
- **Pre-trained Model**: multilingual-mpnet
- **Sampling Strategy**: Cluster Sampling
- **Training Samples**: 50 instances
- **Evaluation Metrics**: Exact Match and Partial Match
