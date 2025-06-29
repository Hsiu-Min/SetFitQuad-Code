import logging
import json
import time  # 紀錄實驗時間
from pathlib import Path
from datetime import datetime
from typing import Dict, Any
from datasets import load_dataset
from setfit import SetFitModel, AbsaModel, Trainer, AbsaTrainer
from setfit import TrainingArguments
from sentence_transformers import SentenceTransformer
import spacy
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from scipy.optimize import linear_sum_assignment
import os
import torch
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# -----------------------
# 使用多語系 mpnet 模型
# -----------------------
MODEL_NAME = "paraphrase-multilingual-mpnet-base-v2"

# -----------------------
# 主要參數設定
# -----------------------
OUTPUT_DIR = "results"
TRAIN_SIZE = 50  # RF sampling
RANDOM_SEED = 42
BASE_PROP = 0.3  # base_prop for RF sampling

Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

# 採樣策略函數定義
def random_seed_sampling(embeddings, num_samples):
    np.random.seed(42)
    return np.random.choice(len(embeddings), num_samples, replace=False)

def grid_sampling(embeddings, num_samples):
    return np.linspace(0, len(embeddings) - 1, num_samples, dtype=int)

def max_min_distance_sampling(embeddings, num_samples):
    center = np.mean(embeddings, axis=0)
    distances = np.linalg.norm(embeddings - center, axis=1)
    selected = [np.argmax(distances)]
    for _ in range(1, num_samples):
        dist_to_selected = np.min(
            np.linalg.norm(embeddings[:, np.newaxis] - embeddings[selected], axis=2),
            axis=1
        )
        next_point = np.argmax(dist_to_selected)
        selected.append(next_point)
    return np.array(selected)

def density_based_sampling(embeddings, num_samples):
    nn = NearestNeighbors(n_neighbors=min(5, len(embeddings) - 1)).fit(embeddings)
    distances, _ = nn.kneighbors(embeddings)
    density_scores = np.sum(distances, axis=1)
    return np.argsort(density_scores)[:num_samples]

def max_entropy_sampling(embeddings, num_samples):
    nn = NearestNeighbors(n_neighbors=min(5, len(embeddings)-1))
    nn.fit(embeddings)
    distances, _ = nn.kneighbors(embeddings)
    distance_probs = np.exp(-distances) / np.sum(np.exp(-distances), axis=1, keepdims=True)
    entropies = -np.sum(distance_probs * np.log(distance_probs + 1e-10), axis=1)
    return np.argsort(entropies)[-num_samples:]

def cluster_sampling(embeddings, num_samples):
    kmeans = KMeans(n_clusters=num_samples, random_state=42).fit(embeddings)
    return np.array([np.where(kmeans.labels_ == i)[0][0] for i in range(num_samples)])

def random_forest_sampling(embeddings, num_samples):
    """
    RF策略的自動抽樣，並回傳抽樣索引與各策略的比例。
    """
    n = embeddings.shape[0]
    sampling_funcs = {
        "Random Seed": random_seed_sampling,
        "Grid": grid_sampling,
        "Max-Min Distance": max_min_distance_sampling,
        "Density-based": density_based_sampling,
        "Max Entropy": max_entropy_sampling,
        "Cluster": cluster_sampling,
    }

    base_prop = 0.3  # 使用0.3作為最佳比例
    base_indices = {
        name: func(embeddings, int(n * base_prop))
        for name, func in sampling_funcs.items()
    }

    # 建立特徵矩陣
    strategy_features = np.zeros((n, len(base_indices) * 4))
    for i, (_, indices) in enumerate(base_indices.items()):
        selected_embeddings = embeddings[indices]
        distances = np.linalg.norm(embeddings[:, np.newaxis] - selected_embeddings, axis=2)

        strategy_features[:, i * 4] = np.min(distances, axis=1)
        strategy_features[:, i * 4 + 1] = np.mean(distances, axis=1)
        strategy_features[:, i * 4 + 2] = np.std(distances, axis=1)

        probs = distances / (np.sum(distances, axis=1, keepdims=True) + 1e-10)
        strategy_features[:, i * 4 + 3] = -np.sum(probs * np.log(probs + 1e-10), axis=1)

    # 特徵標準化
    scaler = StandardScaler()
    strategy_features = scaler.fit_transform(strategy_features)

    # 計算目標值
    center = np.mean(embeddings, axis=0)
    distances_to_center = np.linalg.norm(embeddings - center, axis=1)

    nn = NearestNeighbors(n_neighbors=min(5, len(embeddings)))
    nn.fit(embeddings)
    distances, _ = nn.kneighbors(embeddings)

    density_scores = np.exp(-np.sum(distances, axis=1))
    diversity_scores = np.var(embeddings, axis=1)
    entropy_scores = -np.sum(distances * np.log(distances + 1e-10), axis=1)

    target = (distances_to_center * diversity_scores * (entropy_scores ** 2)) / (density_scores ** 0.5)

    # Random Forest regression
    rf = RandomForestRegressor(n_estimators=100, max_depth=10, min_samples_split=2, random_state=42)
    rf.fit(strategy_features, target)
    feature_importances = rf.feature_importances_

    # 計算每個策略的權重
    strategy_names = list(base_indices.keys())
    strategy_weights = np.zeros(len(strategy_names))
    for i in range(len(strategy_names)):
        strategy_weights[i] = np.mean(feature_importances[i * 4 : (i + 1) * 4])

    # MES 增強
    mes_idx = strategy_names.index("Max Entropy")
    max_weight = np.max(strategy_weights)
    min_mes_weight = max_weight * 0.8
    if strategy_weights[mes_idx] < min_mes_weight:
        strategy_weights[mes_idx] = min_mes_weight

    # 動態閾值過濾
    strategy_weights /= np.sum(strategy_weights)
    importance_threshold = np.percentile(strategy_weights, 25)
    strategy_weights[strategy_weights < importance_threshold] = 0.0

    # 重新歸一化
    total_w = np.sum(strategy_weights)
    if total_w > 0:
        strategy_weights /= total_w
    else:
        strategy_weights = np.ones(len(strategy_names)) / len(strategy_names)

    # 分配樣本數
    min_samples_per_strategy = max(1, int(num_samples * 0.1))
    strategy_samples = {s: min_samples_per_strategy for s, w in zip(strategy_names, strategy_weights) if w > 0}

    remaining = num_samples - sum(strategy_samples.values())
    if remaining > 0:
        valid_strategies = list(strategy_samples.keys())
        valid_weights = np.array([strategy_weights[strategy_names.index(s)] for s in valid_strategies])
        valid_weights /= np.sum(valid_weights)

        for i in range(len(valid_strategies) - 1):
            additional = int(remaining * valid_weights[i])
            strategy_samples[valid_strategies[i]] += additional

        # 分配剩餘
        strategy_samples[valid_strategies[-1]] += num_samples - sum(strategy_samples.values())

    # 收集樣本
    all_indices = []
    proportions = {}
    for strategy, n_samples in strategy_samples.items():
        if n_samples > 0:
            indices = sampling_funcs[strategy](embeddings, n_samples)
            all_indices.extend(indices)
            proportions[strategy] = n_samples / num_samples

    # 去重、修正尺寸
    all_indices = np.unique(all_indices)
    if len(all_indices) > num_samples:
        selected_indices = np.random.choice(all_indices, num_samples, replace=False)
    else:
        remaining = num_samples - len(all_indices)
        if remaining > 0:
            additional_indices = random_seed_sampling(embeddings, remaining)
            selected_indices = np.concatenate([all_indices, additional_indices])
        else:
            selected_indices = all_indices

    return selected_indices, proportions

def lcs_length(x, y):
    m, n = len(x), len(y)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if x[i - 1] == y[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[m][n]

def compute_lcs_score(pred_term, ref_term):
    if not pred_term or not ref_term:
        return 0.0
    length = lcs_length(pred_term, ref_term)
    return length / max(len(pred_term), len(ref_term))

def compute_f1_score(predictions, references, task_key, use_lcs=False):
    n = len(predictions)
    total_precision, total_recall = 0.0, 0.0

    for pred, ref in zip(predictions, references):
        pred_terms = [d[task_key] for d in pred]
        ref_terms = [d[task_key] for d in ref]
        
        if len(pred_terms) == 0 and len(ref_terms) == 0:
            total_precision += 1.0
            total_recall += 1.0
            continue
            
        if len(pred_terms) == 0 or len(ref_terms) == 0:
            continue

        score_matrix = np.zeros((len(pred_terms), len(ref_terms)), dtype=np.float32)
        for i, pterm in enumerate(pred_terms):
            for j, rterm in enumerate(ref_terms):
                if use_lcs:
                    score_matrix[i][j] = compute_lcs_score(pterm, rterm)
                else:
                    score_matrix[i][j] = 1.0 if pterm == rterm else 0.0
                    
        cost = 1 - score_matrix
        row_ind, col_ind = linear_sum_assignment(cost)
        match_score = sum(score_matrix[row_ind[k], col_ind[k]] for k in range(len(row_ind)))

        precision = match_score / len(pred_terms)
        recall = match_score / len(ref_terms)
        
        total_precision += precision
        total_recall += recall

    avg_precision = total_precision / n
    avg_recall = total_recall / n
    
    if (avg_precision + avg_recall) > 0:
        final_f1 = 2 * avg_precision * avg_recall / (avg_precision + avg_recall)
    else:
        final_f1 = 0.0

    return {
        "precision": avg_precision,
        "recall": avg_recall,
        "f1": final_f1,
    }

class SingleRunExperiment:
    def __init__(self):
        self.model_name = MODEL_NAME
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.nlp = spacy.load("en_core_web_sm")
        
        # 加入 SentenceTransformer 並移到正確設備
        self.st_model = SentenceTransformer(self.model_name).to(self.device)
        
        # 初始化其他屬性
        self.model_output_dir = Path(OUTPUT_DIR)/self.model_name
        self.model_output_dir.mkdir(parents=True, exist_ok=True)
        self.results = {
            "model": self.model_name,
            "train_size": TRAIN_SIZE,
            "base_prop": BASE_PROP,
            "random_seed": RANDOM_SEED
        }
        self.models = {}
        self.sample_proportions = None  # 儲存採樣比例
        
        # 清理 GPU 記憶體
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def load_data(self):
        try:
            logging.info("Loading dataset: JaquanTW/fewshot-absaquad")
            dataset = load_dataset("JaquanTW/fewshot-absaquad")
            self.full_dataset = dataset["train"]
            self.val_dataset = dataset["validation"]
            self.test_dataset = dataset["test"]
            logging.info(f"Train={len(self.full_dataset)}, Val={len(self.val_dataset)}, Test={len(self.test_dataset)}")
        except Exception as e:
            logging.error(f"Error loading dataset: {str(e)}")
            raise

    def sample_data(self, data):
        """使用Random Forest策略採樣"""
        try:
            logging.info("Generating embeddings for RF sampling...")
            texts = [x["text"] for x in data]
            embeddings = self.st_model.encode(texts, batch_size=32, show_progress_bar=True)
            
            logging.info("Applying RF sampling strategy...")
            indices, proportions = random_forest_sampling(embeddings, TRAIN_SIZE)
            sampled_data = data.select(indices.tolist())
            self.sample_proportions = proportions
            
            logging.info(f"Sampled {len(sampled_data)} examples using RF strategy")
            for strategy, prop in proportions.items():
                logging.info(f"{strategy}: {prop:.4f}")
                
            return sampled_data, proportions
        except Exception as e:
            logging.error(f"Error in sampling: {str(e)}")
            raise

    def train_ac_model(self):
        logging.info("Training AC model...")
        train_args = TrainingArguments(
            output_dir=str(self.model_output_dir/"ac"),
            num_epochs=5,
            batch_size=8,
            body_learning_rate=2e-5,
            head_learning_rate=1e-3,
        )
        unique_ac_labels = list(set(self.train_dataset["ac"]))
        ac_model = SetFitModel.from_pretrained(self.model_name, labels=unique_ac_labels)

        trainer = Trainer(
            model=ac_model,
            args=train_args,
            train_dataset=self.train_dataset.map(lambda x: {"text": x["text"], "label": x["ac"]}),
            eval_dataset=self.val_dataset.map(lambda x: {"text": x["text"], "label": x["ac"]}),
        )
        trainer.train()
        self.models["ac"] = ac_model

    def train_label_model(self):
        logging.info("Training Polarity model...")
        train_args = TrainingArguments(
            output_dir=str(self.model_output_dir/"label"),
            num_epochs=5,
            batch_size=8,
            body_learning_rate=2e-5,
            head_learning_rate=1e-3,
        )
        unique_labels = list(set(self.train_dataset["label"]))
        label_model = SetFitModel.from_pretrained(self.model_name, labels=unique_labels)

        trainer = Trainer(
            model=label_model,
            args=train_args,
            train_dataset=self.train_dataset.map(lambda x: {"text": x["text"], "label": x["label"]}),
            eval_dataset=self.val_dataset.map(lambda x: {"text": x["text"], "label": x["label"]}),
        )
        trainer.train()
        self.models["label"] = label_model

    def train_absa_model(self):
        logging.info("Training ABSA model...")
        absa_model = AbsaModel.from_pretrained(
            model_id=self.model_name,
            spacy_model="en_core_web_sm",
        )
        train_args = TrainingArguments(
            output_dir=str(self.model_output_dir/"absa"),
            num_epochs=5,
            batch_size=8,
            body_learning_rate=2e-5,
            head_learning_rate=1e-3,
        )
        trainer = AbsaTrainer(
            model=absa_model,
            args=train_args,
            train_dataset=self.train_dataset.map(lambda x: {
                "text": x["text"],
                "aspect": x["span"],
                "label": x["label"]
            }),
            eval_dataset=self.val_dataset.map(lambda x: {
                "text": x["text"],
                "aspect": x["span"],
                "label": x["label"]
            }),
        )
        trainer.train()
        self.models["absa"] = absa_model

    def extract_opinion_terms(self, text, aspect):
        doc = self.nlp(text)
        aspect_token = None
        for token in doc:
            if token.text.lower() == aspect.lower():
                aspect_token = token
                break
        if not aspect_token:
            return []

        opinion_terms = []
        for token in doc:
            if (token.dep_ in {"amod", "advmod"} and token.head == aspect_token) or \
               (token.head == aspect_token and token.pos_ in {"ADJ", "ADV"}):
                opinion_terms.append(token.text)

        window_size = 3
        aspect_index = aspect_token.i
        for i in range(max(0, aspect_index - window_size), min(len(doc), aspect_index + window_size + 1)):
            token = doc[i]
            if token.pos_ in {"ADJ", "ADV"} and token.text not in opinion_terms:
                opinion_terms.append(token.text)

        return opinion_terms
    
    def compute_quad_exact_match(self, gold_data, pred_data):
        """
        exact => (ac,label,span,ot)全部相同 => score=1 else 0
        """
        n = len(gold_data)
        totP, totR = 0, 0
        for i in range(n):
            gold_t = gold_data[i]
            pred_t = pred_data[i]
            if len(gold_t)==0 and len(pred_t)==0:
                totP += 1
                totR += 1
                continue
            if len(gold_t)==0 or len(pred_t)==0:
                continue
                
            score_mat = np.zeros((len(gold_t), len(pred_t)), dtype=np.float32)
            for g_i, g4 in enumerate(gold_t):
                for p_i, p4 in enumerate(pred_t):
                    if g4[0]==p4[0] and g4[1]==p4[1] and g4[2]==p4[2] and g4[3]==p4[3]:
                        score_mat[g_i][p_i] = 1.0
                        
            cost = 1-score_mat
            row_ind, col_ind = linear_sum_assignment(cost)
            match_count = sum(score_mat[row_ind[k], col_ind[k]] for k in range(len(row_ind)))
            precision = match_count/len(pred_t)
            recall = match_count/len(gold_t)
            totP += precision
            totR += recall
                
        avgP = totP/n
        avgR = totR/n
        f1 = 2*avgP*avgR/(avgP+avgR) if (avgP+avgR)>0 else 0
        return avgP, avgR, f1

    def compute_quad_partial_match(self, gold_data, pred_data):
        """
        partial => ac,label exact(0or1), span,ot => lcs(0~1)
        quad_score = (ac_s + lb_s + span_lcs + ot_lcs)/4
        """
        n = len(gold_data)
        totP, totR = 0, 0
        for i in range(n):
            gold_t = gold_data[i]
            pred_t = pred_data[i]
            if len(gold_t)==0 and len(pred_t)==0:
                totP += 1
                totR += 1
                continue
            if len(gold_t)==0 or len(pred_t)==0:
                continue
                
            score_mat = np.zeros((len(gold_t), len(pred_t)), dtype=np.float32)
            for g_i,g4 in enumerate(gold_t):
                for p_i,p4 in enumerate(pred_t):
                    ac_score = 1 if (g4[0]==p4[0]) else 0
                    lb_score = 1 if (g4[1]==p4[1]) else 0
                    sp_lcs = compute_lcs_score(p4[2], g4[2])
                    ot_lcs = compute_lcs_score(p4[3], g4[3])
                    quad_s = (ac_score + lb_score + sp_lcs + ot_lcs)/4
                    score_mat[g_i][p_i] = quad_s
                        
            cost = 1-score_mat
            row_ind, col_ind = linear_sum_assignment(cost)
            sum_score = sum(score_mat[row_ind[k], col_ind[k]] for k in range(len(row_ind)))
            precision = sum_score/len(pred_t)
            recall = sum_score/len(gold_t)
            totP += precision
            totR += recall
                
        avgP = totP/n
        avgR = totR/n
        f1 = 2*avgP*avgR/(avgP+avgR) if (avgP+avgR)>0 else 0
        return avgP, avgR, f1

    def evaluate(self, dataset, desc="Validation"):
        logging.info(f"Evaluating on {desc} set...")
        
        # 初始化評估指標收集器
        y_true_ac, y_pred_ac = [], []
        y_true_label, y_pred_label = [], []
        y_true_span_list, y_pred_span_list = [], []
        y_true_ot_list, y_pred_ot_list = [], []
        gold_data, pred_data = [], []
        
        for example in dataset:
            text = example["text"]
            
            # AC prediction
            ac_pred = self.models["ac"].predict([text])[0]
            y_true_ac.append(example["ac"])
            y_pred_ac.append(ac_pred)
            
            # Polarity prediction
            pol_pred = self.models["label"].predict([text])[0]
            y_true_label.append(example["label"])
            y_pred_label.append(pol_pred)
            
            # ABSA predictions and opinion terms
            absa_preds = self.models["absa"].predict([text])[0]
            
            # Process predictions
            pred_spans = [{"span": term.get("span", "")} for term in absa_preds if isinstance(term, dict)]
            pred_ots = []
            for term in absa_preds:
                if isinstance(term, dict):
                    span = term.get("span", "").strip()
                    if span:
                        ot_terms = self.extract_opinion_terms(text, span)
                        ot_str = " ".join(ot_terms).strip()
                        if ot_str:
                            pred_ots.append({"opinion_term": ot_str})
            
            # Ground truth
            true_spans = [{"span": example["span"]}] if "span" in example else []
            true_ots = []
            if "ot" in example and example["ot"]:
                if isinstance(example["ot"], list):
                    true_ots = [{"opinion_term": " ".join(example["ot"]).strip()}]
                else:
                    true_ots = [{"opinion_term": example["ot"].strip()}]
            true_ots = [ot for ot in true_ots if ot["opinion_term"]]
            
            # Collect span and opinion terms
            y_true_span_list.append(true_spans)
            y_pred_span_list.append(pred_spans)
            y_true_ot_list.append(true_ots)
            y_pred_ot_list.append(pred_ots)
            
            # Prepare quad tuple data
            gold_ac = example["ac"]
            gold_lb = example["label"]
            gold_sp = example["span"] if "span" in example else ""
            gold_ot = true_ots[0]["opinion_term"] if true_ots else ""
            gold_4 = [(gold_ac, gold_lb, gold_sp, gold_ot)]
            
            pred_4 = []
            if absa_preds:
                for term in absa_preds:
                    sp = term.get("span", "")
                    ot_terms = self.extract_opinion_terms(text, sp)
                    ot_str = " ".join(ot_terms).strip()
                    pred_4.append((ac_pred, pol_pred, sp, ot_str))
            else:
                pred_4.append((ac_pred, pol_pred, "", ""))
            
            gold_data.append(gold_4)
            pred_data.append(pred_4)

        # Calculate all metrics
        metrics = self._calculate_metrics(
            y_true_ac, y_pred_ac,
            y_true_label, y_pred_label,
            y_true_span_list, y_pred_span_list,
            y_true_ot_list, y_pred_ot_list,
            gold_data, pred_data
        )
        
        # Add sampling information
        if self.sample_proportions:
            metrics["sampling_strategy"] = {
                "method": "Random Forest",
                "proportions": self.sample_proportions
            }

        logging.info(f"{desc} metrics: {json.dumps(metrics, indent=2)}")
        return metrics

    def _calculate_metrics(self, y_true_ac, y_pred_ac, y_true_label, y_pred_label,
                         y_true_span_list, y_pred_span_list, y_true_ot_list, y_pred_ot_list,
                         gold_data, pred_data):
        # AC metrics
        p_ac, r_ac, _, _ = precision_recall_fscore_support(
            y_true_ac, y_pred_ac, average="weighted", zero_division=0
        )
        f1_ac = 2 * p_ac * r_ac / (p_ac + r_ac) if (p_ac + r_ac) > 0 else 0.0

        # Polarity metrics
        p_label, r_label, _, _ = precision_recall_fscore_support(
            y_true_label, y_pred_label, average="weighted", zero_division=0
        )
        f1_label = 2 * p_label * r_label / (p_label + r_label) if (p_label + r_label) > 0 else 0.0

        # Detailed polarity metrics
        posneg_p, posneg_r, _, _ = precision_recall_fscore_support(
            y_true_label, y_pred_label,
            labels=["positive", "negative"],
            average=None, zero_division=0
        )

        # Span and opinion term metrics
        span_metrics = compute_f1_score(y_pred_span_list, y_true_span_list, "span", use_lcs=True)
        ot_metrics = compute_f1_score(y_pred_ot_list, y_true_ot_list, "opinion_term", use_lcs=True)

        # Quad tuple metrics
        ex_p, ex_r, ex_f = self.compute_quad_exact_match(gold_data, pred_data)
        pm_p, pm_r, pm_f = self.compute_quad_partial_match(gold_data, pred_data)

        # Combine all metrics
        return {
            "aspect_category": {
                "precision": float(p_ac),
                "recall": float(r_ac),
                "f1": float(f1_ac)
            },
            "polarity": {
                "precision": float(p_label),
                "recall": float(r_label),
                "f1": float(f1_label),
                "positive": {
                    "precision": float(posneg_p[0]),
                    "recall": float(posneg_r[0])
                },
                "negative": {
                    "precision": float(posneg_p[1]),
                    "recall": float(posneg_r[1])
                }
            },
            "span": span_metrics,
            "opinion_term": ot_metrics,
            "quad": {
                "exact_match": {
                    "precision": ex_p,
                    "recall": ex_r,
                    "f1": ex_f
                },
                "partial_match": {
                    "precision": pm_p,
                    "recall": pm_r,
                    "f1": pm_f
                }
            }
        }

    def run(self):
        try:
            start_time = time.time()
            
            # Load and sample data
            self.load_data()
            logging.info(f"Sampling {TRAIN_SIZE} examples from training set...")
            self.train_dataset, self.sample_proportions = self.sample_data(self.full_dataset)
            logging.info("Using official validation set")

            # Training
            self.train_ac_model()
            self.train_label_model()
            self.train_absa_model()

            # Evaluation
            val_metrics = self.evaluate(self.val_dataset, "Validation")
            self.results["validation_metrics"] = val_metrics

            test_metrics = self.evaluate(self.test_dataset, "Test")
            self.results["test_metrics"] = test_metrics

            # Record experiment time
            end_time = time.time()
            self.results["experiment_duration"] = end_time - start_time
            self.results["timestamp"] = datetime.now().isoformat()
            
            # Save results
            self._save_results()
            
            # Clean up
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        except Exception as e:
            logging.error(f"Experiment failed: {str(e)}")
            raise

    def _save_results(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = self.model_output_dir / f"results_{timestamp}.json"
        
        try:
            with open(results_file, "w", encoding="utf-8") as f:
                json.dump(self.results, f, ensure_ascii=False, indent=2)
            logging.info(f"Results saved to: {results_file}")
        except Exception as e:
            logging.error(f"Failed to save results: {str(e)}")
            raise

if __name__ == "__main__":
    try:
        # Set random seeds
        np.random.seed(RANDOM_SEED)
        torch.manual_seed(RANDOM_SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(RANDOM_SEED)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            
            device_name = torch.cuda.get_device_name(0)
            device_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            logging.info(f"Using GPU: {device_name} ({device_memory:.2f} GB)")
        else:
            logging.info("Using CPU")

        # Run experiment
        experiment = SingleRunExperiment()
        experiment.run()
        
    except Exception as e:
        logging.error(f"Experiment failed: {str(e)}")
        raise