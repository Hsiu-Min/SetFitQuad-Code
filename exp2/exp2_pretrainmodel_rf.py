import logging
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any
from datasets import load_dataset
from setfit import SetFitModel, AbsaModel, Trainer, AbsaTrainer
from setfit import TrainingArguments
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import KFold
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

# Define models configuration
MODELS_CONFIG = {
    "all-MiniLM-L6-v2": {
        "path": "sentence-transformers/all-MiniLM-L6-v2",
        "type": "general",
        "description": "Fast and good quality general purpose model"
    },
    "paraphrase-TinyBERT-L6-v2": {
        "path": "sentence-transformers/paraphrase-TinyBERT-L6-v2",
        "type": "lightweight",
        "description": "Lightweight model optimized for paraphrase tasks"
    },
    "all-mpnet-base-v2": {
        "path": "sentence-transformers/all-mpnet-base-v2",
        "type": "general",
        "description": "Best quality general purpose model"
    },
    "multi-qa-mpnet-base-cos-v1": {
        "path": "sentence-transformers/multi-qa-mpnet-base-cos-v1",
        "type": "task-specific",
        "description": "Optimized for QA and semantic search"
    },
    "paraphrase-multilingual-mpnet-base-v2": {
        "path": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        "type": "multilingual",
        "description": "High quality multilingual model"
    }
}

OUTPUT_DIR = "results"
SAMPLE_SIZE = 50
N_FOLDS = 5
RANDOM_SEED = 42

def random_seed_sampling(embeddings, num_samples):
    np.random.seed(42)
    return np.random.choice(len(embeddings), num_samples, replace=False)

def max_entropy_sampling(embeddings, num_samples):
    nn = NearestNeighbors(n_neighbors=min(5, len(embeddings)-1))
    nn.fit(embeddings)
    distances, _ = nn.kneighbors(embeddings)
    distance_probs = np.exp(-distances) / np.sum(np.exp(-distances), axis=1, keepdims=True)
    entropies = -np.sum(distance_probs * np.log(distance_probs + 1e-10), axis=1)
    return np.argsort(entropies)[-num_samples:]

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

def cluster_sampling(embeddings, num_samples):
    kmeans = KMeans(n_clusters=num_samples, random_state=42).fit(embeddings)
    return np.array([np.where(kmeans.labels_ == i)[0][0] for i in range(num_samples)])

def random_forest_sampling(embeddings, num_samples):
    """
    綜合策略的自動抽樣，並回傳抽樣索引與各策略的比例。
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

    base_indices = {
        name: func(embeddings, int(n * 0.3))
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

    # 計算目標值（綜合空間分佈、密度、熵等等）
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

    # 平均每組策略 (4 維特徵)
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

    # 根據權重分配
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

    logging.info("Sampling proportions:")
    for strategy, prop in proportions.items():
        logging.info(f"{strategy}: {prop:.4f}")

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
    """
    使用匈牙利演算法計算多對多 partial match 的 F1，F1 直接使用 P 和 R 計算。
    """
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

        cost_matrix = 1.0 - score_matrix
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        total_match_score = sum(score_matrix[row_ind[i], col_ind[i]] for i in range(len(row_ind)))
        precision = total_match_score / len(pred_terms)
        recall = total_match_score / len(ref_terms)

        total_precision += precision
        total_recall += recall

    avg_precision = total_precision / n
    avg_recall = total_recall / n

    # 正確的 F1 計算公式
    if (avg_precision + avg_recall) > 0:
        final_f1 = 2 * avg_precision * avg_recall / (avg_precision + avg_recall)
    else:
        final_f1 = 0.0

    return {
        "precision": avg_precision,
        "recall": avg_recall,
        "f1": final_f1,
    }



class CheckpointManager:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.checkpoint_file = os.path.join(output_dir, "model_checkpoint.json")

    def save_checkpoint(self, model_name, current_fold):
        checkpoint = self.load_checkpoint() or {"completed_models": {}}
        if model_name in checkpoint["completed_models"]:
            completed_status = checkpoint["completed_models"][model_name].get("completed", False)
        else:
            completed_status = False

        checkpoint["completed_models"][model_name] = {
            "current_fold": current_fold,
            "completed": completed_status
        }
        with open(self.checkpoint_file, "w") as f:
            json.dump(checkpoint, f, indent=2)
        logging.info(f"Checkpoint saved: {self.checkpoint_file}")

    def load_checkpoint(self):
        if os.path.exists(self.checkpoint_file):
            with open(self.checkpoint_file, "r") as f:
                return json.load(f)
        return None

    def mark_completed(self, model_name):
        checkpoint = self.load_checkpoint() or {"completed_models": {}}
        checkpoint["completed_models"][model_name] = {
            "current_fold": N_FOLDS,
            "completed": True
        }
        with open(self.checkpoint_file, "w") as f:
            json.dump(checkpoint, f, indent=2)
        logging.info(f"Marked {model_name} as completed.")

class EnhancedExperiment:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model_config = MODELS_CONFIG[model_name]
        self.model_path = self.model_config["path"]
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.models = {}
        self.nlp = spacy.load("en_core_web_sm")
        self.fold_metrics = []

        self.results = {
            "model_info": self.model_config,
            "metrics": {},
            "timestamp": datetime.now().isoformat(),
            "training_params": {
                "train_size": SAMPLE_SIZE,
                "n_folds": N_FOLDS
            }
        }

        self.model_output_dir = Path(OUTPUT_DIR) / model_name
        self.model_output_dir.mkdir(parents=True, exist_ok=True)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            logging.info(f"Using GPU: {gpu_name} with {gpu_memory:.2f}GB memory")
        else:
            logging.info("Using CPU for training")

        self.st_model = SentenceTransformer(self.model_path).to(self.device)

    def load_data(self):
        logging.info(f"\n{'='*50}")
        logging.info(f"Starting experiment with model: {self.model_name}")
        logging.info(f"{'='*50}")
        dataset = load_dataset("JaquanTW/fewshot-absaquad")
        self.full_dataset = dataset["train"]
        self.test_dataset = dataset["test"]
        logging.info(f"Full training dataset loaded: {len(self.full_dataset)} samples")
        logging.info(f"Test dataset loaded: {len(self.test_dataset)} samples")

    def sample_data(self, data, size):
        if len(data) <= size:
            logging.info(f"Data size {len(data)} is smaller than required size {size}, using full dataset")
            return data, {"full_dataset": 1.0}

        logging.info("Generating embeddings for sampling...")
        texts = [x["text"] for x in data]
        embeddings = self.st_model.encode(texts, batch_size=32, show_progress_bar=True)

        sampled_indices, proportions = random_forest_sampling(embeddings, size)
        sampled_data = data.select(sampled_indices.tolist())

        logging.info(f"Sampled size: {len(sampled_data)} (from {len(data)})")
        logging.info("Sampling proportions:")
        for strategy, prop in proportions.items():
            logging.info(f"{strategy}: {prop:.4f}")

        return sampled_data, proportions

    def process_fold(self, train_fold, val_fold, fold_num):
        logging.info(f"\n{'='*50}")
        logging.info(f"Processing fold {fold_num + 1}/{N_FOLDS}")
        logging.info(f"{'='*50}")
        logging.info(f"Original train fold size: {len(train_fold)} samples")
        logging.info(f"Original validation fold size: {len(val_fold)} samples")

        train_sampled, proportions = self.sample_data(train_fold, SAMPLE_SIZE)
        self.sample_proportions = proportions

        logging.info(f"\nSampling completed:")
        logging.info(f"Training samples selected: {len(train_sampled)}/{len(train_fold)}")
        logging.info("Strategy proportions:")
        for strategy, prop in proportions.items():
            logging.info(f"  {strategy}: {prop:.4f}")

        logging.info(f"\nFinal dataset sizes:")
        logging.info(f"Training set: {len(train_sampled)} samples")
        logging.info(f"Validation set: {len(val_fold)} samples")

        return train_sampled, val_fold

    def train_ac_model(self):
        logging.info("Training AC model...")
        train_args = TrainingArguments(
            output_dir=str(self.model_output_dir / "ac"),
            num_epochs=5,
            batch_size=8,
            body_learning_rate=2e-5,
            head_learning_rate=1e-3,
        )
        unique_ac_labels = list(set(self.train_dataset["ac"]))
        ac_model = SetFitModel.from_pretrained(self.model_path, labels=unique_ac_labels)

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
            output_dir=str(self.model_output_dir / "label"),
            num_epochs=5,
            batch_size=8,
            body_learning_rate=2e-5,
            head_learning_rate=1e-3,
        )
        unique_labels = list(set(self.train_dataset["label"]))
        label_model = SetFitModel.from_pretrained(self.model_path, labels=unique_labels)

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
            model_id=self.model_path,
            spacy_model="en_core_web_sm",
        )
        train_args = TrainingArguments(
            output_dir=str(self.model_output_dir / "absa"),
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
        """簡易意見術語擴充"""
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

        # 簡單 window 搜索
        window_size = 3
        aspect_index = aspect_token.i
        for i in range(max(0, aspect_index - window_size), min(len(doc), aspect_index + window_size + 1)):
            token = doc[i]
            if token.pos_ in {"ADJ", "ADV"} and token.text not in opinion_terms:
                opinion_terms.append(token.text)

        return opinion_terms

    def evaluate(self, fold_num=None):
        """Evaluate models and return results."""
        logging.info(f"Evaluating models for fold {fold_num + 1 if fold_num is not None else 'final'}...")

        # 初始化：收集 AC / Polarity 的標籤與預測
        y_true_ac, y_pred_ac = [], []
        y_true_label, y_pred_label = [], []

        # 收集 Span / OT（多對多部分匹配）
        all_y_true_spans, all_y_pred_spans = [], []
        all_y_true_ots,   all_y_pred_ots   = [], []

        # 驗證集或測試集
        eval_dataset = self.val_dataset if fold_num is not None else self.test_dataset

        for example in eval_dataset:
            text = example["text"]

            # 1. AC / Polarity，用套件拿到 p、r，但等一下我們自己覆蓋 f1
            ac_prediction = self.models["ac"].predict([text])[0]
            polarity_prediction = self.models["label"].predict([text])[0]
            y_true_ac.append(example["ac"])
            y_pred_ac.append(ac_prediction)
            y_true_label.append(example["label"])
            y_pred_label.append(polarity_prediction)

            # 2. ABSA 預測 (Span, OT)
            absa_predictions = self.models["absa"].predict([text])[0]
            logging.info(f"absa_predictions: {absa_predictions}")

            if isinstance(absa_predictions, list):
                # 預測到的 aspect span
                y_pred_spans = [
                    {"span": term.get("span", "")}
                    for term in absa_predictions if isinstance(term, dict)
                ]
                # 預測到的 opinion terms (用 extract_opinion_terms)
                y_pred_ots = [
                    {"opinion_term": " ".join(self.extract_opinion_terms(text, term.get("span", ""))).strip()}
                    for term in absa_predictions if isinstance(term, dict) and term.get("span", "").strip()
                ]
            else:
                y_pred_spans = []
                y_pred_ots   = []

            # 過濾空字串
            y_pred_ots = [ot for ot in y_pred_ots if ot["opinion_term"].strip()]

            # 3. Ground Truth: Span / OT
            y_true_spans = [{"span": example["span"]}] if "span" in example else []
            if "ot" in example and example["ot"]:
                if isinstance(example["ot"], list):
                    y_true_ots = [{"opinion_term": " ".join(example["ot"]).strip()}]
                else:
                    y_true_ots = [{"opinion_term": example["ot"].strip()}]
            else:
                y_true_ots = []
            y_true_ots = [ot for ot in y_true_ots if ot["opinion_term"].strip()]

            all_y_pred_spans.append(y_pred_spans)
            all_y_true_spans.append(y_true_spans)
            all_y_pred_ots.append(y_pred_ots)
            all_y_true_ots.append(y_true_ots)

        # === (A) 計算 AC / Polarity: 用 scikit-learn 取得 p,r，但 f1 自行套公式
        from sklearn.metrics import precision_recall_fscore_support

        p_ac, r_ac, _, _ = precision_recall_fscore_support(
            y_true_ac, y_pred_ac, average="weighted", zero_division=0
        )
        if (p_ac + r_ac) > 0:
            f1_ac = 2 * p_ac * r_ac / (p_ac + r_ac)
        else:
            f1_ac = 0.0

        p_label, r_label, _, _ = precision_recall_fscore_support(
            y_true_label, y_pred_label, average="weighted", zero_division=0
        )
        if (p_label + r_label) > 0:
            f1_label = 2 * p_label * r_label / (p_label + r_label)
        else:
            f1_label = 0.0

        # === (B) Span / OT 用 compute_f1_score (匈牙利+LCS)，內部也用 (2PR)/(P+R)
        metrics_span = compute_f1_score(all_y_pred_spans, all_y_true_spans, "span", use_lcs=True)
        metrics_ot   = compute_f1_score(all_y_pred_ots,   all_y_true_ots,   "opinion_term", use_lcs=True)

        # === 最終組裝
        metrics = {
            "AC Metrics": {
                "Precision": p_ac,
                "Recall":    r_ac,
                "F1":        f1_ac,
            },
            "Polarity Metrics": {
                "Precision": p_label,
                "Recall":    r_label,
                "F1":        f1_label,
            },
            "Span Metrics": metrics_span,
            "Opinion Term Metrics": metrics_ot,
        }

        if fold_num is not None:
            metrics["Fold"] = fold_num + 1

        metrics["Sample Proportions"] = self.sample_proportions

        logging.info(f"AC Metrics: {metrics['AC Metrics']}")
        logging.info(f"Polarity Metrics: {metrics['Polarity Metrics']}")
        logging.info(f"Span Metrics: {metrics['Span Metrics']}")
        logging.info(f"Opinion Term Metrics: {metrics['Opinion Term Metrics']}")

        return metrics


    def run(self):
        checkpoint_manager = CheckpointManager(OUTPUT_DIR)
        checkpoint = checkpoint_manager.load_checkpoint()

        # 檢查是否已完成
        if (checkpoint and "completed_models" in checkpoint and
            self.model_name in checkpoint["completed_models"] and
            checkpoint["completed_models"][self.model_name].get("completed", False)):
            logging.info(f"{self.model_name} 已完成，直接跳過。")
            return

        logging.info(f"\n{'='*50}")
        logging.info(f"Starting experiment with model: {self.model_name}")
        logging.info(f"{'='*50}")
        self.load_data()

        kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
        dataset_indices = list(range(len(self.full_dataset)))

        start_fold = 0
        if checkpoint and "completed_models" in checkpoint and self.model_name in checkpoint["completed_models"]:
            start_fold = checkpoint["completed_models"][self.model_name]["current_fold"]
            logging.info(f"\nResuming {self.model_name} from fold {start_fold + 1}")

        for fold_num, (train_idx, val_idx) in enumerate(kf.split(dataset_indices)):
            if fold_num < start_fold:
                continue

            logging.info(f"\n{'='*50}")
            logging.info(f"Processing fold {fold_num + 1}/{N_FOLDS} for {self.model_name}")
            logging.info(f"{'='*50}")

            checkpoint_manager.save_checkpoint(self.model_name, fold_num)

            train_fold = self.full_dataset.select(train_idx.tolist())
            val_fold = self.full_dataset.select(val_idx.tolist())
            logging.info(f"Original fold sizes - Train: {len(train_fold)}, Val: {len(val_fold)}")

            self.train_dataset, self.val_dataset = self.process_fold(train_fold, val_fold, fold_num)

            try:
                logging.info("\n==============================")
                logging.info("Starting training pipeline")
                logging.info("==============================")

                logging.info("\nTraining AC model...")
                self.train_ac_model()
                logging.info("AC model training completed")

                logging.info("\nTraining Polarity model...")
                self.train_label_model()
                logging.info("Polarity model training completed")

                logging.info("\nTraining ABSA model...")
                self.train_absa_model()
                logging.info("ABSA model training completed")

                logging.info("\nEvaluating models...")
                fold_metrics = self.evaluate(fold_num)

                fold_results = {
                    "fold_metrics": fold_metrics,
                    "sampling_proportions": self.sample_proportions
                }

                results_file = os.path.join(OUTPUT_DIR, f"{self.model_name}_fold{fold_num + 1}_results.json")
                with open(results_file, 'w', encoding='utf-8') as f:
                    json.dump(fold_results, f, ensure_ascii=False, indent=2)

                logging.info("Evaluation completed and results saved")
            except Exception as e:
                logging.error(f"Error in fold {fold_num + 1} for {self.model_name}: {str(e)}")
                raise e

            logging.info(f"\nCompleted fold {fold_num + 1}/{N_FOLDS}")

        # 完成所有 fold 後直接標記模型完成，不做平均
        checkpoint_manager.mark_completed(self.model_name)
        logging.info(f"\nExperiment completed for {self.model_name}")
        logging.info(f"{'='*50}")

def run_all_experiments():
    checkpoint_manager = CheckpointManager(OUTPUT_DIR)
    checkpoint = checkpoint_manager.load_checkpoint()

    if checkpoint and "completed_models" in checkpoint:
        logging.info("Loaded checkpoint with completed model statuses.")
    else:
        logging.info("No checkpoint found. Starting fresh.")
        checkpoint = {"completed_models": {}}

    all_results = {}
    for model_name in MODELS_CONFIG.keys():
        completed = checkpoint["completed_models"].get(model_name, {}).get("completed", False)
        current_fold = checkpoint["completed_models"].get(model_name, {}).get("current_fold", 0)
        logging.info(f"Checking status for {model_name}: completed={completed}, current_fold={current_fold}")

        if completed:
            logging.info(f"Skipping {model_name}: already completed.")
            continue

        logging.info(f"Starting experiments with {model_name}")
        try:
            experiment = EnhancedExperiment(model_name)
            experiment.run()
            all_results[model_name] = experiment.results
            checkpoint_manager.mark_completed(model_name)
        except Exception as e:
            logging.error(f"Error with {model_name}: {e}")
            continue

    # 不再計算或存任何平均結果，直接存各模型結果
    comparative_results_file = Path(OUTPUT_DIR) / "comparative_results.json"
    with open(comparative_results_file, 'w', encoding='utf-8') as f:
        json.dump({"model_results": all_results}, f, ensure_ascii=False, indent=2)
    logging.info(f"Comparative results saved to {comparative_results_file}")

def generate_summary_report():
    comparative_file = Path(OUTPUT_DIR) / "comparative_results.json"
    if not comparative_file.exists():
        logging.error("No comparative results found. Please run experiments first.")
        return
    
    with open(comparative_file, 'r', encoding='utf-8') as f:
        results = json.load(f)

    from datetime import datetime
    report = {
        "timestamp": datetime.now().isoformat(),
        "message": "Summary not implemented yet"
    }
    
    report_file = Path(OUTPUT_DIR) / "summary_report.json"
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    logging.info(f"\nSummary report saved to {report_file}")

if __name__ == "__main__":
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    logging.info("Starting all experiments...")

    checkpoint_manager = CheckpointManager(OUTPUT_DIR)
    checkpoint = checkpoint_manager.load_checkpoint()

    if checkpoint and "completed_models" in checkpoint:
        uncompleted_models = [
            model
            for model, status in checkpoint["completed_models"].items()
            if not status.get("completed", False)
        ]
        if uncompleted_models:
            start_model = uncompleted_models[0]
        else:
            logging.info("All models completed.")
            exit()
    else:
        start_model = list(MODELS_CONFIG.keys())[0]

    for model_name in MODELS_CONFIG.keys():
        if model_name < start_model:
            continue
        logging.info(f"Starting experiment with model: {model_name}")
        experiment = EnhancedExperiment(model_name)
        experiment.run()
        torch.cuda.empty_cache()
