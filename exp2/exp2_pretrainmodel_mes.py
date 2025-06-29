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
from sklearn.neighbors import NearestNeighbors 
from datasets import load_dataset, Dataset  # 加入 Dataset


logging.basicConfig(level=logging.INFO)

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

def lcs_length(x, y):
    """計算字串 x, y 的最長共同子序列 (LCS) 長度。"""
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
    """計算 pred_term 與 ref_term 的 LCS 部分匹配分數。"""
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
        logging.debug(f"Before save: {json.dumps(checkpoint, indent=2)}")  # Debug檢查
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
        logging.debug(f"After save: {json.dumps(checkpoint, indent=2)}")  # Debug檢查

    def load_checkpoint(self):
        if os.path.exists(self.checkpoint_file):
            with open(self.checkpoint_file, "r") as f:
                return json.load(f)
        return None

    def mark_completed(self, model_name):
        checkpoint = self.load_checkpoint() or {"completed_models": {}}
        logging.info(f"Before marking {model_name} completed: {json.dumps(checkpoint, indent=2)}")
        
        checkpoint["completed_models"][model_name] = {
            "current_fold": N_FOLDS,
            "completed": True
        }

        with open(self.checkpoint_file, "w") as f:
            json.dump(checkpoint, f, indent=2)

        logging.info(f"Marked {model_name} as completed. Updated checkpoint: {json.dumps(checkpoint, indent=2)}")

class EnhancedExperiment:
    def __init__(self, model_name: str):
        self.model_config = MODELS_CONFIG[model_name]
        self.model_name = model_name
        self.model_path = self.model_config["path"]
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.models = {}
        self.nlp = spacy.load("en_core_web_sm")
        self.fold_metrics = []
        
        # Result dict
        self.results = {
            "model_info": self.model_config,
            "metrics": {},
            "timestamp": datetime.now().isoformat(),
            "training_params": {
                "train_size": SAMPLE_SIZE,
                "n_folds": N_FOLDS
            }
        }
        
        # 輸出目錄
        self.model_output_dir = Path(OUTPUT_DIR) / model_name
        self.model_output_dir.mkdir(parents=True, exist_ok=True)
        
        # GPU/CPU 設置
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            logging.info(f"Using GPU: {gpu_name} with {gpu_memory:.2f}GB memory")
        else:
            logging.info("Using CPU for training")
        
        # 初始化 SentenceTransformer
        self.st_model = SentenceTransformer(self.model_path)
        self.st_model = self.st_model.to(self.device)

    def load_data(self):
        """加載數據集"""
        logging.info(f"Loading datasets for model: {self.model_name}...")
        dataset = load_dataset("JaquanTW/fewshot-absaquad")
        self.full_dataset = dataset["train"]
        self.test_dataset = dataset["test"]
        logging.info(f"Full dataset size: {len(self.full_dataset)}")
        logging.info(f"Test dataset size: {len(self.test_dataset)}")

    # 以下是舊的 cluster_sampling，可以保留或刪除(但我們這裡保留以保留程式長度)
    def cluster_sampling(self, embeddings, num_samples):
        """
        Cluster Sampling (CS) - 原本程式碼中的方法，
        這裡只是保留函式，但在 sample_data 不再使用它。
        """
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=num_samples, random_state=RANDOM_SEED).fit(embeddings)
        
        cluster_indices = []
        for c in range(num_samples):
            members = np.where(kmeans.labels_ == c)[0]
            if len(members) > 0:
                cluster_indices.append(members[0])
                
        if len(cluster_indices) < num_samples:
            needed = num_samples - len(cluster_indices)
            all_indices = set(range(len(embeddings)))
            used = set(cluster_indices)
            remain_indices = list(all_indices - used)
            np.random.shuffle(remain_indices)
            cluster_indices.extend(remain_indices[:needed])
        
        return np.array(cluster_indices)

    def max_entropy_sampling(self, embeddings, num_samples):
        """最大熵抽樣（Max Entropy Sampling, MES）"""
        # 確保數據量足夠
        if len(embeddings) < 2:
            return np.array(range(len(embeddings)))
            
        # 動態調整鄰居數
        n_neighbors = min(5, len(embeddings) - 1)
        n_neighbors = max(1, n_neighbors)  # 確保至少有1個鄰居
        
        nn = NearestNeighbors(n_neighbors=n_neighbors)
        nn.fit(embeddings)
        distances, _ = nn.kneighbors(embeddings)
        
        # 加入更安全的 epsilon
        epsilon = np.finfo(float).eps
        distances = distances + epsilon
        
        entropies = -np.sum(distances * np.log(distances), axis=1)
        return np.argsort(entropies)[-num_samples:]

    def max_entropy_sampling_with_proportions(self, embeddings, num_samples):
        """最大熵抽樣 + 產生 proportions"""
        indices = self.max_entropy_sampling(embeddings, num_samples)
        proportions = {"Max Entropy (MES)": 1.0}
        return indices, proportions

    # def sample_data(self, data, size):
    #     """
    #     只使用最大熵抽樣 (MES)，
    #     之前 cluster_sampling 的呼叫已改為 MES。
    #     """
    #     if len(data) <= size:
    #         logging.info(f"Data size {len(data)} is smaller than required size {size}, using full dataset")
    #         return data, {"full_dataset": 1.0}

    #     logging.info(f"Generating embeddings for MES sampling...")
    #     texts = [x["text"] for x in data]
    #     embeddings = self.st_model.encode(texts, batch_size=32, show_progress_bar=True)

    #     # 直接改成只做 MES
    #     sampled_indices, proportions = self.max_entropy_sampling_with_proportions(embeddings, size)

    #     sampled_data = data.select(sampled_indices)
    #     logging.info(f"Sampled size: {len(sampled_data)} (from {len(data)}) using MES")
    #     return sampled_data, proportions
    
    def sample_fold_data(self, fold_data):
        """對每個 fold 的訓練數據進行採樣"""
        logging.info("Applying MES sampling strategy...")
        
        # 確保每個類別至少有 min_samples_per_class 個樣本
        min_samples_per_class = 2
        from collections import Counter
        label_counts = Counter(fold_data["label"])
        
        texts = [x["text"] for x in fold_data]
        embeddings = self.st_model.encode(texts, show_progress_bar=True)
        
        # 使用 MES 進行初始採樣
        sampled_indices, proportions = self.max_entropy_sampling_with_proportions(embeddings, SAMPLE_SIZE)
        
        # 檢查每個類別的樣本數
        selected_data = fold_data.select(sampled_indices)
        selected_label_counts = Counter(selected_data["label"])
        
        # 補充樣本
        for label, count in selected_label_counts.items():
            if count < min_samples_per_class:
                # 找出該類別的所有樣本
                label_indices = [i for i, x in enumerate(fold_data) 
                            if x["label"] == label]
                # 隨機選擇額外的樣本
                additional_needed = min_samples_per_class - count
                if additional_needed > 0:
                    additional_indices = np.random.choice(
                        label_indices, 
                        size=min(additional_needed, len(label_indices)), 
                        replace=False
                    )
                    sampled_indices = np.concatenate([sampled_indices, additional_indices])
        
        # 確保不超過要求的樣本數
        if len(sampled_indices) > SAMPLE_SIZE:
            sampled_indices = np.random.choice(
                sampled_indices, SAMPLE_SIZE, replace=False)
        
        sampled_data = fold_data.select(sampled_indices)
        return sampled_data, proportions

    def process_fold(self, train_fold, val_fold, fold_num):
        """處理每個 fold 的數據，只對訓練集進行抽樣"""
        logging.info(f"Processing fold {fold_num + 1} data...")

        # 使用新的採樣方法
        train_sampled, proportions = self.sample_fold_data(train_fold)
        self.sample_proportions = proportions

        logging.info(f"Fold {fold_num + 1} - Train size: {len(train_sampled)}, Val size: {len(val_fold)}")
        return train_sampled, val_fold

    def train_ac_model(self):
        """訓練 Aspect Category 模型"""
        logging.info(f"Training AC model...")
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
            train_dataset=self.train_dataset.map(
                lambda x: {"text": x["text"], "label": x["ac"]}
            ),
            eval_dataset=self.val_dataset.map(
                lambda x: {"text": x["text"], "label": x["ac"]}
            ),
        )
        trainer.train()
        self.models["ac"] = ac_model

    def train_label_model(self, fold_num):
        """Train Polarity model."""
        logging.info(f"Training Polarity model for fold {fold_num + 1}...")
        
        # 檢查每個類別的樣本數
        from collections import Counter
        labels = self.train_dataset["label"]
        label_counts = Counter(labels)
        logging.info(f"Label distribution: {dict(label_counts)}")
        
        total_desired_samples = SAMPLE_SIZE  # 50
        
        # 如果只有一個類別，重新平衡
        if len(label_counts) == 1:
            logging.warning("Only one class found in training data. Rebalancing samples.")
            existing_label = list(label_counts.keys())[0]
            synthetic_label = "negative" if existing_label == "positive" else "positive"
            
            # 分配樣本數 (70% 主類別, 30% 合成類別)
            main_class_samples = int(total_desired_samples * 0.7)  # 35 samples
            synthetic_class_samples = total_desired_samples - main_class_samples  # 15 samples
            
            # 從原始數據中隨機選擇主類別樣本
            original_data = list(self.train_dataset)
            selected_main = np.random.choice(
                original_data, 
                size=main_class_samples, 
                replace=False
            ).tolist()
            
            # 創建合成樣本
            base_example = dict(original_data[0])
            synthetic_samples = []
            for _ in range(synthetic_class_samples):
                synthetic_example = dict(base_example)
                synthetic_example["label"] = synthetic_label
                synthetic_samples.append(synthetic_example)
            
            # 合併數據
            balanced_data = selected_main + synthetic_samples
            
            logging.info(f"Created balanced dataset with {main_class_samples} {existing_label} "
                        f"and {synthetic_class_samples} {synthetic_label} samples")
        else:
            balanced_data = list(self.train_dataset)
        
        self.train_dataset = Dataset.from_list(balanced_data)
        
        # 再次檢查類別分布
        new_labels = self.train_dataset["label"]
        new_label_counts = Counter(new_labels)
        logging.info(f"Balanced label distribution: {dict(new_label_counts)}")
    
        
        # 訓練代碼
        train_args = TrainingArguments(
            output_dir=f"{OUTPUT_DIR}/fold_{fold_num + 1}/label",
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
            train_dataset=self.train_dataset.map(
                lambda x: {"text": x["text"], "label": x["label"]}
            ),
            eval_dataset=self.val_dataset.map(
                lambda x: {"text": x["text"], "label": x["label"]}
            ),
        )
        trainer.train()
        self.models["label"] = label_model

    def train_absa_model(self):
        """訓練 ABSA 模型"""
        logging.info(f"Training ABSA model...")
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
            train_dataset=self.train_dataset.map(
                lambda x: {"text": x["text"], "aspect": x["span"], "label": x["label"]}
            ),
            eval_dataset=self.val_dataset.map(
                lambda x: {"text": x["text"], "aspect": x["span"], "label": x["label"]}
            ),
        )
        trainer.train()
        self.models["absa"] = absa_model

    def extract_opinion_terms(self, text, aspect):
        """改進的意見術語提取方法"""
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

        # 簡單的上下文 window
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

        y_true_ac, y_pred_ac = [], []
        y_true_label, y_pred_label = [], []

        all_y_true_spans, all_y_pred_spans = [], []
        all_y_true_ots,   all_y_pred_ots   = [], []

        eval_dataset = self.val_dataset if fold_num is not None else self.test_dataset

        for example in eval_dataset:
            text = example["text"]

            # AC
            ac_prediction = self.models["ac"].predict([text])[0]
            y_true_ac.append(example["ac"])
            y_pred_ac.append(ac_prediction)

            # Polarity
            polarity_prediction = self.models["label"].predict([text])[0]
            y_true_label.append(example["label"])
            y_pred_label.append(polarity_prediction)

            # ABSA
            absa_predictions = self.models["absa"].predict([text])[0]
            logging.info(f"absa_predictions: {absa_predictions}")

            if isinstance(absa_predictions, list):
                y_pred_spans = [
                    {"span": term.get("span", "")}
                    for term in absa_predictions if isinstance(term, dict)
                ]
                y_pred_ots = [
                    {"opinion_term": " ".join(self.extract_opinion_terms(text, term.get("span", ""))).strip()}
                    for term in absa_predictions if isinstance(term, dict) and term.get("span", "").strip()
                ]
            else:
                y_pred_spans = []
                y_pred_ots   = []

            y_pred_ots = [ot for ot in y_pred_ots if ot["opinion_term"].strip()]

            # ground truth
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

        from sklearn.metrics import precision_recall_fscore_support
        p_ac, r_ac, _, _ = precision_recall_fscore_support(y_true_ac, y_pred_ac, average="weighted", zero_division=0)
        if (p_ac + r_ac) > 0:
            f1_ac = 2 * p_ac * r_ac / (p_ac + r_ac)
        else:
            f1_ac = 0.0

        p_label, r_label, _, _ = precision_recall_fscore_support(y_true_label, y_pred_label, average="weighted", zero_division=0)
        if (p_label + r_label) > 0:
            f1_label = 2 * p_label * r_label / (p_label + r_label)
        else:
            f1_label = 0.0

        metrics_span = compute_f1_score(all_y_pred_spans, all_y_true_spans, "span", use_lcs=True)
        metrics_ot   = compute_f1_score(all_y_pred_ots,   all_y_true_ots,   "opinion_term", use_lcs=True)

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

    def calculate_average_metrics(self):
        """計算所有fold的平均指標"""
        avg_metrics = {
            "aspect_category": {"precision": 0.0, "recall": 0.0, "f1": 0.0},
            "polarity": {"precision": 0.0, "recall": 0.0, "f1": 0.0},
            "span": {"precision": 0.0, "recall": 0.0, "f1": 0.0},
            "opinion_term": {"precision": 0.0, "recall": 0.0, "f1": 0.0}
        }

        for metrics in self.fold_metrics:
            for task in avg_metrics:
                for metric in avg_metrics[task]:
                    avg_metrics[task][metric] += metrics[task][metric]

        n_folds = len(self.fold_metrics)
        for task in avg_metrics:
            for metric in avg_metrics[task]:
                avg_metrics[task][metric] /= n_folds

        self.results["average_metrics"] = avg_metrics
        self._log_average_metrics()
        return avg_metrics

    def _log_average_metrics(self):
        metrics = self.results["average_metrics"]
        logging.info(f"\nAverage Results for model: {self.model_name}")
        logging.info("=" * 50)
        for task, task_metrics in metrics.items():
            logging.info(f"\n{task.title()}:")
            for metric, value in task_metrics.items():
                logging.info(f"{metric}: {value:.4f}")

    def run(self):
        checkpoint_manager = CheckpointManager(OUTPUT_DIR)
        checkpoint = checkpoint_manager.load_checkpoint()

        if (checkpoint and "completed_models" in checkpoint and 
            self.model_name in checkpoint["completed_models"] and
            checkpoint["completed_models"][self.model_name].get("completed", False)):
            logging.info(f"{self.model_name} 已完成，直接跳過。")
            return

        self.load_data()

        kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
        dataset_indices = list(range(len(self.full_dataset)))

        start_fold = 0
        if checkpoint and "completed_models" in checkpoint and self.model_name in checkpoint["completed_models"]:
            start_fold = checkpoint["completed_models"][self.model_name]["current_fold"]
            logging.info(f"Resuming {self.model_name} from fold {start_fold + 1}")

        for fold_num, (train_idx, val_idx) in enumerate(kf.split(dataset_indices)):
            if fold_num < start_fold:
                continue

            logging.info(f"\nProcessing fold {fold_num + 1}/{N_FOLDS} for {self.model_name}")
            checkpoint_manager.save_checkpoint(self.model_name, fold_num)

            train_fold = self.full_dataset.select(train_idx.tolist())
            val_fold = self.full_dataset.select(val_idx.tolist())
            self.train_dataset, self.val_dataset = self.process_fold(train_fold, val_fold, fold_num)

            try:
                self.train_ac_model()
                self.train_label_model(fold_num)  # 修改這裡，加入 fold_num 參數
                self.train_absa_model()

                fold_metrics = self.evaluate(fold_num)
                self.fold_metrics.append(fold_metrics)
                results_file = os.path.join(OUTPUT_DIR, f"{self.model_name}_fold{fold_num + 1}_results.json")
                with open(results_file, 'w', encoding='utf-8') as f:
                    json.dump(fold_metrics, f, ensure_ascii=False, indent=2)
                    
            except Exception as e:
                logging.error(f"Error in fold {fold_num + 1} for {self.model_name}: {str(e)}")
                raise e

            logging.info(f"Completed fold {fold_num + 1}/{N_FOLDS}")

        checkpoint_manager.mark_completed(self.model_name)
        logging.info(f"Experiment completed for {self.model_name}")

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
            if not status["completed"]
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
