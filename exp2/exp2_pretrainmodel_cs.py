import logging
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any
from datasets import load_dataset
from setfit import SetFitModel, AbsaModel, Trainer, AbsaTrainer
from setfit import TrainingArguments
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import KFold  # 新增這行
import spacy
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from scipy.optimize import linear_sum_assignment
import os
import torch
from sklearn.neighbors import NearestNeighbors 

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
        
        # 設置輸出目錄
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

    def cluster_sampling(self, embeddings, num_samples):
        """使用 KMeans 進行 cluster sampling"""
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


    def process_fold(self, train_fold, val_fold, fold_num):
        """處理每個 fold 的數據，只對訓練集採樣，驗證集保持完整"""
        logging.info(f"Processing fold {fold_num + 1} data...")

        # 只對訓練集進行採樣
        train_sampled, proportions = self.sample_data(train_fold, SAMPLE_SIZE)

        # 設置 self.sample_proportions，避免 AttributeError
        self.sample_proportions = proportions

        # 驗證集保持完整
        val_dataset = val_fold

        logging.info(f"Fold {fold_num + 1} - Train size: {len(train_sampled)}, Val size: {len(val_dataset)}")
        return train_sampled, val_dataset



    def max_entropy_sampling(self, embeddings, num_samples):
        """最大熵抽樣（Max Entropy Sampling, MES）"""
        nn = NearestNeighbors(n_neighbors=min(5, len(embeddings) - 1))
        nn.fit(embeddings)
        distances, _ = nn.kneighbors(embeddings)
        entropies = -np.sum(np.log(distances + 1e-10) * distances, axis=1)
        return np.argsort(entropies)[-num_samples:]

    def max_entropy_sampling_with_proportions(self, embeddings, num_samples):
        """最大熵抽樣 + 產生 proportions"""
        indices = self.max_entropy_sampling(embeddings, num_samples)  # 加上 self.
        proportions = {"Max Entropy (MES)": 1.0}
        return indices, proportions

    def sample_data(self, data, size, use_mes=False):
        """對數據進行抽樣，可選擇使用最大熵抽樣 (MES) 或 Cluster Sampling (CS)"""
        if len(data) <= size:
            logging.info(f"Data size {len(data)} is smaller than required size {size}, using full dataset")
            return data, {"full_dataset": 1.0}  

        logging.info(f"Generating embeddings for sampling...")
        texts = [x["text"] for x in data]
        embeddings = self.st_model.encode(texts, batch_size=32, show_progress_bar=True)

        if use_mes:
            sampled_indices, proportions = self.max_entropy_sampling_with_proportions(embeddings, size)  # 加上 self.
        else:
            sampled_indices = self.cluster_sampling(embeddings, size)
            proportions = {"Cluster Sampling (CS)": 1.0}

        sampled_data = data.select(sampled_indices)
        logging.info(f"Sampled size: {len(sampled_data)} (from {len(data)}) using {'MES' if use_mes else 'CS'}")

        return sampled_data, proportions
    
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

    def train_label_model(self):
        """訓練極性分類模型"""
        logging.info(f"Training Polarity model...")
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

    def _log_average_metrics(self):
        """記錄平均指標"""
        metrics = self.results["average_metrics"]
        logging.info(f"\nAverage Results for model: {self.model_name}")
        logging.info("=" * 50)
        for task, task_metrics in metrics.items():
            logging.info(f"\n{task.title()}:")
            for metric, value in task_metrics.items():
                logging.info(f"{metric}: {value:.4f}")

    def run(self):
        """執行完整的實驗流程"""
        checkpoint_manager = CheckpointManager(OUTPUT_DIR)
        checkpoint = checkpoint_manager.load_checkpoint()

        # 檢查是否已完成
        if (checkpoint and "completed_models" in checkpoint and 
            self.model_name in checkpoint["completed_models"] and
            checkpoint["completed_models"][self.model_name].get("completed", False)):
            logging.info(f"{self.model_name} 已完成，直接跳過。")
            return

        self.load_data()

        # 創建 KFold 分割器
        kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
        dataset_indices = list(range(len(self.full_dataset)))

        # 確定起始 fold
        start_fold = 0
        if checkpoint and "completed_models" in checkpoint and self.model_name in checkpoint["completed_models"]:
            start_fold = checkpoint["completed_models"][self.model_name]["current_fold"]
            logging.info(f"Resuming {self.model_name} from fold {start_fold + 1}")

        # 執行每個 fold
        for fold_num, (train_idx, val_idx) in enumerate(kf.split(dataset_indices)):
            if fold_num < start_fold:
                continue

            logging.info(f"\nProcessing fold {fold_num + 1}/{N_FOLDS} for {self.model_name}")
            checkpoint_manager.save_checkpoint(self.model_name, fold_num)

            # 準備當前 fold 的數據
            train_fold = self.full_dataset.select(train_idx.tolist())
            val_fold = self.full_dataset.select(val_idx.tolist())
            self.train_dataset, self.val_dataset = self.process_fold(train_fold, val_fold, fold_num)

            try:
                # 訓練模型
                self.train_ac_model()
                self.train_label_model()
                self.train_absa_model()

                # 評估和保存結果
                fold_metrics = self.evaluate(fold_num)
                self.fold_metrics.append(fold_metrics)
                results_file = os.path.join(OUTPUT_DIR, f"{self.model_name}_fold{fold_num + 1}_results.json")
                with open(results_file, 'w', encoding='utf-8') as f:
                    json.dump(fold_metrics, f, ensure_ascii=False, indent=2)
                    
            except Exception as e:
                logging.error(f"Error in fold {fold_num + 1} for {self.model_name}: {str(e)}")
                raise e

            logging.info(f"Completed fold {fold_num + 1}/{N_FOLDS}")

        # 完成所有fold後的處理
        self._save_final_results()
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
        # 加強檢查已完成模型的邏輯
        completed = checkpoint["completed_models"].get(model_name, {}).get("completed", False)
        current_fold = checkpoint["completed_models"].get(model_name, {}).get("current_fold", 0)
        logging.info(f"Checking status for {model_name}: completed={completed}, current_fold={current_fold}")

        if completed:
            logging.info(f"Skipping {model_name}: already completed.")
            continue

        # 未完成模型執行實驗
        logging.info(f"Starting experiments with {model_name}")
        try:
            experiment = EnhancedExperiment(model_name)
            experiment.run()
            all_results[model_name] = experiment.results
            checkpoint_manager.mark_completed(model_name)  # 標記完成
        except Exception as e:
            logging.error(f"Error with {model_name}: {e}")
            continue  # 不影響其他模型

    # 保存比較結果
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
        # 這裡可以再擴充對 results 做排名、比較等
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
            model for model, status in checkpoint["completed_models"].items()
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