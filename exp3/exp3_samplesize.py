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
from multiprocessing import Process  # 新增：匯入 Process
from transformers import EarlyStoppingCallback

logging.basicConfig(level=logging.INFO)

MODEL_CONFIG = {
    "paraphrase-multilingual-mpnetbase-v2": {
        "path": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        "type": "multilingual",
        "description": "High quality multilingual model"
    }
}

OUTPUT_DIR = "results"
SAMPLE_SIZES = [1, 10, 50, 100, 150, 200]
N_FOLDS = 5
RANDOM_SEED = 42

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
    def __init__(self, output_dir, sample_size):
        self.output_dir = output_dir
        self.sample_size = sample_size
        self.checkpoint_file = os.path.join(output_dir, f"model_checkpoint_{sample_size}.json")
        
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

class MajorityClassifier:
    def __init__(self, default_label):
        self.default_label = default_label
        
    def predict(self, texts):
        return [self.default_label for _ in texts]

class DummyABSAModel:
    def __init__(self):
        self.empty_result = []
        
    def predict(self, texts):
        return [[{"span": "", "label": "neutral"}] for _ in texts]
    
class EnhancedExperiment:
    def __init__(self, sample_size: int):
        self.model_name = "paraphrase-multilingual-mpnetbase-v2"
        self.sample_proportions = {"Cluster Sampling": 1.0} 
        self.model_config = MODEL_CONFIG[self.model_name]
        self.model_path = self.model_config["path"]
        self.sample_size = sample_size
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
                "train_size": sample_size,
                "n_folds": N_FOLDS
            }
        }
        
        self.model_output_dir = Path(OUTPUT_DIR) / f"{self.model_name}_{sample_size}"
        self.model_output_dir.mkdir(parents=True, exist_ok=True)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            logging.info(f"Using GPU: {gpu_name} with {gpu_memory:.2f}GB memory")
        else:
            logging.info("Using CPU for training")
        
        self.st_model = SentenceTransformer(self.model_path)
        self.st_model = self.st_model.to(self.device)

    def load_data(self):
        logging.info(f"Loading datasets...")
        dataset = load_dataset("JaquanTW/fewshot-absaquad")
        self.full_dataset = dataset["train"]
        self.test_dataset = dataset["test"]
        logging.info(f"Full dataset size: {len(self.full_dataset)}")
        logging.info(f"Test dataset size: {len(self.test_dataset)}")

    def cluster_sampling(self, embeddings, num_samples):
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

    def sample_data(self, data, size):
        if len(data) <= size:
            logging.info(f"Data size {len(data)} is smaller than required size {size}, using full dataset")
            return data
            
        logging.info(f"Generating embeddings for sampling...")
        texts = [x["text"] for x in data]
        embeddings = self.st_model.encode(texts, batch_size=32, show_progress_bar=True)
        
        sampled_indices = self.cluster_sampling(embeddings, size)
        sampled_data = data.select(sampled_indices)
        
        logging.info(f"Sampled size: {len(sampled_data)} (from {len(data)})")
        return sampled_data

    def process_fold(self, train_fold, val_fold, fold_num):
        logging.info(f"Processing fold {fold_num + 1} data...")
        train_sampled = self.sample_data(train_fold, self.sample_size)
        val_dataset = val_fold
        logging.info(f"Fold {fold_num + 1} - Train size: {len(train_sampled)}, Val size: {len(val_dataset)}")
        return train_sampled, val_dataset
    
    def train_ac_model(self):
        if len(self.train_dataset) < 2:  # 檢查訓練資料是否太少
            logging.warning(f"Insufficient training samples ({len(self.train_dataset)})")
            unique_ac_labels = list(set(self.train_dataset["ac"]))
            self.models["ac"] = MajorityClassifier(unique_ac_labels[0])
            return

        logging.info(f"Training AC model...")
        train_args = TrainingArguments(
            output_dir=str(self.model_output_dir / "ac"),
            num_epochs=5,
            batch_size=8,
            body_learning_rate=2e-5,
            head_learning_rate=1e-3,
            load_best_model_at_end=True,  # 新增
            save_total_limit=1,          # 新增
            eval_strategy="epoch", # 新增
            save_strategy="epoch",       # 與評估策略一致
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
            callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],

        )
        trainer.train()
        self.models["ac"] = ac_model

    def train_label_model(self):
        if len(self.train_dataset) < 2:
            logging.warning(f"Insufficient training samples ({len(self.train_dataset)})")
            unique_labels = list(set(self.train_dataset["label"]))
            self.models["label"] = MajorityClassifier(unique_labels[0])
            return

        logging.info(f"Training Polarity model...")
        train_args = TrainingArguments(
            output_dir=str(self.model_output_dir / "label"),
            num_epochs=5,
            batch_size=8,
            body_learning_rate=2e-5,
            head_learning_rate=1e-3,
            load_best_model_at_end=True,  # 新增
            save_total_limit=1,          # 新增
            eval_strategy="epoch", # 新增
            save_strategy="epoch",       # 與評估策略一致
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
            callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
        )
        trainer.train()
        self.models["label"] = label_model

    def train_absa_model(self):
        if len(self.train_dataset) < 2:
            logging.warning(f"Insufficient training samples ({len(self.train_dataset)})")
            self.models["absa"] = DummyABSAModel()
            return

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
            load_best_model_at_end=True,  # 新增
            save_total_limit=1,          # 新增
            eval_strategy="epoch", # 新增
            save_strategy="epoch",       # 與評估策略一致
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
            callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
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

        logging.info(f"AC Metrics: {metrics['AC Metrics']}")
        logging.info(f"Polarity Metrics: {metrics['Polarity Metrics']}")
        logging.info(f"Span Metrics: {metrics['Span Metrics']}")
        logging.info(f"Opinion Term Metrics: {metrics['Opinion Term Metrics']}")

        return metrics
    
    def calculate_average_metrics(self):
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
        logging.info(f"\nAverage Results for model with sample size: {self.sample_size}")
        logging.info("=" * 50)
        for task, task_metrics in metrics.items():
            logging.info(f"\n{task.title()}:")
            for metric, value in task_metrics.items():
                logging.info(f"{metric}: {value:.4f}")

    def _save_final_results(self):
        avg_metrics = self.calculate_average_metrics()
        final_metrics = {
            "average_metrics": avg_metrics,
            "Sample_size": self.sample_size
        }
        results_file = os.path.join(OUTPUT_DIR, f"{self.model_name}_{self.sample_size}_average_results.json")
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(final_metrics, f, ensure_ascii=False, indent=2)

    #---------------------------------------------------------------------#
    # **刪除**: 原本 run() 裡的多fold邏輯，改為只針對 "單一 fold" 或 "整個流程" 
    # 下面保留 run() 只做「最終存平均」的事；或你可以直接刪掉。
    #---------------------------------------------------------------------#
    def run(self):
        """原本有多fold迴圈的地方，先簡化改成只負責算 average + 存最終檔."""
        # 這裡什麼都不做也行，看你要不要保留
        pass


#=============================================================================#
# 新增一個「單一 fold 執行流程」函式: run_single_fold
# 這會在子進程裡執行，把做某個 fold 的訓練/評估/Checkpoint 都放進來
#=============================================================================#
def run_single_fold(sample_size, fold_num):
    logging.info(f"[Process] Start fold={fold_num}, sample_size={sample_size}")

    # 初始化實驗物件 & CheckpointManager
    experiment = EnhancedExperiment(sample_size=sample_size)
    ckpt_manager = CheckpointManager(OUTPUT_DIR, sample_size)

    # 讀取 checkpoint，判斷此 fold 是否已完成
    ckpt = ckpt_manager.load_checkpoint()
    if ckpt and "completed_models" in ckpt and experiment.model_name in ckpt["completed_models"]:
        done_fold = ckpt["completed_models"][experiment.model_name]["current_fold"]
        if done_fold > fold_num:
            logging.info(f"Fold {fold_num+1} already done, skip.")
            return

    # 載入完整資料
    experiment.load_data()

    # 準備該 fold 的 train/val
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    dataset_indices = list(range(len(experiment.full_dataset)))
    train_idx, val_idx = list(kf.split(dataset_indices))[fold_num]
    train_fold = experiment.full_dataset.select(train_idx.tolist())
    val_fold = experiment.full_dataset.select(val_idx.tolist())
    experiment.train_dataset, experiment.val_dataset = experiment.process_fold(train_fold, val_fold, fold_num)

    # 執行訓練
    experiment.train_ac_model()
    experiment.train_label_model()
    experiment.train_absa_model()

    # 評估
    fold_metrics = experiment.evaluate(fold_num)
    experiment.fold_metrics.append(fold_metrics)

    # 存 fold 的結果
    results_file = os.path.join(
        OUTPUT_DIR, f"{experiment.model_name}_{sample_size}_fold{fold_num+1}_results.json"
    )
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(fold_metrics, f, ensure_ascii=False, indent=2)

    # 更新 checkpoint
    ckpt_manager.save_checkpoint(experiment.model_name, fold_num+1)
    if fold_num+1 == N_FOLDS:
        ckpt_manager.mark_completed(experiment.model_name)

    logging.info(f"[Process] Fold {fold_num+1} done, exiting process.")


#=============================================================================#
# 主程式: 將每個 fold 以子進程的方式執行
#=============================================================================#
def run_experiments():
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    logging.info("Starting experiments with different sample sizes...")

    for sample_size in SAMPLE_SIZES:
        logging.info(f"\n[Main] Start experiment sample_size={sample_size}")

        # 先看 checkpoint 是否已全部完成
        ckpt_manager = CheckpointManager(OUTPUT_DIR, sample_size)
        ckpt = ckpt_manager.load_checkpoint()
        model_name = "paraphrase-multilingual-mpnetbase-v2"
        if (ckpt and "completed_models" in ckpt and model_name in ckpt["completed_models"] and
            ckpt["completed_models"][model_name].get("completed", False)):
            logging.info(f"sample_size={sample_size} 已全部完成，跳過。")
            continue

        # 逐一 fold 開子進程
        for fold_num in range(N_FOLDS):
            # 也檢查一下是否需要跳過
            ckpt = ckpt_manager.load_checkpoint()
            if (ckpt and "completed_models" in ckpt and model_name in ckpt["completed_models"]):
                done_fold = ckpt["completed_models"][model_name]["current_fold"]
                if done_fold > fold_num:
                    logging.info(f"Fold {fold_num+1} already done, skip.")
                    continue

            # 開新的子進程執行該 fold
            p = Process(target=run_single_fold, args=(sample_size, fold_num))
            p.start()
            p.join()  # 等子進程結束，才能繼續跑下一個 fold

            # 子進程結束後，GPU記憶體自動釋放
            torch.cuda.empty_cache()

        logging.info(f"[Main] All folds done for sample_size={sample_size}")

        # 如果想在這裡計算平均再存檔，可以再用 EnhancedExperiment 做一次彙整
        # (也可以不做，或看你需要)
        # ----------------------------------------------------------------------
        # example:
        exp = EnhancedExperiment(sample_size)
        # 假設只是想要收集 fold 結果，需自己讀回 fold_metrics ...
        # 這裡簡化不做了

        # exp._save_final_results()  # 如果要算 average 的話
        # ----------------------------------------------------------------------


if __name__ == "__main__":
    run_experiments()