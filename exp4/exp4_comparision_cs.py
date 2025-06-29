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

logging.basicConfig(level=logging.INFO)

# -----------------------
# 使用多語系 mpnet 模型
# -----------------------
MODEL_NAME = "paraphrase-multilingual-mpnet-base-v2"

# -----------------------
# 主要參數設定
# -----------------------
OUTPUT_DIR = "results"
TRAIN_SIZE = 50  # cluster sampling
RANDOM_SEED = 42

Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

# -----------------------
# LCS & partial match 函式
# -----------------------
def lcs_length(x, y):
    m, n= len(x), len(y)
    dp=[[0]*(n+1) for _ in range(m+1)]
    for i in range(1,m+1):
        for j in range(1,n+1):
            if x[i-1]==y[j-1]:
                dp[i][j]=dp[i-1][j-1]+1
            else:
                dp[i][j]=max(dp[i-1][j], dp[i][j-1])
    return dp[m][n]

def compute_lcs_score(pred_term, ref_term):
    if not pred_term or not ref_term:
        return 0.0
    l= lcs_length(pred_term, ref_term)
    return l/ max(len(pred_term), len(ref_term))

def compute_f1_score(predictions, references, task_key, use_lcs=False):
    import numpy as np
    from scipy.optimize import linear_sum_assignment
    n = len(predictions)
    total_precision, total_recall = 0.0, 0.0

    for pred, ref in zip(predictions, references):
        pred_list = [d[task_key] for d in pred]
        ref_list = [d[task_key] for d in ref]
        
        if len(pred_list) == 0 and len(ref_list) == 0:
            total_precision += 1.0
            total_recall += 1.0
            continue
            
        if len(pred_list) == 0 or len(ref_list) == 0:
            continue

        score_mat = np.zeros((len(pred_list), len(ref_list)), dtype=np.float32)
        for i, pterm in enumerate(pred_list):
            for j, rterm in enumerate(ref_list):
                if use_lcs:
                    score_mat[i][j] = compute_lcs_score(pterm, rterm)
                else:
                    score_mat[i][j] = 1.0 if pterm == rterm else 0.0
                    
        cost = 1 - score_mat
        row_ind, col_ind = linear_sum_assignment(cost)
        match_score = sum(score_mat[row_ind[k], col_ind[k]] for k in range(len(row_ind)))

        precision = match_score / len(pred_list)
        recall = match_score / len(ref_list)
        
        total_precision += precision
        total_recall += recall

    avg_precision = total_precision / n
    avg_recall = total_recall / n
    
    # Calculate F1 using averaged precision and recall
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
        self.model_name= MODEL_NAME
        self.device= torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.nlp= spacy.load("en_core_web_sm")
        self.st_model= SentenceTransformer(self.model_name).to(self.device)

        self.model_output_dir= Path(OUTPUT_DIR)/self.model_name
        self.model_output_dir.mkdir(parents=True, exist_ok=True)
        self.results={"model": self.model_name, "train_size": TRAIN_SIZE}
        self.models={}

    def load_data(self):
        logging.info("Loading dataset: JaquanTW/fewshot-absaquad")
        dataset= load_dataset("JaquanTW/fewshot-absaquad")
        self.full_train_dataset= dataset["train"]
        self.val_dataset       = dataset["validation"]
        self.test_dataset      = dataset["test"]
        logging.info(f"Train={len(self.full_train_dataset)}, Val={len(self.val_dataset)}, Test={len(self.test_dataset)}")

    def cluster_sampling(self, data, num_samples):
        from sklearn.cluster import KMeans
        texts=[x["text"] for x in data]
        logging.info(f"Generating embeddings for cluster sampling: {len(texts)} samples...")
        emb= self.st_model.encode(texts, batch_size=32, show_progress_bar=True)
        kmeans= KMeans(n_clusters=num_samples, random_state=RANDOM_SEED).fit(emb)
        cluster_indices=[]
        for c in range(num_samples):
            members=np.where(kmeans.labels_==c)[0]
            if len(members)>0:
                cluster_indices.append(members[0])
        if len(cluster_indices)< num_samples:
            needed= num_samples- len(cluster_indices)
            all_inds= set(range(len(data)))
            used= set(cluster_indices)
            remain=list(all_inds- used)
            np.random.shuffle(remain)
            cluster_indices.extend(remain[:needed])
        subset= data.select(cluster_indices)
        return subset

    def train_ac_model(self):
        logging.info("Training AC model...")
        train_args= TrainingArguments(
            output_dir=str(self.model_output_dir/"ac"),
            num_epochs=5,batch_size=8,
            body_learning_rate=2e-5, head_learning_rate=1e-3,
        )
        unique_ac= list(set(self.train_dataset["ac"]))
        ac_model= SetFitModel.from_pretrained(self.model_name, labels=unique_ac)
        trainer= Trainer(
            model=ac_model, args=train_args,
            train_dataset= self.train_dataset.map(lambda x: {"text": x["text"], "label": x["ac"]}),
            eval_dataset= self.val_dataset.map(lambda x: {"text": x["text"], "label": x["ac"]})
        )
        trainer.train()
        self.models["ac"]= ac_model

    def train_label_model(self):
        logging.info("Training Polarity model...")
        train_args= TrainingArguments(
            output_dir=str(self.model_output_dir/"label"),
            num_epochs=5,batch_size=8,
            body_learning_rate=2e-5, head_learning_rate=1e-3,
        )
        unique_labels= list(set(self.train_dataset["label"]))
        label_model= SetFitModel.from_pretrained(self.model_name, labels=unique_labels)
        trainer= Trainer(
            model=label_model, args=train_args,
            train_dataset=self.train_dataset.map(lambda x:{"text": x["text"], "label": x["label"]}),
            eval_dataset=self.val_dataset.map(lambda x:{"text": x["text"], "label": x["label"]})
        )
        trainer.train()
        self.models["label"]= label_model

    def train_absa_model(self):
        logging.info("Training ABSA model...")
        absa_model= AbsaModel.from_pretrained(model_id=self.model_name, spacy_model="en_core_web_sm")
        train_args= TrainingArguments(
            output_dir=str(self.model_output_dir/"absa"),
            num_epochs=5,batch_size=8,
            body_learning_rate=2e-5, head_learning_rate=1e-3,
        )
        trainer= AbsaTrainer(
            model=absa_model, args=train_args,
            train_dataset=self.train_dataset.map(lambda x:{
                "text": x["text"], "aspect": x["span"], "label": x["label"]}),
            eval_dataset=self.val_dataset.map(lambda x:{
                "text": x["text"], "aspect": x["span"], "label": x["label"]})
        )
        trainer.train()
        self.models["absa"]= absa_model

    def extract_opinion_terms(self, text, aspect):
        doc= self.nlp(text)
        aspect_token=None
        for token in doc:
            if token.text.lower()== aspect.lower():
                aspect_token=token
                break
        if not aspect_token:
            return []
        opinion_terms=[]
        for token in doc:
            if (token.dep_ in {"amod","advmod"} and token.head==aspect_token) or \
               (token.head==aspect_token and token.pos_ in {"ADJ","ADV"}):
                opinion_terms.append(token.text)
        window_size=3
        idx= aspect_token.i
        for i in range(max(0, idx-window_size), min(len(doc), idx+window_size+1)):
            t= doc[i]
            if t.pos_ in {"ADJ","ADV"} and t.text not in opinion_terms:
                opinion_terms.append(t.text)
        return opinion_terms

    def evaluate(self, dataset, desc="Validation"):
            logging.info(f"Evaluating on {desc} set...")
            y_true_ac, y_pred_ac=[], []
            y_true_label, y_pred_label=[], []
            y_true_span_list, y_pred_span_list=[], []
            y_true_ot_list, y_pred_ot_list=[], []

            gold_data=[]
            pred_data=[]

            for example in dataset:
                text= example["text"]
                # AC
                ac_pred= self.models["ac"].predict([text])[0]
                y_true_ac.append(example["ac"])
                y_pred_ac.append(ac_pred)
                # Polarity
                pol_pred= self.models["label"].predict([text])[0]
                y_true_label.append(example["label"])
                y_pred_label.append(pol_pred)
                # ABSA
                absa_preds= self.models["absa"].predict([text])[0]
                pred_spans=[{"span": term.get("span","")} for term in absa_preds]
                pred_ots=[]
                for term in absa_preds:
                    sp_= term.get("span","")
                    if sp_.strip():
                        ot_terms= self.extract_opinion_terms(text, sp_)
                        ot_str= " ".join(ot_terms).strip()
                        if ot_str:
                            pred_ots.append({"opinion_term": ot_str})
                # gold spans/ots
                true_spans=[{"span": example["span"]}] if "span" in example else []
                if "ot" in example and example["ot"]:
                    if isinstance(example["ot"], list):
                        true_ots=[{"opinion_term": " ".join(example["ot"]).strip()}]
                    else:
                        true_ots=[{"opinion_term": example["ot"].strip()}]
                else:
                    true_ots=[{"opinion_term":""}]
                true_ots=[ot for ot in true_ots if ot["opinion_term"]]

                y_true_span_list.append(true_spans)
                y_pred_span_list.append(pred_spans)
                y_true_ot_list.append(true_ots)
                y_pred_ot_list.append(pred_ots)

                # gold 4-tuple
                gold_ac= example["ac"]
                gold_lb= example["label"]
                gold_sp= example["span"] if "span" in example else ""
                if len(true_ots)>0:
                    gold_ot= true_ots[0]["opinion_term"]
                else:
                    gold_ot=""
                gold_4=[ (gold_ac, gold_lb, gold_sp, gold_ot) ]

                # pred 4-tuple
                pred_4=[]
                if absa_preds:
                    for term in absa_preds:
                        sp__= term.get("span","")
                        ot_terms= self.extract_opinion_terms(text, sp__)
                        ot_str= " ".join(ot_terms).strip()
                        pred_4.append( (ac_pred, pol_pred, sp__, ot_str) )
                else:
                    pred_4.append((ac_pred, pol_pred,"",""))

                gold_data.append(gold_4)
                pred_data.append(pred_4)

            # Weighted metrics for AC
            p_ac, r_ac, _, _ = precision_recall_fscore_support(
                y_true_ac, y_pred_ac, average="weighted", zero_division=0
            )
            # Calculate F1 using formula
            if (p_ac + r_ac) > 0:
                f1_ac = 2 * p_ac * r_ac / (p_ac + r_ac)
            else:
                f1_ac = 0.0

            # Weighted metrics for polarity
            p_label, r_label, _, _ = precision_recall_fscore_support(
                y_true_label, y_pred_label, average="weighted", zero_division=0
            )
            # Calculate F1 using formula
            if (p_label + r_label) > 0:
                f1_label = 2 * p_label * r_label / (p_label + r_label)
            else:
                f1_label = 0.0

            # Get positive/negative metrics
            posneg_m_p, posneg_m_r, _,_ = precision_recall_fscore_support(
                y_true_label, y_pred_label,
                labels=["positive","negative"],
                average=None, zero_division=0
            )

            # partial on span/ot
            span_m= compute_f1_score(y_pred_span_list, y_true_span_list,"span", use_lcs=True)
            ot_m  = compute_f1_score(y_pred_ot_list, y_true_ot_list,"opinion_term", use_lcs=True)

            metrics={
                "aspect_category":{
                    "precision": float(p_ac),
                    "recall": float(r_ac),
                    "f1": float(f1_ac),
                },
                "polarity":{
                    "precision": float(p_label),
                    "recall": float(r_label), 
                    "f1": float(f1_label),
                    "positive_precision": float(posneg_m_p[0]),
                    "positive_recall": float(posneg_m_r[0]),
                    "negative_precision": float(posneg_m_p[1]),
                    "negative_recall": float(posneg_m_r[1]),
                },
                "span": span_m,
                "opinion_term": ot_m,
            }

            # Calculate quad exact match
            ex_p, ex_r, ex_f= self.compute_quad_exact_match(gold_data, pred_data)
            # Calculate quad partial match  
            pm_p, pm_r, pm_f= self.compute_quad_partial_match(gold_data, pred_data)

            metrics.setdefault("quad", {})
            metrics["quad"]["exact_4all_precision"]= ex_p
            metrics["quad"]["exact_4all_recall"]   = ex_r
            metrics["quad"]["exact_4all_f1"]       = ex_f
            metrics["quad"]["partial_4all_precision"]= pm_p
            metrics["quad"]["partial_4all_recall"]   = pm_r
            metrics["quad"]["partial_4all_f1"]       = pm_f

            logging.info(f"{desc} metrics: {json.dumps(metrics, indent=2)}")
            return metrics

    def compute_quad_exact_match(self, gold_data, pred_data):
        """
        exact => (ac,label,span,ot)全部相同 => score=1 else 0
        """
        import numpy as np
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
        import numpy as np
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

    def run(self):
        start_t= time.time()
        self.load_data()
        logging.info(f"Sampling {TRAIN_SIZE} from train set...")
        self.train_dataset= self.cluster_sampling(self.full_train_dataset, TRAIN_SIZE)
        logging.info("Using official val/test")

        self.train_ac_model()
        self.train_label_model()
        self.train_absa_model()

        val_metrics= self.evaluate(self.val_dataset, "Validation")
        self.results["val_metrics"]= val_metrics

        test_metrics= self.evaluate(self.test_dataset, "Test")
        self.results["test_metrics"]= test_metrics

        end_t= time.time()
        self.results["total_experiment_time_sec"]= (end_t- start_t)
        logging.info(f"Total experiment time= {end_t- start_t:.2f}s")

        stamp= datetime.now().strftime("%Y%m%d_%H%M%S")
        out_file= Path(OUTPUT_DIR)/ f"{self.model_name}_results_{stamp}.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2)
        logging.info(f"Results saved to: {out_file}")

if __name__=="__main__":
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)
        gname= torch.cuda.get_device_name(0)
        gmem = torch.cuda.get_device_properties(0).total_memory/(1024**3)
        logging.info(f"Using GPU: {gname} ({gmem:.2f} GB)")
    else:
        logging.info("Using CPU")

    experiment= SingleRunExperiment()
    experiment.run()
