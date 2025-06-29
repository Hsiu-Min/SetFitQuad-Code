import logging
import os
import json
import numpy as np
from datasets import load_dataset
from setfit import SetFitModel, AbsaModel, Trainer, AbsaTrainer
from setfit import TrainingArguments
from sentence_transformers import SentenceTransformer
from sklearn.metrics import precision_recall_fscore_support
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.linear_model import Lasso, Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV  # 新增這行
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from scipy.optimize import linear_sum_assignment
from sklearn.model_selection import KFold
import spacy
import json
import os
import math
from functools import partial

logging.basicConfig(level=logging.INFO)

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
OUTPUT_DIR = "results"
TRAIN_SIZE = 50
VAL_SIZE = 10
N_FOLDS = 5 # 新增: 定義fold數量

# 所有的抽樣方法函數保持不變
def save_results(output_dir, sampling_name, metrics, fold=None):
    """將結果保存為 JSON 檔案，支援fold的結果保存"""
    if fold is not None:
        filepath = f"{output_dir}/{sampling_name}_fold{fold}_results.json"
    else:
        filepath = f"{output_dir}/{sampling_name}_results.json"
    
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    logging.info(f"Results saved to {filepath}")

def random_seed_sampling(embeddings, num_samples):
    """隨機種子抽樣"""
    np.random.seed(42)
    return np.random.choice(len(embeddings), num_samples, replace=False)

def random_seed_sampling_with_proportions(embeddings, num_samples):
    """隨機種子抽樣，比例設定為 100%。"""
    indices = random_seed_sampling(embeddings, num_samples)
    proportions = {"Random Seed (RS)": 1.0}  # 單一策略比例為 100%
    return indices, proportions


def grid_sampling(embeddings, num_samples):
    """網格抽樣"""
    return np.linspace(0, len(embeddings) - 1, num_samples, dtype=int)

def grid_sampling_with_proportions(embeddings, num_samples):
    """網格抽樣，比例設定為 100%。"""
    indices = grid_sampling(embeddings, num_samples)
    proportions = {"Grid Sampling (GS)": 1.0}  # 單一策略比例為 100%
    return indices, proportions

def max_min_distance_sampling(embeddings, num_samples):
    """最大-最小距離抽樣"""
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
    
def max_min_distance_sampling_with_proportions(embeddings, num_samples):
    indices = max_min_distance_sampling(embeddings, num_samples)
    proportions = {"Max-Min Distance (MMDS)": 1.0}
    return indices, proportions


def density_based_sampling(embeddings, num_samples):
    """密度基礎抽樣"""
    nn = NearestNeighbors(n_neighbors=min(5, len(embeddings) - 1)).fit(embeddings)
    distances, _ = nn.kneighbors(embeddings)
    density_scores = np.sum(distances, axis=1)
    return np.argsort(density_scores)[:num_samples]

def density_based_sampling_with_proportions(embeddings, num_samples):
    indices = density_based_sampling(embeddings, num_samples)
    proportions = {"Density-based (DBS)": 1.0}
    return indices, proportions


def max_entropy_sampling(embeddings, num_samples):
    """最大熵抽樣"""
    nn = NearestNeighbors(n_neighbors=min(5, len(embeddings) - 1))
    nn.fit(embeddings)
    distances, _ = nn.kneighbors(embeddings)
    entropies = -np.sum(np.log(distances + 1e-10) * distances, axis=1)
    return np.argsort(entropies)[-num_samples:]

def max_entropy_sampling_with_proportions(embeddings, num_samples):
    indices = max_entropy_sampling(embeddings, num_samples)
    proportions = {"Max Entropy (MES)": 1.0}
    return indices, proportions


def cluster_sampling(embeddings, num_samples):
    """聚類抽樣"""
    kmeans = KMeans(n_clusters=num_samples, random_state=42).fit(embeddings)
    return np.array([np.where(kmeans.labels_ == i)[0][0] for i in range(num_samples)])

def cluster_sampling_with_proportions(embeddings, num_samples):
    indices = cluster_sampling(embeddings, num_samples)
    proportions = {"Cluster Sampling (CS)": 1.0}
    return indices, proportions



def set_default_proportions(self, proportions):
    """設置默認比例，若 proportions 為 None，設為 100%。"""
    if proportions is None:
        proportions = {self.sampling_name: 1.0}
    self.sample_proportions = proportions





def create_enhanced_features(embeddings, indices):
    """計算更全面的特徵"""
    selected_embeddings = embeddings[indices]
    
    # 1. 基本距離特徵
    distances = np.linalg.norm(embeddings[:, np.newaxis] - selected_embeddings, axis=2)
    min_distances = np.min(distances, axis=1)
    mean_distances = np.mean(distances, axis=1)
    max_distances = np.max(distances, axis=1)
    
    # 2. 信息熵特徵
    distance_probs = distances / (np.sum(distances, axis=1, keepdims=True) + 1e-10)
    entropy = -np.sum(distance_probs * np.log(distance_probs + 1e-10), axis=1)
    
    # 3. 多樣性特徵
    selected_distances = np.linalg.norm(selected_embeddings[:, np.newaxis] - selected_embeddings, axis=2)
    diversity = np.mean(selected_distances[selected_distances > 0])
    diversity_scores = np.full_like(min_distances, diversity)
    
    # 4. 覆蓋度特徵
    coverage = np.mean(np.exp(-min_distances))
    coverage_scores = np.full_like(min_distances, coverage)
    
    # 組合所有特徵
    features = np.column_stack([
        min_distances,   # 最小距離
        mean_distances,  # 平均距離
        max_distances,   # 最大距離
        entropy,         # 熵
        diversity_scores,# 多樣性
        coverage_scores  # 覆蓋度
    ])
    
    return features

def random_forest_combination_with_proportions(embeddings, num_samples, base_prop=0.3):
    n = embeddings.shape[0]
    
    # =====================
    # Step 1: 建立「基礎策略」樣本 (跟原本相同)
    # =====================
    base_indices = {
        "Random Seed (RS)": random_seed_sampling(embeddings, int(n * base_prop)),
        "Grid Sampling (GS)": grid_sampling(embeddings, int(n * base_prop)),
        "Max-Min Distance (MMDS)": max_min_distance_sampling(embeddings, int(n * base_prop)),
        "Density-based (DBS)": density_based_sampling(embeddings, int(n * base_prop)),
        "Max Entropy (MES)": max_entropy_sampling(embeddings, int(n * base_prop)),
        "Cluster Sampling (CS)": cluster_sampling(embeddings, int(n * base_prop)),
    }

    # =====================
    # Step 2: 建立「豐富特徵表示」(每個策略四種特徵)
    # =====================
    strategy_features = np.zeros((n, len(base_indices) * 4))
    for i, (strategy_name, indices) in enumerate(base_indices.items()):
        selected_embeddings = embeddings[indices]
        distances = np.linalg.norm(embeddings[:, np.newaxis] - selected_embeddings, axis=2)
        min_distances = np.min(distances, axis=1)
        mean_distances = np.mean(distances, axis=1)
        std_distances = np.std(distances, axis=1)
        
        # 熵特徵 (針對 MES，但這裡對所有都計算)
        probs = distances / (np.sum(distances, axis=1, keepdims=True) + 1e-10)
        entropy = -np.sum(probs * np.log(probs + 1e-10), axis=1)
        
        # 填入 strategy_features
        strategy_features[:, i*4]   = min_distances
        strategy_features[:, i*4+1] = mean_distances
        strategy_features[:, i*4+2] = std_distances
        strategy_features[:, i*4+3] = entropy

    # =====================
    # Step 3: 特徵標準化
    # =====================
    scaler = StandardScaler()
    strategy_features = scaler.fit_transform(strategy_features)

    # =====================
    # Step 4: 豐富目標值
    # =====================
    center = np.mean(embeddings, axis=0)
    distances_to_center = np.linalg.norm(embeddings - center, axis=1)
    
    nn = NearestNeighbors(n_neighbors=min(5, len(embeddings)))
    nn.fit(embeddings)
    distances, _ = nn.kneighbors(embeddings)
    
    # RBF 密度
    density_scores = np.exp(-np.sum(distances, axis=1))
    # 多樣性 (方差)
    diversity_scores = np.var(embeddings, axis=1)
    # 訊息熵
    entropy_scores = -np.sum(distances * np.log(distances + 1e-10), axis=1)
    
    # 組合 target
    target = (distances_to_center * diversity_scores * (entropy_scores ** 2)) / (density_scores ** 0.5)

    # =====================
    # Step 5: RandomForest + GridSearchCV
    #       (跟原程式相同，做5次取平均重要度)
    # =====================
    rf_models = []
    importances = []
    for seed_offset in range(5):
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [None, 10],
            'min_samples_split': [2, 5]
        }
        rf = GridSearchCV(
            RandomForestRegressor(random_state=42 + seed_offset),
            param_grid,
            cv=5,
            scoring='neg_mean_squared_error'
        )
        rf.fit(strategy_features, target)
        rf_models.append(rf.best_estimator_)
        importances.append(rf.best_estimator_.feature_importances_)
    
    # 合併所有模型的特徵重要性
    final_importances = np.mean(importances, axis=0)

    # 每個策略對應4個特徵 => 取平均
    strategy_names = list(base_indices.keys())
    strategy_weights = np.zeros(len(strategy_names))
    for i in range(len(strategy_names)):
        strategy_weights[i] = np.mean(final_importances[i*4:(i+1)*4])

    # =====================
    # Step 5.1: 針對 MES 做下限
    # =====================
    mes_idx = strategy_names.index("Max Entropy (MES)")
    max_weight = np.max(strategy_weights)
    min_mes_weight = max_weight * 0.8  # MES 至少是 max_weight 的 80%
    if strategy_weights[mes_idx] < min_mes_weight:
        strategy_weights[mes_idx] = min_mes_weight
    
    # =====================
    # Step 6: 動態閾值 (去掉過低的權重)
    # =====================
    strategy_weights /= np.sum(strategy_weights)  # 先歸一化
    importance_threshold = np.percentile(strategy_weights, 25)  # 第25百分位
    strategy_weights[strategy_weights < importance_threshold] = 0.0

    # 再次歸一化，確保 sum=1
    total_w = np.sum(strategy_weights)
    if total_w > 0:
        strategy_weights /= total_w
    else:
        # 如果全部都是 0，就平均分
        strategy_weights = np.ones(len(strategy_names)) / len(strategy_names)

    # =====================
    # Step 7: 分配樣本（改良後分兩輪）
    # =====================
    # -- 第一輪：給權重大於0的策略每人min_samples
    strategy_samples = {}
    min_samples_per_strategy = max(1, int(num_samples * 0.1))

    # 找出非零權重的策略
    valid_strategies = [
        s for s, w in zip(strategy_names, strategy_weights) if w > 0
    ]

    # 先給這些策略 min_samples
    for s in valid_strategies:
        strategy_samples[s] = min_samples_per_strategy

    total_allocated = sum(strategy_samples.values())
    if total_allocated > num_samples:
        # 如果連最小配額都超過了 num_samples，做一個fallback處理：
        logging.warning("Min samples sum exceed total. Reducing to 1 per valid strategy.")
        strategy_samples = {s: 1 for s in valid_strategies}
        total_allocated = len(valid_strategies)

    remaining = num_samples - total_allocated

    # -- 第二輪：把剩餘樣本依比例分配
    if remaining > 0 and valid_strategies:
        # 取得 valid_strategies 的權重，做一次歸一化
        valid_weights = np.array([
            strategy_weights[strategy_names.index(s)] for s in valid_strategies
        ])
        vw_sum = valid_weights.sum()
        if vw_sum > 0:
            valid_weights /= vw_sum
            distributed = 0
            for i in range(len(valid_strategies) - 1):
                add_num = int(remaining * valid_weights[i])
                strategy_samples[valid_strategies[i]] += add_num
                distributed += add_num
            # 剩餘都給最後一個
            leftover = remaining - distributed
            strategy_samples[valid_strategies[-1]] += leftover

    # 最終檢查
    assert sum(strategy_samples.values()) == num_samples, "Sample allocation mismatch."

    # =====================
    # Step 8: 依分配量抽樣 & 計算實際比例
    # =====================
    all_indices = []
    for s, n_samples_s in strategy_samples.items():
        if n_samples_s > 0:
            sampling_function = SAMPLING_METHODS.get(s)
            if sampling_function is None:
                raise KeyError(f"Sampling method {s} not found.")
            sampled = sampling_function(embeddings, n_samples_s)
            if isinstance(sampled, tuple):
                sampled = sampled[0]
            all_indices.extend(sampled)

    # 去重
    all_indices = np.unique(all_indices)
    # 如果超量，就隨機壓縮；不足就用 random_seed 補
    if len(all_indices) > num_samples:
        selected_indices = np.random.choice(all_indices, num_samples, replace=False)
    else:
        need_more = num_samples - len(all_indices)
        if need_more > 0:
            add_indices = random_seed_sampling(embeddings, need_more)
            selected_indices = np.concatenate([all_indices, add_indices])
        else:
            selected_indices = all_indices

    # 重新計算「實際」的抽樣分配
    # 為了符合最終實際狀況(若有重複index或去重/補抽)，可做一次統計
    final_counts = {s: 0 for s in strategy_samples}
    for s, n_samples_s in strategy_samples.items():
        final_counts[s] = n_samples_s  # 此處直接用「理論分配」, 
                                       # 若想更細，需再看 selected_indices 裡哪些來自哪策略(較複雜)

    # 最終比例 = 分配量 / num_samples
    proportions = {
        s: final_counts[s] / num_samples for s in final_counts
    }

    return selected_indices, proportions


def lasso_sample_selection_with_proportions(embeddings, num_samples, base_prop=0.3):
    n = embeddings.shape[0]

    # Step 1: 使用完整數據集產生基礎策略樣本
    base_indices = {
        "Random Seed (RS)": random_seed_sampling(embeddings, int(n * base_prop)),
        "Grid Sampling (GS)": grid_sampling(embeddings, int(n * base_prop)),
        "Max-Min Distance (MMDS)": max_min_distance_sampling(embeddings, int(n * base_prop)),
        "Density-based (DBS)": density_based_sampling(embeddings, int(n * base_prop)),
        "Max Entropy (MES)": max_entropy_sampling(embeddings, int(n * base_prop)),
        "Cluster Sampling (CS)": cluster_sampling(embeddings, int(n * base_prop)),
    }

    # Step 2: 建立改進的特徵矩陣
    strategy_features = np.zeros((n, len(base_indices) * 4))  # 每個策略4個特徵
    for i, (strategy_name, indices) in enumerate(base_indices.items()):
        selected_embeddings = embeddings[indices]
        
        # 計算距離矩陣
        distances = np.linalg.norm(embeddings[:, np.newaxis] - selected_embeddings, axis=2)
        
        # 1. 距離特徵
        min_distances = np.min(distances, axis=1)
        mean_distances = np.mean(distances, axis=1)
        
        # 2. 熵特徵
        distance_probs = distances / (np.sum(distances, axis=1, keepdims=True) + 1e-10)
        entropy = -np.sum(distance_probs * np.log(distance_probs + 1e-10), axis=1)
        
        # 3. 多樣性特徵（樣本間的差異性）
        diversity = np.std(distances, axis=1)
        
        # 存儲特徵
        strategy_features[:, i*4] = min_distances
        strategy_features[:, i*4+1] = mean_distances
        strategy_features[:, i*4+2] = entropy
        strategy_features[:, i*4+3] = diversity

    # Step 3: 特徵標準化
    scaler = StandardScaler()
    strategy_features = scaler.fit_transform(strategy_features)

    # Step 4: 使用改進的目標值計算
    # 4.1 距離中心性
    center = np.mean(embeddings, axis=0)
    distances_to_center = np.linalg.norm(embeddings - center, axis=1)
    
    # 4.2 局部密度（使用RBF核）
    nn = NearestNeighbors(n_neighbors=min(5, len(embeddings)))
    nn.fit(embeddings)
    distances, _ = nn.kneighbors(embeddings)
    density_scores = np.exp(-np.sum(distances, axis=1))
    
    # 4.3 信息熵
    entropy_scores = -np.sum(distances * np.log(distances + 1e-10), axis=1)
    
    # 組合目標值，增加熵的權重，降低密度的影響
    target = (distances_to_center * (entropy_scores ** 2)) / (density_scores ** 0.5)
    # 標準化目標值
    target = (target - np.mean(target)) / np.std(target)

    # Step 5: 使用 Lasso 並進行交叉驗證
    param_grid = {
        'alpha': [0.0001, 0.001, 0.01]  # 使用較小的 alpha 值以減少稀疏性
    }
    lasso_cv = GridSearchCV(
        Lasso(random_state=42, max_iter=10000, tol=1e-4),
        param_grid,
        cv=5,
        scoring='neg_mean_squared_error'
    )
    lasso_cv.fit(strategy_features, target)
    model = lasso_cv.best_estimator_

    # Step 6: 計算稀疏權重
    strategy_weights = np.zeros(len(base_indices))
    for i in range(len(base_indices)):
        strategy_weights[i] = np.mean(np.abs(model.coef_[i*4:(i+1)*4]))
    
    # 確保 MES 的權重不會太低
    mes_idx = list(base_indices.keys()).index("Max Entropy (MES)")
    min_mes_weight = np.max(strategy_weights) * 0.8  # MES 至少要有最大權重的 80%
    if strategy_weights[mes_idx] < min_mes_weight:
        strategy_weights[mes_idx] = min_mes_weight
    
    # 重新標準化權重
    strategy_weights = strategy_weights / np.sum(strategy_weights)

    # Step 7: 分配樣本
    strategy_samples = {}
    total_allocated = 0
    min_samples_per_strategy = max(1, int(num_samples * 0.1))
    
    # 根據權重排序策略
    sorted_strategies = sorted(base_indices.keys(), 
                             key=lambda x: strategy_weights[list(base_indices.keys()).index(x)],
                             reverse=True)
    
    # 第一輪：分配最小樣本數給有效策略
    for strategy_name in sorted_strategies:
        strategy_idx = list(base_indices.keys()).index(strategy_name)
        if strategy_weights[strategy_idx] > 0:
            strategy_samples[strategy_name] = min_samples_per_strategy
            total_allocated += min_samples_per_strategy

    # 分配剩餘樣本
    remaining_samples = num_samples - total_allocated
    if remaining_samples > 0 and strategy_samples:
        valid_strategies = list(strategy_samples.keys())
        valid_weights = np.array([strategy_weights[list(base_indices.keys()).index(s)] 
                                for s in valid_strategies])
        valid_weights = valid_weights / np.sum(valid_weights)
        
        for i in range(len(valid_strategies)-1):
            strategy_name = valid_strategies[i]
            additional_samples = int(remaining_samples * valid_weights[i])
            strategy_samples[strategy_name] += additional_samples
            total_allocated += additional_samples
        
        final_remaining = num_samples - total_allocated
        last_strategy = valid_strategies[-1]
        strategy_samples[last_strategy] += final_remaining

    # Step 8: 生成最終樣本
    all_indices = []
    proportions = {}
    for strategy_name, n_samples in strategy_samples.items():
        if n_samples > 0:
            sampling_function = SAMPLING_METHODS.get(strategy_name)
            if sampling_function is None:
                raise KeyError(f"Sampling method {strategy_name} not found.")
            indices = sampling_function(embeddings, n_samples)
            if isinstance(indices, tuple):
                indices = indices[0]
            all_indices.extend(indices)
            proportions[strategy_name] = n_samples / num_samples

    # 確保樣本數量正確
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

def ridge_sample_selection_with_proportions(embeddings, num_samples, base_prop=0.3):
    n = embeddings.shape[0]
    """
    加強版 Ridge 特徵選擇策略，強化 MES 特徵對最終取樣比例的影響。
    
    主要步驟：
    1. 使用每個基礎策略產生初始樣本（較大量），並對整個數據集計算「到該策略樣本最小距離」作為特徵。
    2. 對特徵做標準化（StandardScaler）。
    3. 定義複合目標值（距中心距離 * 負密度）。
    4. 使用 GridSearchCV 對 Ridge 進行交叉驗證，取得最佳 alpha。
    5. 根據模型係數計算策略權重，並強化 MES 係數。
    6. 根據最終權重，按兩輪分配樣本（先分配最小數量，再按權重分配剩餘）。
    7. 產生最終選取的索引與各策略所佔比例。
    """
    
    # =====================
    # Step 1: 使用完整數據集產生基礎策略樣本
    # =====================
    base_indices = {
        "Random Seed (RS)": random_seed_sampling(embeddings, int(n * base_prop)),
        "Grid Sampling (GS)": grid_sampling(embeddings, int(n * base_prop)),
        "Max-Min Distance (MMDS)": max_min_distance_sampling(embeddings, int(n * base_prop)),
        "Density-based (DBS)": density_based_sampling(embeddings, int(n * base_prop)),
        "Max Entropy (MES)": max_entropy_sampling(embeddings, int(n * base_prop)),
        "Cluster Sampling (CS)": cluster_sampling(embeddings, int(n * base_prop)),
    }

    # =====================
    # Step 2: 建立「加強版」特徵矩陣
    #  - strategy_features[:, i] = 到第 i 個策略所選樣本之「最近距離」
    # =====================
    strategy_features = np.zeros((n, len(base_indices)))
    for i, (strategy_name, indices) in enumerate(base_indices.items()):
        selected_embeddings = embeddings[indices]
        # 計算每個點到該策略選取樣本的最近距離
        distances = np.min(
            np.linalg.norm(embeddings[:, np.newaxis] - selected_embeddings, axis=2),
            axis=1
        )
        strategy_features[:, i] = distances

    # =====================
    # Step 3: 特徵標準化
    # =====================
    scaler = StandardScaler()
    strategy_features = scaler.fit_transform(strategy_features)

    # =====================
    # Step 4: 定義「加強版」目標值
    #   - 距中心距離 (越遠可能越需代表性)
    #   - 負密度 (周圍越密集，越需要代表)
    #   => target = 距中心距離 * 負密度
    # =====================
    center = np.mean(embeddings, axis=0)
    distances_to_center = np.linalg.norm(embeddings - center, axis=1)
    
    # 使用最近鄰估計局部密度（此處簡單用距離和的負值代表）
    nn = NearestNeighbors(n_neighbors=min(5, len(embeddings)))
    nn.fit(embeddings)
    density_scores = -np.sum(nn.kneighbors(embeddings)[0], axis=1)
    
    target = distances_to_center * density_scores

    # =====================
    # Step 5: 使用 Ridge + GridSearchCV 進行交叉驗證
    # =====================
    param_grid = {'alpha': [0.0001, 0.001, 0.01, 0.1, 1.0]}
    ridge_cv = GridSearchCV(
        Ridge(random_state=42, max_iter=10000),
        param_grid,
        cv=5,
        scoring='neg_mean_squared_error'
    )
    ridge_cv.fit(strategy_features, target)
    model = ridge_cv.best_estimator_

    # =====================
    # Step 6: 根據 Ridge 係數計算策略權重
    #   - 先取絕對值
    #   - 做個閾值過濾（選擇性，可依實際需求調整）
    #   - 重新歸一化
    # =====================
    coef_threshold = np.std(model.coef_) * 0.1  # 係數標準差的 10% 作為閾值 (可調)
    strategy_weights = np.abs(model.coef_)
    strategy_weights[strategy_weights < coef_threshold] = 0.0

    if np.sum(strategy_weights) == 0:
        logging.warning("All Ridge coefficients are below threshold. Using uniform weights.")
        strategy_weights = np.ones(len(base_indices)) / len(base_indices)
    else:
        strategy_weights = strategy_weights / np.sum(strategy_weights)

    # =====================
    # Step 6.1: 強化 MES 特徵
    #   - 這裡簡單示範：對 MES 的權重再乘上一個倍數 (例如 1.2)
    #   - 調整後要重新歸一化
    # =====================
    mes_name = "Max Entropy (MES)"
    mes_index = list(base_indices.keys()).index(mes_name)
    # 你可以依需求調整此倍數，或改成固定加法等方式
    mes_boost_factor = 1.2
    strategy_weights[mes_index] *= mes_boost_factor
    strategy_weights /= np.sum(strategy_weights)  # 重新歸一化

    # =====================
    # Step 7: 根據最終權重分配樣本
    #   - 與原先的兩輪分配邏輯相同：
    #     先給「有效策略」每個最小樣本數，再依權重分配剩餘
    # =====================
    strategy_samples = {}
    total_allocated = 0
    min_samples_per_strategy = max(1, int(num_samples * 0.1))  # 最小樣本數（10%）
    
    # 根據權重排序策略（從大到小）
    sorted_strategies = sorted(
        base_indices.keys(),
        key=lambda x: strategy_weights[list(base_indices.keys()).index(x)],
        reverse=True
    )
    
    # 第一輪：分配最小樣本數給有效策略（權重大於 0）
    for strategy_name in sorted_strategies:
        strategy_idx = list(base_indices.keys()).index(strategy_name)
        if strategy_weights[strategy_idx] > 0:
            strategy_samples[strategy_name] = min_samples_per_strategy
            total_allocated += min_samples_per_strategy

    # 計算剩餘可分配樣本數
    remaining_samples = num_samples - total_allocated
    
    # 第二輪：根據權重比例分配剩餘樣本
    if remaining_samples > 0 and strategy_samples:
        valid_strategies = list(strategy_samples.keys())
        valid_weights = np.array([
            strategy_weights[list(base_indices.keys()).index(s)] 
            for s in valid_strategies
        ])
        valid_weights = valid_weights / np.sum(valid_weights)
        
        for i in range(len(valid_strategies) - 1):
            strategy_name = valid_strategies[i]
            additional_samples = int(remaining_samples * valid_weights[i])
            strategy_samples[strategy_name] += additional_samples
            total_allocated += additional_samples
        
        # 將最後的餘量給最後一個策略
        final_remaining = num_samples - total_allocated
        last_strategy = valid_strategies[-1]
        strategy_samples[last_strategy] += final_remaining

    # 驗證總樣本數
    assert sum(strategy_samples.values()) == num_samples, "Sample allocation error"

    # =====================
    # Step 8: 根據最終的分配量進行取樣，產出 indices 與 proportions
    # =====================
    all_indices = []
    proportions = {}
    for strategy_name, n_samples in strategy_samples.items():
        if n_samples > 0:
            sampling_function = SAMPLING_METHODS.get(strategy_name)
            if sampling_function is None:
                raise KeyError(f"Sampling method {strategy_name} not found.")
            indices = sampling_function(embeddings, n_samples)
            # 有些策略可能回傳 (indices, info)，因此需要判斷
            if isinstance(indices, tuple):
                indices = indices[0]
            all_indices.extend(indices)
            proportions[strategy_name] = n_samples / num_samples

    # 驗證比例總和
    if not math.isclose(sum(proportions.values()), 1.0, rel_tol=1e-9):
        logging.warning("Proportions do not sum to 1.0, got: %.5f", sum(proportions.values()))

    # 確保樣本數量正確（如果超過就隨機抽取）
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


def elastic_net_sample_selection_with_proportions(embeddings, num_samples, base_prop=0.3):
    n = embeddings.shape[0]
    """ElasticNet 特徵選擇策略，基於完整數據的 L1 和 L2 正則化動態調整策略比例，並加強 MES 權重。"""
    # =====================
    # Step 1: 產生基礎策略樣本
    # =====================
    base_indices = {
        "Random Seed (RS)": random_seed_sampling(embeddings, int(n * base_prop)),
        "Grid Sampling (GS)": grid_sampling(embeddings, int(n * base_prop)),
        "Max-Min Distance (MMDS)": max_min_distance_sampling(embeddings, int(n * base_prop)),
        "Density-based (DBS)": density_based_sampling(embeddings, int(n * base_prop)),
        "Max Entropy (MES)": max_entropy_sampling(embeddings, int(n * base_prop)),
        "Cluster Sampling (CS)": cluster_sampling(embeddings, int(n * base_prop)),
    }

    # =====================
    # Step 2: 建立特徵矩陣 (多個距離度量的組合)
    # =====================
    strategy_features = np.zeros((n, len(base_indices)))
    for i, (strategy_name, indices) in enumerate(base_indices.items()):
        selected_embeddings = embeddings[indices]
        # 計算多種距離度量
        distances = np.min(np.linalg.norm(embeddings[:, np.newaxis] - selected_embeddings, axis=2), axis=1)
        mean_distance = np.mean(np.linalg.norm(embeddings[:, np.newaxis] - selected_embeddings, axis=2), axis=1)
        max_distance = np.max(np.linalg.norm(embeddings[:, np.newaxis] - selected_embeddings, axis=2), axis=1)
        # 依不同權重組合
        strategy_features[:, i] = 0.4 * distances + 0.3 * mean_distance + 0.3 * max_distance

    # =====================
    # Step 3: 特徵標準化
    # =====================
    scaler = StandardScaler()
    strategy_features = scaler.fit_transform(strategy_features)

    # =====================
    # Step 4: 定義目標值 (結合距離中心 + 負密度 + 方差)
    # =====================
    center = np.mean(embeddings, axis=0)
    distances_to_center = np.linalg.norm(embeddings - center, axis=1)

    nn = NearestNeighbors(n_neighbors=min(5, len(embeddings)))
    nn.fit(embeddings)
    density_scores = -np.sum(nn.kneighbors(embeddings)[0], axis=1)

    variance = np.var(embeddings, axis=1)

    target = distances_to_center * density_scores * variance
    # 為了讓訓練更穩定，對 target 做一次標準化
    target = (target - np.mean(target)) / (np.std(target) + 1e-9)

    # =====================
    # Step 5: 使用 ElasticNet + GridSearchCV 調參
    # =====================
    param_grid = {
        'alpha': [0.00001, 0.0001, 0.001],  # 更低的 alpha 範圍
        'l1_ratio': [0.1, 0.25, 0.5, 0.75, 0.9]
    }
    elastic_net_cv = GridSearchCV(
        ElasticNet(random_state=42, max_iter=10000, tol=1e-4),
        param_grid,
        cv=5,
        scoring='neg_mean_squared_error'
    )
    elastic_net_cv.fit(strategy_features, target)
    
    logging.info(f"Best ElasticNet parameters: {elastic_net_cv.best_params_}")
    model = elastic_net_cv.best_estimator_

    # =====================
    # Step 6: 計算稀疏權重 (基於模型絕對係數)
    # =====================
    coef_abs = np.abs(model.coef_)
    # 自適應閾值
    coef_threshold = np.mean(coef_abs) - 0.5 * np.std(coef_abs)
    strategy_weights = coef_abs.copy()
    strategy_weights[strategy_weights < coef_threshold] = 0.0

    if np.sum(strategy_weights) == 0:
        # 如果全部被清空，就改用原始係數比例
        strategy_weights = coef_abs.copy()

    # =====================
    # Step 6.1: 強化 MES 特徵
    #   - 對 MES 的權重乘上一個倍數，然後重新歸一化
    # =====================
    mes_name = "Max Entropy (MES)"
    mes_index = list(base_indices.keys()).index(mes_name)
    mes_boost_factor = 1.2  # 你可以依需求調整此倍數
    strategy_weights[mes_index] *= mes_boost_factor

    # 最後做一次歸一化
    strategy_weights_sum = np.sum(strategy_weights)
    if strategy_weights_sum > 0:
        strategy_weights /= strategy_weights_sum
    else:
        # 理論上不應該走到這裡，但若權重全為 0，就平均分配
        strategy_weights = np.ones(len(base_indices)) / len(base_indices)

    for name, weight in zip(base_indices.keys(), strategy_weights):
        logging.info(f"{name}: {weight:.4f}")

    # =====================
    # Step 7: 根據最終權重分配樣本 (兩輪分配)
    # =====================
    strategy_samples = {}
    total_allocated = 0
    min_samples_per_strategy = max(1, int(num_samples * 0.1))

    # 依照權重排序
    sorted_strategies = sorted(
        base_indices.keys(),
        key=lambda x: strategy_weights[list(base_indices.keys()).index(x)],
        reverse=True
    )

    # 第一輪：給每個有權重的策略一個最小量
    for strategy_name in sorted_strategies:
        strategy_idx = list(base_indices.keys()).index(strategy_name)
        if strategy_weights[strategy_idx] > 0:
            strategy_samples[strategy_name] = min_samples_per_strategy
            total_allocated += min_samples_per_strategy
            logging.info(f"Initial allocation for {strategy_name}: {min_samples_per_strategy}")

    # 第二輪：剩餘樣本依權重分配
    remaining_samples = num_samples - total_allocated
    logging.info(f"Remaining samples after initial allocation: {remaining_samples}")

    if remaining_samples > 0 and strategy_samples:
        valid_strategies = list(strategy_samples.keys())
        valid_weights = np.array([
            strategy_weights[list(base_indices.keys()).index(s)]
            for s in valid_strategies
        ])
        valid_weights /= (np.sum(valid_weights) + 1e-9)

        for i in range(len(valid_strategies) - 1):
            strategy_name = valid_strategies[i]
            additional_samples = int(remaining_samples * valid_weights[i])
            strategy_samples[strategy_name] += additional_samples
            total_allocated += additional_samples
            logging.info(f"Additional allocation for {strategy_name}: {additional_samples}")

        # 最後餘量都給最後一個
        final_remaining = num_samples - total_allocated
        last_strategy = valid_strategies[-1]
        strategy_samples[last_strategy] += final_remaining
        logging.info(f"Final allocation for {last_strategy}: {final_remaining}")

    # 驗證總數量
    total_samples = sum(strategy_samples.values())
    assert total_samples == num_samples, f"Sample allocation error: got {total_samples}, expected {num_samples}"

    # =====================
    # Step 8: 產生最終樣本
    # =====================
    all_indices = []
    proportions = {}
    for strategy_name, n_samples in strategy_samples.items():
        if n_samples > 0:
            sampling_function = SAMPLING_METHODS.get(strategy_name)
            if sampling_function is None:
                raise KeyError(f"Sampling method {strategy_name} not found.")
            indices = sampling_function(embeddings, n_samples)
            if isinstance(indices, tuple):
                indices = indices[0]
            all_indices.extend(indices)
            proportions[strategy_name] = n_samples / num_samples
            logging.info(f"Final proportion for {strategy_name}: {proportions[strategy_name]:.4f}")

    # 確認比例之和
    total_proportion = sum(proportions.values())
    assert math.isclose(total_proportion, 1.0, rel_tol=1e-9), \
        f"Proportion error: got {total_proportion}, expected 1.0"

    # 若最終索引超量，就隨機抽取；否則若不足就補抽
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


def equal_proportion_sampling(embeddings, num_samples):
    """
    均等分配每個策略的樣本數量。
    """
    num_per_strategy = num_samples // 6  # 平均分為 6 部分
    all_indices = []

    proportions = {}  # 初始化比例字典

    # 使用每個策略選取樣本
    for strategy_name in ["Random Seed (RS)", "Grid Sampling (GS)", 
                          "Max-Min Distance (MMDS)", "Density-based (DBS)", 
                          "Max Entropy (MES)", "Cluster Sampling (CS)"]:
        sampling_function = SAMPLING_METHODS.get(strategy_name)
        if sampling_function is None:
            raise KeyError(f"Sampling method {strategy_name} not found.")
        
        # 為每個策略選取樣本
        indices, _ = sampling_function(embeddings, num_per_strategy)
        all_indices.extend(indices)

        # 設定該策略的比例
        proportions[strategy_name] = num_per_strategy / num_samples

    # 去重並檢查樣本是否足夠
    all_indices = np.unique(all_indices)
    if len(all_indices) > num_samples:
        selected_indices = np.random.choice(all_indices, num_samples, replace=False)
    elif len(all_indices) < num_samples:
        additional_indices = random_seed_sampling(embeddings, num_samples - len(all_indices))
        selected_indices = np.concatenate([all_indices, additional_indices])
    else:
        selected_indices = all_indices

    # 確保總比例為1
    total_allocated = sum(proportions.values())
    for strategy_name in proportions:
        proportions[strategy_name] /= total_allocated

    return selected_indices, proportions



SAMPLING_METHODS = {
    "Lasso Selection (LS)_20": partial(lasso_sample_selection_with_proportions, base_prop=0.2),
    "Lasso Selection (LS)_30": partial(lasso_sample_selection_with_proportions, base_prop=0.3),
    "Lasso Selection (LS)_40": partial(lasso_sample_selection_with_proportions, base_prop=0.4),

    "Ridge Selection (RidgeS)_20": partial(ridge_sample_selection_with_proportions, base_prop=0.2),
    "Ridge Selection (RidgeS)_30": partial(ridge_sample_selection_with_proportions, base_prop=0.3),
    "Ridge Selection (RidgeS)_40": partial(ridge_sample_selection_with_proportions, base_prop=0.4),

    "Elastic Net (EN)_20": partial(elastic_net_sample_selection_with_proportions, base_prop=0.2),
    "Elastic Net (EN)_30": partial(elastic_net_sample_selection_with_proportions, base_prop=0.3),
    "Elastic Net (EN)_40": partial(elastic_net_sample_selection_with_proportions, base_prop=0.4),

    "Random Forest (RF)_20": partial(random_forest_combination_with_proportions, base_prop=0.2),
    "Random Forest (RF)_30": partial(random_forest_combination_with_proportions, base_prop=0.3),
    "Random Forest (RF)_40": partial(random_forest_combination_with_proportions, base_prop=0.4),

    "Random Seed (RS)": random_seed_sampling_with_proportions,
    "Grid Sampling (GS)": grid_sampling_with_proportions,
    "Max-Min Distance (MMDS)": max_min_distance_sampling_with_proportions,
    "Density-based (DBS)": density_based_sampling_with_proportions,
    "Max Entropy (MES)": max_entropy_sampling_with_proportions,
    "Cluster Sampling (CS)": cluster_sampling_with_proportions,
    "Equal Proportion Sampling (EPS)": equal_proportion_sampling
}




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
    使用匈牙利演算法 (Hungarian algorithm) 計算多對多 partial match 的 F1 分數。
    
    predictions, references: List of List of dict
    task_key: str, 要比對的欄位，例如 "span", "polarity", "opinion_term"
    use_lcs: 是否使用 LCS 計算相似度（用於提取任務）
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

    # 最後的 Precision 和 Recall
    avg_precision = total_precision / n
    avg_recall = total_recall / n

    # 使用 (2 * P * R) / (P + R) 計算最終 F1
    if (avg_precision + avg_recall) > 0:
        final_f1 = 2 * avg_precision * avg_recall / (avg_precision + avg_recall)
    else:
        final_f1 = 0.0

    return {
        "precision": avg_precision,
        "recall": avg_recall,
        "f1": final_f1,
    }


def allocate_samples(strategy_weights, num_samples):
    """
    基於權重分配樣本數，確保總和為 num_samples
    
    Args:
        strategy_weights: 各策略的權重
        num_samples: 總樣本數
    """
    # 首先標準化權重確保總和為1
    normalized_weights = strategy_weights / np.sum(strategy_weights)
    
    # 計算每個策略的理論樣本數（可能有小數）
    theoretical_samples = normalized_weights * num_samples
    
    # 首先分配整數部分
    allocated_samples = np.floor(theoretical_samples).astype(int)
    remaining_samples = num_samples - np.sum(allocated_samples)
    
    # 對剩餘樣本，根據小數部分大小排序分配
    decimal_parts = theoretical_samples - allocated_samples
    if remaining_samples > 0:
        # 取前 remaining_samples 個最大的小數部分對應的索引
        indices = np.argsort(decimal_parts)[-int(remaining_samples):]
        allocated_samples[indices] += 1
    
    return allocated_samples

class CheckpointManager:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.checkpoint_file = os.path.join(output_dir, "checkpoint.json")
        
    def save_checkpoint(self, sampling_method, current_fold):
        checkpoint = self.load_checkpoint() or {"completed_methods": {}}

        if sampling_method in checkpoint["completed_methods"]:
            completed_status = checkpoint["completed_methods"][sampling_method].get("completed", False)
        else:
            completed_status = False

        checkpoint["completed_methods"][sampling_method] = {
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
        
    def mark_completed(self, sampling_method):
        # 讀取現有的 checkpoint
        checkpoint = self.load_checkpoint() or {"completed_methods": {}}
        
        # 標記當前方法為已完成
        if sampling_method in checkpoint["completed_methods"]:
            checkpoint["completed_methods"][sampling_method]["completed"] = True
        else:
            checkpoint["completed_methods"][sampling_method] = {
                "current_fold": 5,
                "completed": True
            }
        
        # 保存更新後的 checkpoint
        with open(self.checkpoint_file, "w") as f:
            json.dump(checkpoint, f, indent=2)


class Experiment:
    def __init__(self, sampling_name="Random Seed"):
        self.sampling_name = sampling_name
        self.sampling_func = SAMPLING_METHODS[sampling_name]
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.models = {}
        self.nlp = spacy.load("en_core_web_sm")
        self.fold_metrics = []
        # 初始化 SentenceTransformer
        self.model = SentenceTransformer(MODEL_NAME)

    def load_data(self):
        """只加載原始數據，不進行採樣"""
        logging.info(f"Loading datasets...")
        dataset = load_dataset("JaquanTW/fewshot-absaquad")
        self.full_dataset = dataset["train"]  # 保存完整訓練集
        self.test_dataset = dataset["test"]
        logging.info(f"Full training dataset size: {len(self.full_dataset)}")
        logging.info(f"Test dataset size: {len(self.test_dataset)}")

    def sample_fold_data(self, fold_data):
        """對每個 fold 的訓練數據進行採樣"""
        logging.info(f"Applying sampling strategy: {self.sampling_name}")
        
        # 生成當前 fold 的 embeddings
        texts = [x["text"] for x in fold_data]
        embeddings = self.model.encode(texts, show_progress_bar=True)
        
        # 應用採樣策略
        if self.sampling_name in ["Lasso Selection (LS)", "Elastic Net (EN)", 
                                "Random Forest (RF)", "Ridge Selection (RidgeS)"]:
            sampled_indices, proportions = self.sampling_func(embeddings, TRAIN_SIZE)
        else:
            result = self.sampling_func(embeddings, TRAIN_SIZE)
            if isinstance(result, tuple):
                sampled_indices, proportions = result
            else:
                sampled_indices = result
                proportions = {self.sampling_name: 1.0}

        # 處理採樣索引
        if isinstance(sampled_indices, np.ndarray):
            sampled_indices = sampled_indices.flatten().astype(int).tolist()
        elif isinstance(sampled_indices, list):
            sampled_indices = [int(i) for i in sampled_indices]
        elif isinstance(sampled_indices, tuple):
            sampled_indices = [int(i) for i in np.ravel(sampled_indices)]

        # 返回採樣後的數據和比例
        sampled_data = fold_data.select(sampled_indices)
        logging.info(f"Sampled data size: {len(sampled_data)}")
        return sampled_data, proportions
    
    def train_and_evaluate_fold(self, train_data, val_data, fold_num):
        """在單個fold上訓練和評估模型"""
        logging.info(f"Training and evaluating fold {fold_num + 1}/{N_FOLDS}")
        
        # 設置當前fold的數據
        self.train_dataset = train_data
        self.val_dataset = val_data
        
        # 訓練模型
        self.train_ac_model(fold_num)
        self.train_label_model(fold_num)
        self.train_absa_model(fold_num)
        
        # 評估並返回結果
        metrics = self.evaluate(fold_num)
        return metrics

    def train_ac_model(self, fold_num):
        """Train Aspect Category model."""
        logging.info(f"Training AC model for fold {fold_num + 1}...")
        train_args = TrainingArguments(
            output_dir=f"{OUTPUT_DIR}/fold_{fold_num + 1}/ac",
            num_epochs=5,
            batch_size=8,
            body_learning_rate=2e-5,
            head_learning_rate=1e-3,
        )
        unique_ac_labels = list(set(self.train_dataset["ac"]))
        ac_model = SetFitModel.from_pretrained(MODEL_NAME, labels=unique_ac_labels)

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
        ac_model.save_pretrained(f"{OUTPUT_DIR}/fold_{fold_num + 1}/ac_model")

    def train_label_model(self, fold_num):
        """Train Polarity model."""
        logging.info(f"Training Polarity model for fold {fold_num + 1}...")
        train_args = TrainingArguments(
            output_dir=f"{OUTPUT_DIR}/fold_{fold_num + 1}/label",
            num_epochs=5,
            batch_size=8,
            body_learning_rate=2e-5,
            head_learning_rate=1e-3,
        )
        unique_labels = list(set(self.train_dataset["label"]))
        label_model = SetFitModel.from_pretrained(MODEL_NAME, labels=unique_labels)

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
        label_model.save_pretrained(f"{OUTPUT_DIR}/fold_{fold_num + 1}/label_model")

    def train_absa_model(self, fold_num):
        """Train ABSA model."""
        logging.info(f"Training ABSA model for fold {fold_num + 1}...")
        absa_model = AbsaModel.from_pretrained(
            model_id=MODEL_NAME,
            spacy_model="en_core_web_sm",
        )
        train_args = TrainingArguments(
            output_dir=f"{OUTPUT_DIR}/fold_{fold_num + 1}/absa",
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
        absa_model.save_pretrained(f"{OUTPUT_DIR}/fold_{fold_num + 1}/absa_model")




    def extract_opinion_terms(self, text, aspect):
        """
        改進的 Opinion Term 提取方法，基於依存句法和上下文窗口。
        """
        doc = self.nlp(text)
        aspect_token = None

        # 找到 aspect 對應的 token
        for token in doc:
            if token.text.lower() == aspect.lower():
                aspect_token = token
                break

        if not aspect_token:
            return []

        # 提取與 aspect 關聯的 opinion terms
        opinion_terms = []
        for token in doc:
            if (token.dep_ in {"amod", "advmod"} and token.head == aspect_token) or \
            (token.head == aspect_token and token.pos_ in {"ADJ", "ADV"}):
                opinion_terms.append(token.text)

        # 加入上下文窗口的詞
        window_size = 3  # 可調參數
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


    def calculate_average_metrics(self):
        """計算所有fold的平均指標"""
        avg_metrics = {
            "AC Metrics": {"Precision": 0.0, "Recall": 0.0, "F1": 0.0},
            "Polarity Metrics": {"Precision": 0.0, "Recall": 0.0, "F1": 0.0},
            "Span Metrics": {"precision": 0.0, "recall": 0.0, "f1": 0.0},
            "Opinion Term Metrics": {"precision": 0.0, "recall": 0.0, "f1": 0.0}
        }

        # 計算平均值
        for metrics in self.fold_metrics:
            # AC and Polarity metrics
            for metric_type in ["AC Metrics", "Polarity Metrics"]:
                for key in ["Precision", "Recall", "F1"]:
                    avg_metrics[metric_type][key] += metrics[metric_type][key]

            # Span and Opinion Term metrics
            for metric_type in ["Span Metrics", "Opinion Term Metrics"]:
                for key in ["precision", "recall", "f1"]:
                    avg_metrics[metric_type][key] += metrics[metric_type][key]

        # Calculate averages
        n_folds = len(self.fold_metrics)
        for metric_type in avg_metrics:
            for key in avg_metrics[metric_type]:
                avg_metrics[metric_type][key] /= n_folds

        return avg_metrics

    def run(self):
        """執行完整的實驗流程，包含斷點恢復機制"""
        checkpoint_manager = CheckpointManager(OUTPUT_DIR)
        checkpoint = checkpoint_manager.load_checkpoint()

        if (checkpoint and "completed_methods" in checkpoint and 
            self.sampling_name in checkpoint["completed_methods"] and
            checkpoint["completed_methods"][self.sampling_name].get("completed", False)):
            logging.info(f"{self.sampling_name} already completed. Skipping.")
            return
        
        self.load_data()
        
        # 創建 KFold 對象
        kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
        dataset_indices = list(range(len(self.full_dataset)))
        
        # 確定起始 fold
        start_fold = 0
        if checkpoint and "completed_methods" in checkpoint and self.sampling_name in checkpoint["completed_methods"]:
            start_fold = checkpoint["completed_methods"][self.sampling_name]["current_fold"]
            logging.info(f"Resuming from fold {start_fold + 1}")
        
        # 執行剩餘的 folds
        for fold_num, (train_idx, val_idx) in enumerate(kf.split(dataset_indices)):
            if fold_num < start_fold:
                continue
                
            logging.info(f"\nStarting fold {fold_num + 1}/{N_FOLDS}")
            checkpoint_manager.save_checkpoint(self.sampling_name, fold_num)
            
            # 分割數據
            train_fold = self.full_dataset.select(train_idx.tolist())
            val_fold = self.full_dataset.select(val_idx.tolist())
            
            # 只對訓練集進行採樣
            self.train_dataset, self.sample_proportions = self.sample_fold_data(train_fold)
            # 驗證集保持原樣
            self.val_dataset = val_fold
            
            logging.info(f"Training set size after sampling: {len(self.train_dataset)}")
            logging.info(f"Validation set size: {len(self.val_dataset)}")
            
            try:
                self.train_ac_model(fold_num)
                self.train_label_model(fold_num)
                self.train_absa_model(fold_num)
                
                fold_metrics = self.evaluate(fold_num)
                self.fold_metrics.append(fold_metrics)
                save_results(OUTPUT_DIR, self.sampling_name, fold_metrics, fold_num)
                
            except Exception as e:
                logging.error(f"Error in fold {fold_num + 1}: {str(e)}")
                raise e
            
            logging.info(f"Completed fold {fold_num + 1}/{N_FOLDS}")
        
        # 計算和保存最終結果
        avg_metrics = self.calculate_average_metrics()
        avg_metrics["Sample_Proportions"] = self.sample_proportions
        save_results(OUTPUT_DIR, f"{self.sampling_name}_5fold_average", avg_metrics)
        
        test_metrics = self.evaluate()
        save_results(OUTPUT_DIR, f"{self.sampling_name}_test", test_metrics)
        
        checkpoint_manager.mark_completed(self.sampling_name)
        logging.info("Experiment completed successfully")

        
if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 從 checkpoint 讀取上次中斷的方法
    checkpoint_manager = CheckpointManager(OUTPUT_DIR)
    checkpoint = checkpoint_manager.load_checkpoint()
    
    # 檢查是否有未完成的方法
    if checkpoint and "completed_methods" in checkpoint:
        uncompleted_methods = [
            method for method, status in checkpoint["completed_methods"].items() 
            if not status["completed"]
        ]
        if uncompleted_methods:
            logging.info(f"Found uncompleted methods: {uncompleted_methods}")
            start_method = uncompleted_methods[0]
        else:
            logging.info("All methods completed. Exiting.")
            exit()
    else:
        start_method = None


    # 執行所有抽樣方法
    methods_to_run = list(SAMPLING_METHODS.keys())
    if start_method:
        # 如果有未完成的方法，從那個方法開始
        start_idx = methods_to_run.index(start_method)
        methods_to_run = methods_to_run[start_idx:]
        
    for sampling_name in methods_to_run:
        logging.info(f"Running experiment with sampling method: {sampling_name}")
        experiment = Experiment(sampling_name=sampling_name)
        experiment.run()