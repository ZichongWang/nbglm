# evaluate_predictions.py (版本 3.1 - 性能优化与标准确认)
import anndata
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.spatial.distance import cdist
import argparse
import warnings
import os
import pickle
from joblib import Parallel, delayed
from tqdm import tqdm
import time

# --- 全局设置 ---
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=anndata.ImplicitModificationWarning)

# --- 辅助函数 ---

def normalize_if_needed(adata: anndata.AnnData, label: str):
    is_raw_counts = (np.issubdtype(adata.X.dtype, np.integer) and adata.X.max() > 20)
    if True:
        print(f"检测到 '{label}' 数据为原始计数。正在执行标准化和 log1p 转换...")
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
    return adata

def get_mean_profiles(adata, pert_genes, control_id):
    profiles = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        control_mean_result = adata[adata.obs['target_gene'] == control_id].X.mean(axis=0)
        profiles[control_id] = np.asarray(control_mean_result).flatten()
        for gene in pert_genes:
            pert_mean_result = adata[adata.obs['target_gene'] == gene].X.mean(axis=0)
            profiles[gene] = np.asarray(pert_mean_result).flatten()
    return pd.DataFrame(profiles, index=adata.var_names).T

def calculate_mae(pred_profiles, true_profiles, pert_genes):
    mae_scores = [np.mean(np.abs(pred_profiles.loc[g] - true_profiles.loc[g])) for g in pert_genes]
    return np.mean(mae_scores) if mae_scores else 0.0

def calculate_pds(pred_profiles, true_profiles, pert_genes):
    pred_pert_profiles = pred_profiles.loc[pert_genes]
    true_pert_profiles = true_profiles.loc[pert_genes]
    N = len(pert_genes)
    if N == 0: return 0.0

    dist_matrix = cdist(pred_pert_profiles.values, true_pert_profiles.values, metric='cityblock')
    
    for i, p_gene in enumerate(pert_genes):
        correction_vector = np.abs(pred_pert_profiles.at[p_gene, p_gene] - true_pert_profiles[p_gene].values)
        dist_matrix[i, :] -= correction_vector

    dist_df = pd.DataFrame(dist_matrix, index=pert_genes, columns=pert_genes)
    ranks = [np.where(dist_df.loc[p].sort_values().index == p)[0][0] + 1 for p in pert_genes]
    pds_scores = [1 - (rank - 1) / N for rank in ranks]
    return np.mean(pds_scores) if pds_scores else 0.0

def _get_de_genes_from_slice(adata_slice, pert_gene, control_id):
    """(内部函数) 从数据切片中计算单个扰动的DE基因"""
    adata_local = adata_slice.copy()
    try:
        # 使用 Wilcoxon rank-sum test。Scanpy 会自动使用 Benjamini-Hochberg 校正 p 值，得到 pvals_adj。
        sc.tl.rank_genes_groups(adata_local, groupby='target_gene', groups=[pert_gene], reference=control_id, method='wilcoxon', use_raw=False, n_genes=adata_local.shape[1])
        res = adata_local.uns['rank_genes_groups']
        de_df = pd.DataFrame({
            'names': res['names'][pert_gene],
            'pvals_adj': res['pvals_adj'][pert_gene],
            'logfoldchanges': res['logfoldchanges'][pert_gene]
        })
        de_df.to_csv("dedf.csv")
        de_genes_df = de_df[de_df['pvals_adj'] < 0.05]
        return de_genes_df
    except Exception as e:
        print(f"警告: 在为 {pert_gene} 计算DE基因时出错: {e}。将返回空结果。")
        return pd.DataFrame(columns=['names', 'pvals_adj', 'logfoldchanges'])

def get_ground_truth_de_genes_cached(true_adata, pert_genes, control_id, n_cores, cache_path):
    """
    计算或加载缓存的真实DE基因结果。
    """
    cache_path = "/home/wzc26/work/vcc/NB/true_de_cache.pkl"
    if cache_path and os.path.exists(cache_path):
        print(f"发现缓存文件，正在加载: {cache_path}")
        with open(cache_path, 'rb') as f:
            return pickle.load(f)

    print(f"未找到缓存。正在为 {len(pert_genes)} 个扰动基因计算真实DE基因 (此过程仅需一次)...")
    tasks = [delayed(_get_de_genes_from_slice)(
        true_adata[(true_adata.obs['target_gene'] == gene) | (true_adata.obs['target_gene'] == control_id)],
        gene, control_id
    ) for gene in pert_genes]
    
    de_results_list = Parallel(n_jobs=n_cores)(tqdm(tasks, desc="计算真实DE基因"))
    
    true_de_map = dict(zip(pert_genes, de_results_list))
    print(f"cache_path {cache_path}")
    if cache_path:
        print(f"正在保存DE结果到缓存: {cache_path}")
        with open(cache_path, 'wb') as f:
            pickle.dump(true_de_map, f)
            
    return true_de_map

def calculate_des_parallel(pred_adata, true_de_map, pert_genes, control_id, n_cores):
    """
    使用并行处理，将预测DE基因与缓存的真实DE基因进行比较。
    """
    print(f"正在使用 {n_cores} 个CPU核心并行计算 DES...")

    def _compare_des_for_one_gene(pert_gene):
        true_de_df = true_de_map[pert_gene]
        true_de_genes = set(true_de_df['names'])
        n_k_true = len(true_de_genes)

        if n_k_true == 0:
            return 1.0

        pred_adata_slice = pred_adata[(pred_adata.obs['target_gene'] == pert_gene) | (pred_adata.obs['target_gene'] == control_id)]
        full_pred_de_df = _get_de_genes_from_slice(pred_adata_slice, pert_gene, control_id)
        
        if len(full_pred_de_df) > n_k_true:
            top_pred_de_df = full_pred_de_df.reindex(full_pred_de_df.logfoldchanges.abs().sort_values(ascending=False).index)
            final_pred_genes = set(top_pred_de_df.head(n_k_true)['names'])
        else:
            final_pred_genes = set(full_pred_de_df['names'])
            
        intersection_size = len(final_pred_genes.intersection(true_de_genes))
        return intersection_size / n_k_true

    tasks = [delayed(_compare_des_for_one_gene)(gene) for gene in pert_genes]
    des_scores = Parallel(n_jobs=n_cores)(tqdm(tasks, desc="计算DES分数"))
    
    return np.mean(des_scores) if des_scores else 0.0

# --- 主逻辑 ---

def main(args):
    """主评估函数"""
    print("--- 开始评估 (版本 3.1 - 带缓存优化) ---")
    start = time.time()
    print(f"加载预测数据: {args.prediction}")
    print(f"加载真实数据: {args.ground_truth}")
    pred_adata = anndata.read_h5ad(args.prediction)
    true_adata = anndata.read_h5ad(args.ground_truth)
    
    if not all(pred_adata.var_names == true_adata.var_names):
        print("警告: 预测文件和真实文件的基因顺序不一致。正在对齐...")
        common_genes = pred_adata.var_names.intersection(true_adata.var_names)
        pred_adata = pred_adata[:, common_genes].copy()
        true_adata = true_adata[:, common_genes].copy()
    
    pred_adata = normalize_if_needed(pred_adata, "预测")
    true_adata = normalize_if_needed(true_adata, "真实")

    control_id = 'non-targeting'
    pert_genes = sorted([p for p in true_adata.obs['target_gene'].unique() if p != control_id])
    print(f"\n将对 {len(pert_genes)} 个扰动基因进行评估。")

    n_cores = max(1, int(os.cpu_count() * 0.8))

    # --- MAE 和 PDS 计算 ---
    print("\n正在计算 MAE 和 PDS...")
    pred_profiles = get_mean_profiles(pred_adata, pert_genes, control_id)
    true_profiles = get_mean_profiles(true_adata, pert_genes, control_id)
    mae = calculate_mae(pred_profiles, true_profiles, pert_genes)
    pds = calculate_pds(pred_profiles, true_profiles, pert_genes)
    print(f"  - PDS (扰动区分分数): {pds:.6f}")
    print(f"  - MAE (平均绝对误差): {mae:.6f}")
    # --- DES 计算 (使用缓存) ---
    dir_name = os.path.dirname(args.prediction)
    cache_path = os.path.join(dir_name, "true_de_cache.pkl")
    true_de_map = get_ground_truth_de_genes_cached(true_adata, pert_genes, control_id, n_cores, cache_path)
    des = calculate_des_parallel(pred_adata, true_de_map, pert_genes, control_id, n_cores)
    

    
    print("\n--- 评估完成 ---\n")
    print(f"最终分数:")
    print(f"  - DES (差异表达分数): {des:.6f}")
    print(f"  - PDS (扰动区分分数): {pds:.6f}")
    print(f"  - MAE (平均绝对误差): {mae:.6f}")
    des_baseline = 0.106
    pds_baseline = 0.516
    mae_baseline = 0.027
    overall = (des - des_baseline) / (1 - des_baseline) + (pds - pds_baseline) / (1 - pds_baseline) + ((mae_baseline - mae) / mae_baseline).clip(0, 1)
    overall *= 100/3

    print(f"\n综合得分 (基于 DES, PDS and MAE): {overall:.6f}")

    overall_1 = (des - des_baseline) / (1 - des_baseline) + (pds - pds_baseline) / (1 - pds_baseline)
    overall_1 *= 100/3
    print(f"\n综合得分 (仅基于 DES and PDS): {overall_1:.6f}")

    end = time.time()
    print(f"\n总评估时间: {end - start:.2f} 秒")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="评估扰动预测模型的性能 (版本 3.1 - 带缓存优化)。")
    parser.add_argument('--prediction', type=str, required=True, help="模型预测结果的 h5ad 文件路径。")
    parser.add_argument(
        '--ground_truth', 
        type=str, 
        default="/home/wzc26/work/vcc/vcc_data/Official_Data_Split/test.h5ad",
        help="真实的 h5ad 文件路径。默认为: /home/wzc26/work/vcc/vcc_data/Official_Data_Split/test.h5ad"
    )
    
    args = parser.parse_args()
    main(args)