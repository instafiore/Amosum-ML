import pandas as pd
import numpy as np
import joblib
import subprocess
import time
import os
import warnings
import argparse
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings('ignore')

# ==========================================
# 0. FOLD-R++ WRAPPER (FOR JOBLIB DESERIALIZATION)
# ==========================================
try:
    from foldrpp import Foldrpp
except ImportError:
    pass

class FoldRPPCascade:
    def __init__(self, ratio=0.5): 
        self.ratio = ratio
        self.models = {}      
        self.label_encoder = LabelEncoder()
        self.classes_ = None
        self.feature_names_in_ = None
        self.majority_class_ = None

    def _prepare_feature_lists(self, X):
        str_attrs, num_attrs = [], []
        for col in X.columns:
            if X[col].dtype == 'object' or X[col].dtype.name == 'category':
                str_attrs.append(col)
            else:
                num_attrs.append(col)
        return str_attrs, num_attrs

    def fit(self, X, y): pass

    def predict(self, X):
        if not isinstance(X, pd.DataFrame): X = pd.DataFrame(X)
        predictions = []
        for idx in range(X.shape[0]):
            sample = X.iloc[[idx]].copy()
            sample['label'] = 0 
            sample_record = sample.to_dict(orient='records')
            
            scores = {}
            for class_name, model in self.models.items():
                pred = model.predict(sample_record)[0]
                scores[class_name] = 1 if str(pred) in ['1', '1.0', 'True'] else 0
            
            if sum(scores.values()) == 0:
                predicted_class = getattr(self, 'majority_class_', getattr(self, 'classes_', [0])[0])
            else:
                predicted_class = max(scores, key=scores.get)
            predictions.append(predicted_class)
        return np.array(predictions)

    def predict_proba(self, X):
        if not isinstance(X, pd.DataFrame): X = pd.DataFrame(X)
        proba = np.zeros((X.shape[0], len(self.classes_)))
        for idx in range(X.shape[0]):
            sample = X.iloc[[idx]].copy()
            sample['label'] = 0 
            sample_record = sample.to_dict(orient='records')
            for i, class_name in enumerate(self.classes_):
                if class_name in self.models:
                    model = self.models[class_name]
                    pred = model.predict(sample_record)[0]
                    if str(pred) in ['1', '1.0', 'True']:
                        proba[idx, i] = 1.0
        
        row_sums = proba.sum(axis=1)
        proba[row_sums > 0] = proba[row_sums > 0] / row_sums[row_sums > 0, np.newaxis]
        return proba


# ==========================================
# 1. GLOBAL PATHS AND CONFIGURATIONS
# ==========================================
CURR_DIR = os.path.dirname(os.path.abspath(__file__))
# Go back one directory level
BASE_DIR = os.path.join(CURR_DIR, '..')
TIMEOUT_SEC = 1200

FINAL_TEST = {
    'AMO': f'{CURR_DIR}/test_set_AMO_20perc.csv',
    'EO':  f'{CURR_DIR}/test_set_EO_20perc.csv'
}

# Original datasets (used to retrieve min_time)
ORIGINALS = {
    'AMO': [f'{BASE_DIR}/dataset_builder/preprocessed/training_dataset_amo.csv',
            f'{BASE_DIR}/dataset_builder/preprocessed/ijcai_dataset_amo.csv'],
    'EO':  [f'{BASE_DIR}/dataset_builder/preprocessed/training_dataset_eo.csv',
            f'{BASE_DIR}/dataset_builder/preprocessed/ijcai_dataset_eo.csv']
}

MAP_LABELS = {
    'AMO': {0: 'AMOCLINGO-INF-MR', 1: 'AMOCLINGO-L', 2: 'CLINGO-AMO'},
    'EO':  {0: 'CLINGO-AMO', 1: 'EOCLINGO-INF-MR', 2: 'EOCLINGO-L'}
}

# Experiment mapping
EXPERIMENT_CONFIG = {
    1: {"name": "Standalone XGBoost", "out_dir": "esperimento_1", "out_file": "final_validation_detailed",
        "AMO": f'{BASE_DIR}/selettori/modello_AMO_xgboost.joblib',
        "EO": f'{BASE_DIR}/selettori/modello_EO_xgboost.joblib'},
    2: {"name": "Simple Cascade (Threshold 0.75)", "out_dir": "esperimento_2_s75", "out_file": "final_validation_cascade",
        "AMO": f'{BASE_DIR}/selettori/modello_cascata_AMO.joblib',
        "EO": f'{BASE_DIR}/selettori/modello_cascata_EO.joblib'},
    3: {"name": "Optimized Cascade", "out_dir": "esperimento_3", "out_file": "final_validation_cascade",
        "AMO": f'{BASE_DIR}/selettori/modello_cascata_AMO_ottimizzato.joblib',
        "EO": f'{BASE_DIR}/selettori/modello_cascata_EO_ottimizzato.joblib'},
    4: {"name": "FOLD-R++ Cascade", "out_dir": "esperimento_4", "out_file": "final_validation_cascade_fold",
        "AMO": f'{BASE_DIR}/selettori/modello_cascata_fold_AMO.joblib',
        "EO": f'{BASE_DIR}/selettori/modello_cascata_fold_EO.joblib'},
    5: {"name": "AutoML Cascade", "out_dir": "esperimento_5", "out_file": "final_validation_automl",
        "AMO": f'{BASE_DIR}/selettori/modello_automl_cascata_AMO.joblib',
        "EO": f'{BASE_DIR}/selettori/modello_automl_cascata_EO.joblib'},
    6: {"name": "AutoML L1 (Triage Only)", "out_dir": "esperimento_6", "out_file": "final_validation_automl_L1",
        "AMO": f'{BASE_DIR}/selettori/modello_automl_cascata_AMO.joblib',
        "EO": f'{BASE_DIR}/selettori/modello_automl_cascata_EO.joblib'},
    7: {"name": "AutoML L2 (Oracle Only)", "out_dir": "esperimento_7", "out_dir": "esperimento_7", "out_file": "final_validation_automl_L2",
        "AMO": f'{BASE_DIR}/selettori/modello_automl_cascata_AMO.joblib',
        "EO": f'{BASE_DIR}/selettori/modello_automl_cascata_EO.joblib'},
}


# ==========================================
# 2. SUPPORT FUNCTIONS
# ==========================================
def identify_problem(instance_name):
    name = str(instance_name).upper()
    if 'WGC' in name or 'COLORING' in name or 'COLOURING' in name : return 'GraphColouring'
    if 'KC' in name or 'KNAPSACK' in name: return 'Knapsack'
    if 'GA' in name or 'GROUP' in name: return 'GroupAssignment'
    return 'Other'

def find_real_path(base_dir, problem_folder, file_name):
    """
    Scans all subfolders inside the problem folder 
    (e.g., aijcai-instances, training-instances) until it finds the exact file.
    """
    problem_folder_path = os.path.join(base_dir, problem_folder)
    for root, dirs, files in os.walk(problem_folder_path):
        if file_name in files:
            return os.path.join(root, file_name)
    return None

def build_command(domain, problem_folder, prediction, instance_path):
    enc_dir = f"{BASE_DIR}/{problem_folder}"
    if prediction == 'CLINGO-AMO':
        return ["clingo", f"{enc_dir}/encoding-plain-amo.asp", instance_path]
    executable = "amoclingo"
    encoding = f"{enc_dir}/encoding-amosum-{domain.lower()}.asp"
    lazy_val = "true" if prediction.endswith("-L") else "false"
    return [executable, f"-e={encoding}", f"-l={lazy_val}", "-m=minfly", "-lg=cpp", f"-i={instance_path}"]

def load_base_model(clf_loaded):
    if isinstance(clf_loaded, dict) and 'tipo_modello' not in clf_loaded:
        for key, value in clf_loaded.items():
            if hasattr(value, 'predict'): return value
    return clf_loaded

# ==================================================
# 3. MAIN FUNCTION: VALIDATION
# ==================================================
def final_validation(domain, exp_id):
    cfg = EXPERIMENT_CONFIG[exp_id]
    model_path = cfg[domain]
    test_path = FINAL_TEST[domain]
    original_paths = ORIGINALS[domain]
    
    print(f"\n{'='*80}\n🚀 EXPERIMENT {exp_id} ({cfg['name']}) - DOMAIN: {domain}\n{'='*80}")
    
    if not os.path.exists(model_path):
        print(f"⚠️ Unable to find the model at {model_path}.")
        return None

    # --------------------------------------------------
    # A. Data and Model Loading
    # --------------------------------------------------
    package = joblib.load(model_path)
    df_test = pd.read_csv(test_path)
    df_orig = pd.concat([pd.read_csv(p) for p in original_paths], ignore_index=True)
    lookup_min_time = df_orig.set_index('instance_name')['min_time'].to_dict()
    
    # --------------------------------------------------
    # B. Feature Preparation
    # --------------------------------------------------
    df_test_clean = df_test.copy()
    if 'span' in df_test_clean.columns:
        df_test_clean['span_ratio'] = np.where(df_test_clean['bound'] > 0, df_test_clean['span'] / df_test_clean['bound'], 0)
        df_test_clean['span_log'] = np.log1p(df_test_clean['span'])
    
    drop_columns = ['instance_name', 'best_config_TRUE', 'best_config', 'domain']
    X_raw = df_test_clean.drop(columns=[c for c in drop_columns if c in df_test_clean.columns])
    
    # Dynamic feature alignment
    if exp_id == 1:
        clf = load_base_model(package)
        if hasattr(clf, 'feature_names_in_'):
            X = X_raw[clf.feature_names_in_]
        else:
            X = X_raw
    else:
        expected_features = package['feature_names']
        missing_features = [f for f in expected_features if f not in X_raw.columns]
        if missing_features: raise ValueError(f"Missing features: {missing_features}")
        X = X_raw[expected_features]

    # --------------------------------------------------
    # C. Model Inference (Switch based on experiment)
    # --------------------------------------------------
    start_selection = time.time()
    preds_name = []
    usage_l1 = usage_l2 = 0
    
    # EXP 1: Standalone XGBoost
    if exp_id == 1:
        preds_raw = clf.predict(X)
        label_map = MAP_LABELS[domain]
        preds_name = [label_map[p] if isinstance(p, (int, np.integer)) else p for p in preds_raw]
        usage_l2 = len(X)
        
    # EXP 2, 3, 4, 5: Full Cascade
    elif exp_id in [2, 3, 4, 5]:
        dt_triage = package['modello_triage']
        xgb_oracle = package['modello_oracolo']
        le = package['label_encoder']
        threshold = package.get('soglia_confidenza', 0.5)
        
        probabilities = dt_triage.predict_proba(X)
        max_confidence = np.max(probabilities, axis=1)
        
        choices_l1 = dt_triage.predict(X)
        choices_l2 = xgb_oracle.predict(X)
        
        # Patch for FOLD-R++ output
        if exp_id == 4 and isinstance(choices_l1[0], str) and choices_l1[0] in le.classes_:
            choices_l1 = le.transform(choices_l1)
            
        numeric_preds = np.where(max_confidence >= threshold, choices_l1, choices_l2)
        preds_name = le.inverse_transform(numeric_preds.astype(int))
        
        usage_l1 = np.sum(max_confidence >= threshold)
        usage_l2 = len(X) - usage_l1
        
    # EXP 6: Triage Only (L1)
    elif exp_id == 6:
        dt_triage = package['modello_triage']
        le = package['label_encoder']
        choices_l1 = dt_triage.predict(X)
        preds_name = le.inverse_transform(choices_l1)
        usage_l1 = len(X)

    # EXP 7: Oracle Only (L2)
    elif exp_id == 7:
        xgb_oracle = package['modello_oracolo']
        le = package['label_encoder']
        choices_l2 = xgb_oracle.predict(X)
        preds_name = le.inverse_transform(choices_l2)
        usage_l2 = len(X)

    total_selection_time = time.time() - start_selection
    avg_selection_time = total_selection_time / len(X)
    
    print(f"📊 Model Usage: Triage (L1) -> {usage_l1} instances | Oracle (L2) -> {usage_l2} instances.")
    print(f"⏱️ ML inference time: {total_selection_time:.4f}s total ({avg_selection_time:.4f}s avg/instance).")

    # --------------------------------------------------
    # D. Actual Execution
    # --------------------------------------------------
    instances = df_test['instance_name'].values
    results = []

    for i in range(len(instances)):
        inst_path = instances[i]
        inst_base = os.path.basename(inst_path) 
        prob = identify_problem(inst_base)
        config_ml = preds_name[i]
        
        physical_path = find_real_path(BASE_DIR, prob, inst_base)
        if not physical_path:
            print(f"[{i+1}/{len(instances)}] ⚠️ SKIPPED: File '{inst_base}' not physically found.")
            continue
        
        m_time = lookup_min_time.get(inst_path, np.nan)
        command = build_command(domain, prob, config_ml, physical_path)
        
        print(f"\n[{i+1}/{len(instances)}] Testing {inst_base} ({prob}) -> {config_ml}")
        
        start = time.time()
        try:
            res = subprocess.run(command, capture_output=True, text=True, timeout=TIMEOUT_SEC)
            elapsed = time.time() - start
            success = any(x in res.stdout for x in ["SATISFIABLE", "OPTIMUM FOUND"]) or res.returncode in [10, 20]
            
            if not success:
                print(f"   ❌ ERROR/UNSAT (Code {res.returncode})")
            else:
                print(f"   ✅ Solved in {elapsed:.2f}s (Expected Min Time: {m_time:.2f}s)")
                    
        except subprocess.TimeoutExpired:
            elapsed = TIMEOUT_SEC
            success = False
            print(f"   ⏳ TIMEOUT reached ({TIMEOUT_SEC}s).")

        results.append({
            'domain': domain,
            'problem': prob,
            'instance': inst_base,
            'ml_config': config_ml,
            'ml_inference_time': avg_selection_time,
            'ml_time': elapsed,
            'total_time': elapsed + avg_selection_time,
            'real_min_time': m_time,
            'solved': success
        })

    # --------------------------------------------------
    # E. Processing and Saving
    # --------------------------------------------------
    if not results: return None
    df_res = pd.DataFrame(results)
    
    print(f"\n--- AGGREGATED REPORT PER PROBLEM ({domain}) ---")
    summary = df_res.groupby('problem').agg(
        Total_Instances=('instance', 'count'),
        Solved_ML=('solved', 'sum'),
        ML_Resolution_Time=('ml_time', 'mean'),
        Total_Avg_Time=('total_time', 'mean'),
        Ideal_Avg_Time=('real_min_time', 'mean')
    ).round(2)
    print(summary.to_string())
    
    # Create directory if it doesn't exist
    out_dir = cfg['out_dir']
    os.makedirs(out_dir, exist_ok=True)
    
    csv_name = os.path.join(out_dir, f"{cfg['out_file']}_{domain}.csv")
    df_res.to_csv(csv_name, index=False)
    print(f"\n💾 Details saved in: {csv_name}")
    
    return df_res

# ==========================================
# 4. ENTRY POINT
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified script for ML models validation.")
    parser.add_argument('--domain', type=str, choices=['AMO', 'EO', 'ALL'], default='ALL', 
                        help='Choose which domain to validate (AMO, EO, or ALL)')
    parser.add_argument('--experiment', type=str, choices=[str(i) for i in range(1, 8)] + ['ALL'], default='ALL', 
                        help='Choose experiment ID (1=XGB, 2=Cascade_s75, 3=Cascade_Opt, 4=Fold, 5=AutoML, 6=Auto_L1, 7=Auto_L2)')
    
    args = parser.parse_args()
    
    domains_to_run = ['AMO', 'EO'] if args.domain == 'ALL' else [args.domain]
    experiments_to_run = list(range(1, 8)) if args.experiment == 'ALL' else [int(args.experiment)]

    for exp in experiments_to_run:
        for dom in domains_to_run:
            final_validation(dom, exp)

    print("\n" + "="*80)
    print("🎯 GLOBAL EXECUTION COMPLETED")
    print("="*80)