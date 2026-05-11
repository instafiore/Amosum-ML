import pandas as pd
import numpy as np
import joblib
import subprocess
import time
import os
import warnings
import argparse

warnings.filterwarnings('ignore')

# ==========================================
# 0. WRAPPER FOLD-R++
# ==========================================
try:
    from foldrpp import Foldrpp
except ImportError:
    pass

class FoldRPPCascade:
    def __init__(self, ratio=0.5): 
        self.ratio = ratio
        self.models = {}      
        self.classes_ = None

    def _prepare_feature_lists(self, X):
        str_attrs, num_attrs = [], []
        for col in X.columns:
            if X[col].dtype == 'object' or X[col].dtype.name == 'category':
                str_attrs.append(col)
            else:
                num_attrs.append(col)
        return str_attrs, num_attrs

    def fit(self, X, y):
        pass

    def predict(self, X):
        probas = self.predict_proba(X)
        best_indices = np.argmax(probas, axis=1)
        return self.classes_[best_indices]

    def predict_proba(self, X):
        if not isinstance(X, pd.DataFrame): X = pd.DataFrame(X)
        n_samples = X.shape[0]
        n_classes = len(self.classes_)
        probas = np.zeros((n_samples, n_classes))
        
        for idx in range(n_samples):
            sample = X.iloc[[idx]].copy()
            sample['label'] = 0 
            sample_record = sample.to_dict(orient='records')
            
            scores = {}
            for class_idx, class_val in enumerate(self.classes_):
                pred = self.models[class_val].predict(sample_record)[0]
                scores[class_idx] = 1 if str(pred) in ['1', '1.0', 'True'] else 0
            
            if sum(scores.values()) > 0:
                best_idx = max(scores, key=scores.get)
                probas[idx, best_idx] = 1.0
            else:
                probas[idx, :] = 1.0 / n_classes
        return probas

# ==========================================
# 1. PERCORSI E CONFIGURAZIONI
# ==========================================
BASE_DIR = '/home/guests/mmacri/AMOSUM/benchmarks'

MODEL_AMO_PATH = f'{BASE_DIR}/selettori/modello_automl_cascata_AMO.joblib'
MODEL_EO_PATH  = f'{BASE_DIR}/selettori/modello_automl_cascata_EO.joblib'

TEST_FINALE_AMO = f'{BASE_DIR}/test_finale/test_set_AMO_20perc.csv'
TEST_FINALE_EO  = f'{BASE_DIR}/test_finale/test_set_EO_20perc.csv'

ORIGINALI = {
    'AMO': [f'{BASE_DIR}/dataset_builder/preprocessed/train/training_dataset_amo.csv',
            f'{BASE_DIR}/dataset_builder/preprocessed/test/testing_dataset_amo.csv'],
    'EO':  [f'{BASE_DIR}/dataset_builder/preprocessed/train/training_dataset_eo.csv',
            f'{BASE_DIR}/dataset_builder/preprocessed/test/testing_dataset_eo.csv']
}

TIMEOUT_SEC = 1200

# ==========================================
# 2. FUNZIONI DI SUPPORTO
# ==========================================
def identifica_problema(instance_name):
    name = str(instance_name).upper()
    if 'WGC' in name or 'COLORING' in name or 'COLOURING' in name: return 'GraphColouring'
    if 'KC' in name or 'KNAPSACK' in name: return 'Knapsack'
    if 'GA' in name or 'GROUP' in name: return 'GroupAssignment'
    return 'Altro'

def cerca_percorso_reale(base_dir, problem_folder, nome_file):
    cartella_problema = os.path.join(base_dir, problem_folder)
    for root, dirs, files in os.walk(cartella_problema):
        if nome_file in files:
            return os.path.join(root, nome_file)
    return None

def costruisci_comando(dominio, problem_folder, predizione, instance_path):
    enc_dir = f"{BASE_DIR}/{problem_folder}"
    if predizione == 'CLINGO-AMO':
        return ["clingo", f"{enc_dir}/encoding-plain-amo.asp", instance_path]
    eseguibile = "amoclingo"
    encoding = f"{enc_dir}/encoding-amosum-{dominio.lower()}.asp"
    lazy_val = "true" if predizione.endswith("-L") else "false"
    return [eseguibile, f"-e={encoding}", f"-l={lazy_val}", "-m=minfly", "-lg=cpp", f"-i={instance_path}"]

# ==================================================
# FUNZIONE PRINCIPALE: VALIDAZIONE SOLO SU L1
# ==================================================
def validazione_finale(dominio, model_path, test_path, original_paths):
    print(f"\n{'='*80}\n🚀 VALIDAZIONE SOLO TRIAGE (L1): {dominio}\n{'='*80}")
    
    pacchetto = joblib.load(model_path)
    
    dt_triage = pacchetto['modello_triage']
    le = pacchetto['label_encoder']
    feature_attese = pacchetto['feature_names']
    metadata = pacchetto.get('metadata', None)
    
    if metadata:
        print(f"Modello Triage L1 caricato: {metadata.get('Triage_L1')}")
    
    df_test = pd.read_csv(test_path)
    df_orig = pd.concat([pd.read_csv(p) for p in original_paths], ignore_index=True)
    lookup_min_time = df_orig.set_index('instance_name')['min_time'].to_dict()
    
    df_test_clean = df_test.copy()
    if 'span' in df_test_clean.columns:
        df_test_clean['span_ratio'] = np.where(df_test_clean['bound'] > 0, df_test_clean['span'] / df_test_clean['bound'], 0)
        df_test_clean['span_log'] = np.log1p(df_test_clean['span'])
    
    colonne_drop = ['instance_name', 'best_config_TRUE', 'best_config', 'domain']
    X_grezzo = df_test_clean.drop(columns=[c for c in colonne_drop if c in df_test_clean.columns])
    X = X_grezzo[feature_attese]
        
    start_selezione = time.time()
    
    # Esegue solo il classificatore L1
    scelte_albero = dt_triage.predict(X)
    pred_numeriche = scelte_albero
    
    preds_name = le.inverse_transform(pred_numeriche)
    
    tempo_selezione_totale = time.time() - start_selezione
    tempo_selezione_medio = tempo_selezione_totale / len(X)
    
    print(f"📊 Utilizzo Modelli: SOLO Triage (L1) ha risolto tutte le {len(X)} istanze.")
    print(f"⏱️  Tempo di inferenza: {tempo_selezione_totale:.4f}s in totale ({tempo_selezione_medio:.4f}s in media per istanza).")
    
    istanze = df_test['instance_name'].values
    risultati = []

    for i in range(len(istanze)):
        inst_path = istanze[i]
        inst_base = os.path.basename(inst_path) 
        prob = identifica_problema(inst_base)
        config_ml = preds_name[i]
        
        percorso_fisico = cerca_percorso_reale(BASE_DIR, prob, inst_base)
        if not percorso_fisico:
            continue
        
        m_time = lookup_min_time.get(inst_path, np.nan)
        comando = costruisci_comando(dominio, prob, config_ml, percorso_fisico)
        
        start = time.time()
        try:
            res = subprocess.run(comando, capture_output=True, text=True, timeout=TIMEOUT_SEC)
            elapsed = time.time() - start
            success = any(x in res.stdout for x in ["SATISFIABLE", "OPTIMUM FOUND"]) or res.returncode in [10, 20]
        except subprocess.TimeoutExpired:
            elapsed = TIMEOUT_SEC
            success = False

        risultati.append({
            'domain': dominio,
            'problem': prob,
            'instance': inst_base,
            'ml_config': config_ml,
            'ml_inference_time': tempo_selezione_medio,
            'ml_time': elapsed,
            'total_time': elapsed + tempo_selezione_medio,
            'min_time_reale': m_time,
            'solved': success
        })

    if not risultati: return None
    df_res = pd.DataFrame(risultati)
    
    nome_csv = f"validazione_finale_automl_L1_{dominio}.csv"
    df_res.to_csv(nome_csv, index=False)
    print(f"\n💾 Dettagli salvati in: {nome_csv}")
    
    return df_res

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validazione SOLO Modello Triage L1")
    parser.add_argument('--dominio', type=str, choices=['AMO', 'EO', 'ALL'], default='ALL')
    args = parser.parse_args()
    if args.dominio in ['AMO', 'ALL']: validazione_finale("AMO", MODEL_AMO_PATH, TEST_FINALE_AMO, ORIGINALI['AMO'])
    if args.dominio in ['EO', 'ALL']: validazione_finale("EO",  MODEL_EO_PATH,  TEST_FINALE_EO,  ORIGINALI['EO'])
