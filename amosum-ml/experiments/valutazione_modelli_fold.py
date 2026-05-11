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

# =================== WRAPPER CORRETTO PER FOLD-R++ ===================
try:
    from foldrpp import Foldrpp
except ImportError:
    pass # Ignora se viene solo caricato l'oggetto da joblib, altrimenti fallirà durante la predict

class FoldRPPCascade:
    """
    Wrapper per rendere FOLD-R++ compatibile con sklearn e multi-classe.
    Necessario in questo script per poter deserializzare correttamente il modello salvato con joblib.
    """
    def __init__(self, ratio=0.5): 
        self.ratio = ratio
        self.models = {}      
        self.label_encoder = LabelEncoder()
        self.classes_ = None
        self.feature_names_in_ = None
        self.majority_class_ = None

    def _prepare_feature_lists(self, X):
        str_attrs = []
        num_attrs = []
        for col in X.columns:
            if X[col].dtype == 'object' or X[col].dtype.name == 'category':
                str_attrs.append(col)
            else:
                num_attrs.append(col)
        return str_attrs, num_attrs

    def fit(self, X, y):
        pass # Non serve nel test

    def predict(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        
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
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        
        # Per permettere alla logica di routing a cascata di funzionare.
        # Assegna 1.0 (confidenza massima) alla classe se una regola scatta,
        # altrimenti 0 per tutte, causando il fallback all'oracolo.
        proba = np.zeros((X.shape[0], len(self.classes_)))
        
        for idx in range(X.shape[0]):
            sample = X.iloc[[idx]].copy()
            sample['label'] = 0 
            sample_record = sample.to_dict(orient='records')
            
            scores = {}
            for i, class_name in enumerate(self.classes_):
                if class_name in self.models:
                    model = self.models[class_name]
                    pred = model.predict(sample_record)[0]
                    if str(pred) in ['1', '1.0', 'True']:
                        proba[idx, i] = 1.0
        
        # Normalizzazione opzionale (se più regole scattano, divide equamente)
        row_sums = proba.sum(axis=1)
        # Per le righe dove la somma > 0, normalizza, altrimenti lascia tutto a 0
        proba[row_sums > 0] = proba[row_sums > 0] / row_sums[row_sums > 0, np.newaxis]
        
        return proba

# ==========================================
# 1. PERCORSI E CONFIGURAZIONI
# ==========================================
BASE_DIR = '/home/guests/mmacri/AMOSUM/benchmarks'

# Modelli pre-addestrati FOLD-R++ + XGBoost
MODEL_AMO_PATH = f'{BASE_DIR}/selettori/modello_cascata_fold_AMO.joblib'
MODEL_EO_PATH  = f'{BASE_DIR}/selettori/modello_cascata_fold_EO.joblib'

# Dataset di TEST FINALE (20%)
TEST_FINALE_AMO = f'{BASE_DIR}/test_finale/test_set_AMO_20perc.csv'
TEST_FINALE_EO  = f'{BASE_DIR}/test_finale/test_set_EO_20perc.csv'

# Dataset ORIGINALI (per recuperare il min_time)
ORIGINALI = {
    'AMO': [f'{BASE_DIR}/dataset_builder/preprocessed/train/training_dataset_amo.csv',
            f'{BASE_DIR}/dataset_builder/preprocessed/test/testing_dataset_amo.csv'],
    'EO':  [f'{BASE_DIR}/dataset_builder/preprocessed/train/training_dataset_eo.csv',
            f'{BASE_DIR}/dataset_builder/preprocessed/test/testing_dataset_eo.csv']
}

TIMEOUT_SEC = 1200

# Mappatura LabelEncoder (Assicurati che l'ordine sia lo stesso dell'addestramento)
MAP_AMO = {0: 'AMOCLINGO-INF-MR', 1: 'AMOCLINGO-L', 2: 'CLINGO-AMO'}
MAP_EO  = {0: 'CLINGO-AMO', 1: 'EOCLINGO-INF-MR', 2: 'EOCLINGO-L'}

# ==========================================
# 2. FUNZIONI DI SUPPORTO
# ==========================================
def identifica_problema(instance_name):
    name = str(instance_name).upper()
    if 'WGC' in name or 'COLORING' in name or 'COLOURING' in name : return 'GraphColouring'
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
# FUNZIONE PRINCIPALE: VALIDAZIONE FINALE A CASCATA
# ==================================================
def validazione_finale(dominio, model_path, test_path, original_paths):
    print(f"\n{'='*80}\n🚀 VALIDAZIONE FINALE (CASCATA FOLD-R++ + XGBoost): {dominio}\n{'='*80}")
    
    # --------------------------------------------------
    # A. Caricamento Ecosistema a Cascata
    # --------------------------------------------------
    if not os.path.exists(model_path):
        print(f"⚠️ Impossibile trovare il modello in {model_path}. Assicurati di averlo generato e salvato correttamente.")
        return None

    pacchetto = joblib.load(model_path)
    
    if not isinstance(pacchetto, dict):
        raise ValueError(f"Il file {model_path} non è un ecosistema a cascata valido! (Controlla i log per info)")
        
    fold_triage = pacchetto['modello_triage']
    xgb_oracle = pacchetto['modello_oracolo']
    le = pacchetto['label_encoder']
    soglia = pacchetto.get('soglia_confidenza', 0.5)
    feature_attese = pacchetto['feature_names']
        
    df_test = pd.read_csv(test_path)
    
    # --------------------------------------------------
    # B. Tabella di Lookup per i min_time
    # --------------------------------------------------
    df_orig = pd.concat([pd.read_csv(p) for p in original_paths], ignore_index=True)
    lookup_min_time = df_orig.set_index('instance_name')['min_time'].to_dict()
    
    # --------------------------------------------------
    # C. Preparazione Feature e Allineamento
    # --------------------------------------------------
    df_test_clean = df_test.copy()
    
    if 'span' in df_test_clean.columns:
        df_test_clean['span_ratio'] = np.where(df_test_clean['bound'] > 0, df_test_clean['span'] / df_test_clean['bound'], 0)
        df_test_clean['span_log'] = np.log1p(df_test_clean['span'])
    
    colonne_drop = ['instance_name', 'best_config_TRUE', 'best_config', 'domain']
    X_grezzo = df_test_clean.drop(columns=[c for c in colonne_drop if c in df_test_clean.columns])
    
    feature_mancanti = [f for f in feature_attese if f not in X_grezzo.columns]
    if feature_mancanti:
        raise ValueError(f"Mancano queste feature nel dataset di test: {feature_mancanti}")
    X = X_grezzo[feature_attese]
        
    # --------------------------------------------------
    # D. Predizione a Cascata (Triage -> Oracolo)
    # --------------------------------------------------
    start_selezione = time.time()
    
    # 1. Chiediamo a FOLD-R++ di prendere una decisione e valutarne la sicurezza
    # L'implementazione modificata di predict_proba ritorna >= 1.0 se una regola scatta
    probabilita = fold_triage.predict_proba(X)
    confidenza_massima = np.max(probabilita, axis=1)
    
    # Fallback su predict se vogliamo il comportamento base
    scelte_fold = fold_triage.predict(X)
    
    # Attenzione: fold_triage potrebbe riportare label in stringa a seconda di come è stato addestrato.
    # Assicuriamoci che scelte_fold sia codificato in intero (stesso dominio del LabelEncoder)
    if isinstance(scelte_fold[0], str) and scelte_fold[0] in le.classes_:
        scelte_fold = le.transform(scelte_fold)

    scelte_xgboost = xgb_oracle.predict(X)
    
    # 2. Routing dinamico (Se > Soglia usa FOLD-R++, altrimenti usa XGBoost)
    pred_numeriche = np.where(confidenza_massima >= soglia, scelte_fold, scelte_xgboost)
    
    # 3. Riconvertiamo i numeri nelle stringhe originali (es. 'EOCLINGO-L')
    preds_name = le.inverse_transform(pred_numeriche.astype(int))
    
    tempo_selezione_totale = time.time() - start_selezione
    tempo_selezione_medio = tempo_selezione_totale / len(X)
    
    # Statistiche sull'utilizzo della cascata
    uso_fold = np.sum(confidenza_massima >= soglia)
    uso_xgboost = len(X) - uso_fold
    
    print(f"📊 Utilizzo Modelli: Triage(FOLD-R++) ha risolto {uso_fold} istanze, Oracolo(XGBoost) ne ha valutate {uso_xgboost}.")
    print(f"⏱️  Tempo di inferenza: {tempo_selezione_totale:.4f}s in totale ({tempo_selezione_medio:.4f}s in media per istanza).")
    
    istanze = df_test['instance_name'].values
    risultati = []

    # --------------------------------------------------
    # E. Esecuzione Reale
    # --------------------------------------------------
    for i in range(len(istanze)):
        inst_path = istanze[i]
        inst_base = os.path.basename(inst_path) 
        prob = identifica_problema(inst_base)
        config_ml = preds_name[i]
        
        percorso_fisico = cerca_percorso_reale(BASE_DIR, prob, inst_base)
        
        if not percorso_fisico:
            print(f"[{i+1}/{len(istanze)}] ⚠️ SALTO: File '{inst_base}' non trovato fisicamente nella cartella {prob}.")
            continue
        
        m_time = lookup_min_time.get(inst_path, np.nan)
        comando = costruisci_comando(dominio, prob, config_ml, percorso_fisico)
        
        print(f"\n[{i+1}/{len(istanze)}] Testando {inst_base} ({prob}) -> {config_ml}")
        
        start = time.time()
        try:
            res = subprocess.run(comando, capture_output=True, text=True, timeout=TIMEOUT_SEC)
            elapsed = time.time() - start
            
            success = any(x in res.stdout for x in ["SATISFIABLE", "OPTIMUM FOUND"]) or res.returncode in [10, 20]
            
            if not success:
                print(f"   ❌ ERRORE/UNSAT (Codice {res.returncode})")
                if res.stderr:
                    print(f"      Dettaglio: {res.stderr.strip()[:200]}")
            else:
                print(f"   ✅ Risolto in {elapsed:.2f}s (Min Time Atteso: {m_time:.2f}s)")
                    
        except subprocess.TimeoutExpired:
            elapsed = TIMEOUT_SEC
            success = False
            print(f"   ⏳ TIMEOUT raggiunto ({TIMEOUT_SEC}s).")

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

    # --------------------------------------------------
    # F. Elaborazione Statistiche
    # --------------------------------------------------
    if not risultati:
        print("\n⚠️ Nessun risultato generato. Verifica i percorsi.")
        return None

    df_res = pd.DataFrame(risultati)
    
    print(f"\n--- REPORT AGGREGATO PER PROBLEMA ({dominio}) ---")
    summary = df_res.groupby('problem').agg(
        Totale_Istanze=('instance', 'count'),
        Risolte_ML=('solved', 'sum'),
        Tempo_Risoluzione_ML=('ml_time', 'mean'),
        Tempo_Totale_Medio=('total_time', 'mean'),
        Tempo_Ideale_Medio=('min_time_reale', 'mean')
    ).round(2)
    
    print(summary.to_string())
    
    nome_csv = f"validazione_finale_cascata_fold_{dominio}.csv"
    df_res.to_csv(nome_csv, index=False)
    print(f"\n💾 Dettagli salvati in: {nome_csv}")
    
    return df_res

# ==========================================
# 4. ESECUZIONE
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Esegue la validazione dei modelli a cascata FOLD-R++ / XGBoost.")
    parser.add_argument('--dominio', type=str, choices=['AMO', 'EO', 'ALL'], default='ALL', 
                        help='Scegli quale dominio validare (AMO, EO, o ALL)')
    args = parser.parse_args()
    
    if args.dominio in ['AMO', 'ALL']:
        res_amo = validazione_finale("AMO", MODEL_AMO_PATH, TEST_FINALE_AMO, ORIGINALI['AMO'])
    
    if args.dominio in ['EO', 'ALL']:
        res_eo  = validazione_finale("EO",  MODEL_EO_PATH,  TEST_FINALE_EO,  ORIGINALI['EO'])
    
    print("\n" + "="*80)
    print(f"CONFRONTO {args.dominio} A CASCATA (FOLD-R++) COMPLETATO")
    print("="*80)