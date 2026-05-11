import pandas as pd
import numpy as np
import joblib
import subprocess
import time
import os
import warnings

warnings.filterwarnings('ignore')

# ==========================================
# 1. PERCORSI E CONFIGURAZIONI
# ==========================================
BASE_DIR = '/home/guests/mmacri/AMOSUM/benchmarks'

# Modelli pre-addestrati
MODEL_AMO_PATH = f'{BASE_DIR}/selettori/modello_AMO_xgboost.joblib'
MODEL_EO_PATH  = f'{BASE_DIR}/selettori/modello_EO_xgboost.joblib'

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
    """
    Scansiona tutte le sottocartelle dentro la cartella del problema 
    (es. test-instances, training-instances) finché non trova il file esatto.
    """
    cartella_problema = os.path.join(base_dir, problem_folder)
    
    # os.walk entra in tutte le cartelle e sottocartelle
    for root, dirs, files in os.walk(cartella_problema):
        if nome_file in files:
            # Trovato! Restituisce il percorso completo (es: /home/.../training-instances-ga/istanza.asp)
            return os.path.join(root, nome_file)
            
    # Se il ciclo finisce e non ha trovato nulla, restituisce None
    return None

def costruisci_comando(dominio, problem_folder, predizione, instance_path):
    enc_dir = f"{BASE_DIR}/{problem_folder}"
    if predizione == 'CLINGO-AMO':
        return ["clingo", f"{enc_dir}/encoding-plain-amo.asp", instance_path]
    
    eseguibile = "amoclingo"
    encoding = f"{enc_dir}/encoding-amosum-{dominio.lower()}.asp"
    lazy_val = "true" if predizione.endswith("-L") else "false"
    
    return [eseguibile, f"-e={encoding}", f"-l={lazy_val}", "-m=minfly", "-lg=cpp", f"-i={instance_path}"]

# ==========================================
# 3. CORE VALIDATION
# ==========================================
# ==================================================
# FUNZIONE PRINCIPALE: VALIDAZIONE FINALE
# ==================================================
def validazione_finale(dominio, model_path, test_path, original_paths, label_map):
    print(f"\n{'='*80}\n🚀 VALIDAZIONE FINALE: {dominio}\n{'='*80}")
    
    # --------------------------------------------------
    # A. Caricamento Modello 
    # --------------------------------------------------
    clf_loaded = joblib.load(model_path)
    
    # Estrazione del modello se è salvato dentro un dizionario
    if isinstance(clf_loaded, dict):
        clf = None
        for key, value in clf_loaded.items():
            if hasattr(value, 'predict'):
                clf = value
                break
        if clf is None:
            raise ValueError(f"Nessun modello valido trovato nel file {model_path}!")
    else:
        clf = clf_loaded
        
    # Carichiamo il dataset di test
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
    
    # 1. Ricreiamo le feature derivate usate in addestramento
    if 'span' in df_test_clean.columns:
        df_test_clean['span_ratio'] = np.where(df_test_clean['bound'] > 0, df_test_clean['span'] / df_test_clean['bound'], 0)
        df_test_clean['span_log'] = np.log1p(df_test_clean['span'])
    
    # 2. Rimuoviamo le colonne non necessarie al ML
    colonne_drop = ['instance_name', 'best_config_TRUE', 'best_config', 'domain']
    X_grezzo = df_test_clean.drop(columns=[c for c in colonne_drop if c in df_test_clean.columns])
    
    # 3. Allineamento rigoroso delle colonne per XGBoost
    if hasattr(clf, 'feature_names_in_'):
        feature_attese = clf.feature_names_in_
        feature_mancanti = [f for f in feature_attese if f not in X_grezzo.columns]
        if feature_mancanti:
            raise ValueError(f"Mancano queste feature nel dataset di test: {feature_mancanti}")
        X = X_grezzo[feature_attese]
    else:
        X = X_grezzo
        
    # --------------------------------------------------
    # D. Predizione
    # --------------------------------------------------
    start_selezione = time.time()
    
    preds_raw = clf.predict(X)
    preds_name = [label_map[p] if isinstance(p, (int, np.integer)) else p for p in preds_raw]
    
    tempo_selezione_totale = time.time() - start_selezione
    tempo_selezione_medio = tempo_selezione_totale / len(X)
    
    print(f"⏱️  Tempo di inferenza: {tempo_selezione_totale:.4f}s in totale ({tempo_selezione_medio:.4f}s in media per istanza).")
    
    istanze = df_test['instance_name'].values
    risultati = []

    # --------------------------------------------------
    # E. Esecuzione Reale
    # --------------------------------------------------
    for i in range(len(istanze)):
        inst_path = istanze[i]
        inst_base = os.path.basename(inst_path) # Es: "134-group...asp"
        prob = identifica_problema(inst_base)
        config_ml = preds_name[i]
        
        # Ricerca dinamica del percorso reale del file
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
            
            # Controllo successo (SAT o OPTIMUM)
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
    
    nome_csv = f"validazione_finale_dettagliata_{dominio}.csv"
    df_res.to_csv(nome_csv, index=False)
    print(f"\n💾 Dettagli salvati in: {nome_csv}")
    
    return df_res
# ==========================================
# 4. ESECUZIONE
# ==========================================
if __name__ == "__main__":
    res_amo = validazione_finale("AMO", MODEL_AMO_PATH, TEST_FINALE_AMO, ORIGINALI['AMO'], MAP_AMO)
    res_eo = validazione_finale("EO", MODEL_EO_PATH, TEST_FINALE_EO, ORIGINALI['EO'], MAP_EO)
    
    print("\n" + "="*80)
    print("CONFRONTO GLOBALE COMPLETATO")
    print("="*80)