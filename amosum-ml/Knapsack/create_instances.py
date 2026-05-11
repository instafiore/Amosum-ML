#!/usr/bin/python3
import sys
import os
import shutil
import numpy as np

# --- CONFIGURAZIONE TRAINING SET ---
nips = 30  
start_n = 5 #da 5 oggetti a 100 oggetti, con step di 5
end_n = 100
step = 5

# Path di output assoluto
base_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(base_dir, "training-instances-k")
# Standard deviation per i bound
std_dev_weight = 1000
std_dev_value = 5000

# Range di generazione per creare oggetti eterogenei
object_scenarios = [
    {"min_w": 10, "max_w": 100, "min_v": 100, "max_v": 500},   # Oggetti piccoli/leggeri
    {"min_w": 100, "max_w": 200, "min_v": 1000, "max_v": 2000},# Configurazione originale
    {"min_w": 200, "max_w": 1000, "min_v": 500, "max_v": 5000} # Oggetti grandi/pesanti
]

k = 20  #numero massimo di oggetti che puoi prendere per ogni singola categoria.
instances = []
ind = 0

print(f"Generazione Training Set in corso in:\n{output_dir} ...")

for n in range(start_n, end_n + 1, step):
    for scenario in object_scenarios:
        # Generazione oggetti basata sullo scenario corrente
        values = np.random.uniform(scenario["min_v"], scenario["max_v"], n)
        weights = np.random.uniform(scenario["min_w"], scenario["max_w"], n)

        values = np.around(values).astype(int)
        weights = np.around(weights).astype(int)

        mean_v = np.mean(values)
        mean_w = np.mean(weights)

        # Calcolo dei punti di riferimento C1 e C2
        C1_w = n * k * mean_w
        C1_v = n * k * mean_v # valore massimo che puoi ottenere rispettando la regola di prendere "al massimo un oggetto per gruppo" (il vincolo AMO)
        C2_v = n * mean_v * (k * (k + 1)) / 2 # valore se potessi prendere tutto ignorando le regole.

        # Generazione varianti
        for i in range(nips):
            lbs_weight = int(np.random.normal(C1_w * 0.7, std_dev_weight))
            
            rand_val = np.random.random()
            if rand_val < 0.2:
                # Crea problemi facilissimi o impossibili scegliendo casualmente un numero tra 0 e C1 o C2.
                t_type = "type1" if rand_val < 0.1 else "type2"
                base_c = C1_v if t_type == "type1" else C2_v
                lbs_value = int(np.random.uniform(0, base_c))
            elif rand_val < 0.6:
                t_type = "type3"
                lbs_value = int(np.random.uniform(C1_v * 0.5, C1_v * 1.2))
            else:
                t_type = "type4"
                lbs_value = int(np.random.normal(C1_v, std_dev_value))

            lbs_value = max(0, lbs_value)
            lbs_weight = max(0, lbs_weight)

            instance = [
                f"{ind:05}", n, lbs_weight, lbs_value, 
                list(weights), list(values), t_type
            ]
            instances.append(instance)
            ind += 1

if not os.path.exists(output_dir):
    os.makedirs(output_dir)
else:
    # Svuota la cartella se esiste già, senza cancellare la cartella stessa
    for filename in os.listdir(output_dir):
        file_path = os.path.join(output_dir, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f'Impossibile eliminare {file_path}. Motivo: {e}')

# --- SCRITTURA FILE ---
for inst in instances:
    # Usa os.path.join per gestire correttamente i separatori di Windows (\)
    filename = os.path.join(output_dir, f"{inst[0]}-knapsack-n{inst[1]}-w{inst[2]}-v{inst[3]}-{inst[6]}.asp")
    
    with open(filename, 'w') as f:
        # Scrittura oggetti
        for idx in range(len(inst[5])):
            f.write(f"object({idx+1},{inst[4][idx]},{inst[5][idx]}).\n")
        
        # Scrittura bound (Sintassi ASP-Core-2 compatibile con AMOSUM)
        f.write(f"ub({inst[2]},0).\n")
        f.write(f"lb({inst[3]},1).\n")

print(f"Generazione completata: {ind} istanze create con successo.")

 
""" 
Esempio di istanza generata:
n5: Ci sono 5 oggetti (o meglio, 5 categorie di oggetti) nella stanza tra cui puoi scegliere.
w1000 (peso): È la capacità massima del tuo zaino. Puoi prendere oggetti fino a un peso totale di 1000 (è il limite superiore, o Upper Bound).
v500 (valore): È il tuo obiettivo minimo. Il bottino che metti nello zaino deve valere almeno 500 (è il limite inferiore, o Lower Bound).
type3: È la "difficoltà" dell'esame.

"""