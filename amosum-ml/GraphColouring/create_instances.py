#!/usr/bin/python3
import os
import shutil
import random

# --- PATH DI CONFIGURAZIONE PER WSL ---
base_dir = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(base_dir, "training-instances-gc")

# --- CONFIGURAZIONI DEL GRAFO (Modello Erdős-Rényi) ---
NODES_LIST = [20, 40, 60, 80, 100]
DENSITIES = [0.3, 0.5, 0.7] # Probabilità di connessione tra due nodi
GRAPHS_PER_CONFIG = 10      # Grafi generati per ogni combinazione (nodi, densità)

base_percentages = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]

COLORS = {
    'red0': 2,
    'green0': 4,
    'blue0': 8,
    'yellow0': 16,
    'cyan0': 64
}
MAX_WEIGHT = max(COLORS.values()) # 64

def setup_directory(directory_path):
    """Pulisce la cartella di output in modo sicuro su Windows/WSL."""
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
    else:
        for filename in os.listdir(directory_path):
            file_path = os.path.join(directory_path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f'Impossibile eliminare {file_path}. Motivo: {e}')

def generate_random_graph(num_nodes, density):
    """Genera gli archi di un grafo casuale con la densità specificata."""
    edges = []
    for i in range(1, num_nodes):
        for j in range(i + 1, num_nodes + 1):
            if random.random() < density:
                edges.append((i, j))
    return edges

def main():
    print(f"Generazione Training Set WGC in corso in:\n{OUTPUT_DIR} ...")
    setup_directory(OUTPUT_DIR)
    
    count = 0
    graph_id = 1
    
    for num_nodes in NODES_LIST:
        for density in DENSITIES:
            for _ in range(GRAPHS_PER_CONFIG):
                
                # 1. Generiamo la topologia del grafo una volta per ogni configurazione
                edges = generate_random_graph(num_nodes, density)
                lb_max = num_nodes * MAX_WEIGHT
                
                # 2. Creiamo le 9 varianti di "bound" (con jitter) per questo specifico grafo
                for per in base_percentages:
                    jitter = random.uniform(-0.02, 0.02)
                    final_per = max(0.01, min(0.99, per + jitter)) 
                    lb_p = int(lb_max * final_per)
                    
                    # Nome file: es. 0001-wgc-n40-d50-p31.asp
                    out_filename = f"{graph_id:04d}-wgc-n{num_nodes}-d{int(density*100)}-p{int(final_per*100)}.asp"
                    out_filepath = os.path.join(OUTPUT_DIR, out_filename)
                    
                    with open(out_filepath, 'w') as f_out:
                        # Scrittura dei Nodi (riga per riga come nel test set)
                        for i in range(1, num_nodes + 1):
                            f_out.write(f"node({i}).\n")
                        
                        # Scrittura degli Archi (bidirezionali con predicato 'link')
                        for u, v in edges:
                            f_out.write(f"link({u},{v}).\n")
                            f_out.write(f"link({v},{u}).\n")
                        
                        # Scrittura di Colori
                        for color in COLORS.keys():
                            f_out.write(f"colour({color}).\n")
                        
                        # Scrittura della soglia (formato: lb(Valore, 0).)
                        f_out.write(f"lb({lb_p},0).\n")
                            
                        # Scrittura dei pesi dei colori (predicato 'colour_weight')
                        for color, weight in COLORS.items():
                            f_out.write(f"colour_weight({color}, {weight}).\n")
                    
                    count += 1
                graph_id += 1
                
    print(f"Generazione completata: create {count} istanze di training WGC sintetiche.")

if __name__ == "__main__":
    main()