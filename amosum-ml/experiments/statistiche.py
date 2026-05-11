import csv
import os

def load_csv(filepath):
    data = []
    with open(filepath, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append(row)
    return data

def verifica_scelte_ml(tipo, log_file):
    def log(msg):
        print(msg)
        log_file.write(msg + '\n')

    val_file = f'esperimento_7/validazione_finale_automl_L2_{tipo}.csv'
    test_file = f'test_set_{tipo}_20perc.csv'
    
    if not os.path.exists(val_file) or not os.path.exists(test_file):
        log(f"Errore: file mancanti per {tipo}. Verifica che {val_file} e {test_file} esista.")
        return
        
    val_data = load_csv(val_file)
    test_data = load_csv(test_file)
    
    # Crea un mapping instanza -> best_config
    test_dict = {row['instance_name']: row['best_config_TRUE'] for row in test_data if 'instance_name' in row and 'best_config_TRUE' in row}
    
    totali_maggiori = 0
    scelte_corrette = 0
    scelte_errate = []
    
    for row in val_data:
        instance = row.get('instance')
        if not instance or instance not in test_dict:
            continue
            
        ml_time = float(row.get('ml_time', 0))
        min_time_reale = float(row.get('min_time_reale', 0))
        ml_config = row.get('ml_config')
        best_config_true = test_dict[instance]
        
        if ml_time > min_time_reale:
            totali_maggiori += 1
            if ml_config == best_config_true:
                scelte_corrette += 1
            else:
                scelte_errate.append((instance, ml_time, min_time_reale, ml_config, best_config_true))
                
    if totali_maggiori == 0:
        log(f"\n[{tipo}] Nessuna istanza trovata in cui ml_time > min_time_reale.")
        return
        
    percentuale = (scelte_corrette / totali_maggiori) * 100
    
    log(f"\n--- Report per {tipo} ---")
    log(f"Numero totale di istanze in cui ml_time > min_time_reale: {totali_maggiori}")
    log(f"Di queste, istanze in cui ml_config == best_config_TRUE: {scelte_corrette}")
    log(f"Percentuale: {percentuale:.2f}%")
    
    if scelte_errate:
        log(f"\nEsempio di {len(scelte_errate)} scelte non corrispondenti alla 'best_config_TRUE':")
        log(f"{'Instance'.ljust(45)} | {'ml_time':>10} | {'min_time':>10} | {'ml_config':>15} | {'best_config_TRUE':>15}")
        log("-" * 105)
        for item in scelte_errate[:10]:
            log(f"{item[0][:40].ljust(45)} | {item[1]:10.2f} | {item[2]:10.2f} | {item[3]:>15} | {item[4]:>15}")

def verifica_unsat(tipo, log_file):
    def log(msg):
        print(msg)
        log_file.write(msg + '\n')

    val_file = f'esperimento_7/validazione_finale_automl_L2_{tipo}.csv'
    if not os.path.exists(val_file):
        return

    val_data = load_csv(val_file)
    
    non_risolte = []
    for row in val_data:
        solved_str = row.get('solved')
        ml_time = float(row.get('ml_time', 1200))
        
        solved = (solved_str == 'True')
        solved_unsat = (solved_str == 'False' and ml_time < 1200)
        
        # Aumenta le non risolte dal ML solo se entrambe sono false (tempo sforato)
        if not solved and not solved_unsat:
            non_risolte.append(row)
    
    if not non_risolte:
        log(f"\n[{tipo}] Nessuna istanza non risolta dal machine learning.")
        return

    log(f"\n--- Analisi Istanze Non Risolte (effettivamente UNSAT?) per {tipo} ---")
    log(f"Trovate {len(non_risolte)} istanze non risolte dal ML.")
    
    base_train_dir = "/home/guests/mmacri/AMOSUM/benchmarks/training_benchmarks_results"
    base_test_dir = "/home/guests/mmacri/AMOSUM/benchmarks/testing_benchmarks_results"
    
    unsat_count = 0
    sat_count = 0
    unknown_count = 0
    
    # Cache per evitare di rileggere i node_info.log per istanze dello stesso problema
    # mapping_istanze[problem][instance_name] = directory (es: /percorso/.../config2/instanceX)
    mapping_istanze = {}

    for row in non_risolte:
        problem = row.get('problem')
        instance = row.get('instance')
        
        if problem not in mapping_istanze:
            mapping_istanze[problem] = {}
            # Cerca in training
            train_dir = os.path.join(base_train_dir, f"AMOSUM_{problem}_old")
            for conf_test in ["config2", "config3"]: # Usa una config per trovare l'id
                conf_dir = os.path.join(train_dir, conf_test)
                if os.path.exists(conf_dir):
                    for inst_dir in os.listdir(conf_dir):
                        if inst_dir.startswith("instance"):
                            node_log = os.path.join(conf_dir, inst_dir, "run1", "node_info.log")
                            if os.path.exists(node_log):
                                with open(node_log, 'r') as f:
                                    content = f.read()
                                    # Cerca Input: percorso/instance_name
                                    for line in content.splitlines():
                                        if line.startswith("Input:"):
                                            found_instance = os.path.basename(line.strip())
                                            mapping_istanze[problem][found_instance] = (train_dir, inst_dir)
                                            # per compatibilità salviamo anche la riga intera solo col nome
                                            mapping_istanze[problem][line.strip()] = (train_dir, inst_dir)
                
            # Cerca in testing
            test_dir = os.path.join(base_test_dir, f"AMOSUM_{problem}")
            for conf_test in ["config2", "config3"]:
                conf_dir = os.path.join(test_dir, conf_test)
                if os.path.exists(conf_dir):
                    for inst_dir in os.listdir(conf_dir):
                        if inst_dir.startswith("instance"):
                            node_log = os.path.join(conf_dir, inst_dir, "run1", "node_info.log")
                            if os.path.exists(node_log):
                                with open(node_log, 'r') as f:
                                    content = f.read()
                                    for line in content.splitlines():
                                        if line.startswith("Input:"):
                                            found_instance = os.path.basename(line.strip())
                                            mapping_istanze[problem][found_instance] = (test_dir, inst_dir)
                                            mapping_istanze[problem][line.strip()] = (test_dir, inst_dir)
        
        if instance in mapping_istanze[problem]:
            full_dir, inst_dir = mapping_istanze[problem][instance]
        else:
            log(f"Impossibile trovare node_info.log per l'istanza: {instance}")
            continue
        
        configs = ["config2", "config3", "config4", "config5", "config6"]
        status = "UNKNOWN"
        
        for conf in configs:
            log_file_path = os.path.join(full_dir, conf, inst_dir, "run1", "stdout.log")
            if os.path.exists(log_file_path):
                try:
                    with open(log_file_path, 'r', errors='ignore') as f:
                        log_content = f.read().lower()
                        # Evitiamo di cercare solo "unsat" perché clingo stampa "Unsat: 0.00s" nelle statistiche finali
                        if "unsatisfiable\n" in log_content or "\nunsatisfiable" in log_content or "incoherent" in log_content or "\nunsat\n" in log_content:
                            status = "UNSAT"
                            break
                        elif "\nanswer set" in log_content or "answer set [" in log_content or "answer: 1" in log_content or "satisfiable\n" in log_content or "\nsatisfiable" in log_content:
                            status = "SAT"
                            break
                except Exception as e:
                    pass
                    
        if status == "UNSAT":
            unsat_count += 1
            log(f"[UNSAT] {problem} - {instance}")
        elif status == "SAT":
            sat_count += 1
            log(f"[SAT] {problem} - {instance}")
        else:
            unknown_count += 1
            log(f"[UNKNOWN] {problem} - {instance}")
            
    log(f"\nRiepilogo {tipo} non risolte dal ML:")
    log(f"Effettivamente UNSAT: {unsat_count}")
    log(f"Effettivamente SAT: {sat_count}")
    log(f"Sempre UNKNOWN (non risolte da nessuna config): {unknown_count}")


def conteggio_risolte_testset(tipo, log_file):
    def log(msg):
        print(msg)
        log_file.write(msg + '\n')

    val_file = f'esperimento_7/validazione_finale_automl_L2_{tipo}.csv'
    if not os.path.exists(val_file):
        return

    # testset
    test_file = f'test_set_{tipo}_20perc.csv'
    test_data = load_csv(test_file)
    test_instances = set(row['instance_name'] for row in test_data if 'instance_name' in row)
    
    val_data = load_csv(val_file)
    problem_stats = {}
    
    # Group instanze by problem e mapping
    for row in val_data:
        instance = row.get('instance')
        problem = row.get('problem')
        if not instance or instance not in test_instances:
            continue
            
        solved_str = row.get('solved')
        ml_time = float(row.get('ml_time', 1200))
        
        solved_sat = (solved_str == 'True')
        solved_unsat = (solved_str == 'False' and ml_time < 1200)
        
        solved_ml = solved_sat or solved_unsat
        
        if problem not in problem_stats:
            problem_stats[problem] = {'ml_solved': 0, 'total': 0, 'conf_solved': {}}
            
        problem_stats[problem]['total'] += 1
        if solved_ml:
            problem_stats[problem]['ml_solved'] += 1
            
        # Per le config guardiamo nei vecchi risultati (results.csv)
        num_str = instance.split('-')[0]
        try:
            instance_id = str(int(num_str))
        except:
            instance_id = None
        
            
    base_train_dir = "/home/guests/mmacri/AMOSUM/benchmarks/training_benchmarks_results"
    base_test_dir = "/home/guests/mmacri/AMOSUM/benchmarks/testing_benchmarks_results"
    
    if tipo == 'AMO':
        target_configs = {'CLINGO': '2', 'AMOCLINGO-INF-MR': '3', 'AMOCLINGO-L': '4'}
    elif tipo == 'EO':
        target_configs = {'CLINGO': '2', 'EOCLINGO-INF-MR': '5', 'EOCLINGO-L': '6'}
    else:
        target_configs = {}
        
    for problem, stats in problem_stats.items():
        for conf_name in target_configs.keys():
            stats['conf_solved'][conf_name] = 0
            
        old_dir = f"AMOSUM_{problem}_old"
        new_dir = f"AMOSUM_{problem}"
        
        results_files = [
            os.path.join(base_train_dir, old_dir, "results.csv"),
            os.path.join(base_test_dir, new_dir, "results.csv")
        ]
        
        for results_file in results_files:
            if os.path.exists(results_file):
                results_data = load_csv(results_file)
                for row in results_data:
                    conf_id = row.get('config_id')
                    inst_path = row.get('instance')
                    time_val = float(row.get('perf_seconds-time-elapsed', 1200))
                    
                    # check if instance in testset
                    inst_name = os.path.basename(inst_path) if inst_path else ""
                    
                    for conf_name, target_id in target_configs.items():
                        if conf_id == target_id and inst_name in test_instances and time_val < 1200:
                            stats['conf_solved'][conf_name] += 1
                        
    log(f"\n--- Confronto Istanze Risolte (Singole Config vs ML) per {tipo} ---")
    for problem, stats in problem_stats.items():
        log(f"Problema: {problem}")
        log(f"  Totale Istanze Testset: {stats['total']}")
        log(f"  Risolte dal ML: {stats['ml_solved']}")
        for conf_name, count in stats['conf_solved'].items():
            log(f"  Risolte da {conf_name}: {count}")

if __name__ == "__main__":
    os.chdir('/home/guests/mmacri/AMOSUM/benchmarks/test_finale/')
    
    output_filename = 'esperimento_7/report_statistiche_ml.txt'
    
    with open(output_filename, 'w') as log_file:
        def log_main(msg):
            print(msg)
            log_file.write(msg + '\n')
            
        log_main("Analisi validazione finale rispetto ai test set 20%\n" + "="*50)
        verifica_scelte_ml('AMO', log_file)
        verifica_unsat('AMO', log_file)
        conteggio_risolte_testset('AMO', log_file)
        
        verifica_scelte_ml('EO', log_file)
        verifica_unsat('EO', log_file)
        conteggio_risolte_testset('EO', log_file)

        conteggio_risolte_testset('CLINGO', log_file)
        
    print(f"\nReport salvato con successo in: {output_filename}")
