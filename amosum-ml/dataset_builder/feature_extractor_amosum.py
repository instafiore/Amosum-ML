#!/usr/bin/python3
import pandas as pd
import re
import os

"""
Script for extracting the main amosum features for both the training and ijcai datasets,
starting from the preprocessed base datasets (e.g., training_dataset_measp_amo.csv).
"""
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Define the datasets and target types to process
DATASETS = ["training", "ijcai"]
TARGETS = ["AMO", "EO"]

def get_domain_dirs(dataset_type):
    """
    Generates the dynamic paths to the instances directories based on the dataset type.
    Assumes directories are named like 'training-instances-gc' or 'ijcai-instances-gc'.
    """
    return {
        "GraphColouring": os.path.join(BASE_DIR, f"../GraphColouring/{dataset_type}-instances-gc"),
        "GroupAssignment": os.path.join(BASE_DIR, f"../GroupAssignment/{dataset_type}-instances-ga"),
        "Knapsack": os.path.join(BASE_DIR, f"../Knapsack/{dataset_type}-instances-k")
    }

# ==========================================
# DOMAIN FEATURE EXTRACTION FUNCTIONS
# ==========================================

def extract_wgc_features(instance_path, filename):
    match = re.search(r'-n(\d+)-', filename)
    N = int(match.group(1)) if match else 20

    with open(instance_path, 'r') as f:
        content = f.read()
    
    bound_match = re.search(r'lb\((\d+)', content)
    bound = float(bound_match.group(1)) if bound_match else 0.0

    C1_mps = N * 64
    C2_sum = N * 94
    C3_minps = N * 2

    return bound, C1_mps, C2_sum, C3_minps

def extract_ga_features(instance_path, filename):
    with open(instance_path, 'r') as f:
        content = f.read()

    salaries_map = {
        "full_professor": 70,
        "associate_professor": 50,
        "researcher": 30
    }

    total_salary = 0
    for match in re.finditer(r'role\((\d+),\s*(\w+)\)\.', content):
        role = match.group(2)
        total_salary += salaries_map.get(role, 30)

    project_duration = {}
    for match in re.finditer(r'project_month\((\d+),\s*(\d+),\s*(\d+)\)\.', content):
        pro = int(match.group(1))
        sm = int(match.group(2))
        em = int(match.group(3))
        project_duration[pro] = em - sm + 1

    total_lb = 0.0
    total_c1 = 0.0
    total_c2 = 0.0
    total_c3 = 0.0

    max_hours = 125
    min_hours = 40
    sum_hours = 40 + 60 + 80 + 100 + 125

    for match in re.finditer(r'project\((\d+),\s*(\d+),\s*(\d+)\)\.', content):
        pro = int(match.group(1))
        lb = float(match.group(2))
        
        dur = project_duration.get(pro, 1)

        c1 = total_salary * max_hours * dur
        c2 = total_salary * sum_hours * dur
        c3 = total_salary * min_hours * dur

        total_lb += lb
        total_c1 += c1
        total_c2 += c2
        total_c3 += c3

    return total_lb, total_c1, total_c2, total_c3

def extract_kc_features(instance_path, filename):
    with open(instance_path, 'r') as f:
        content = f.read()
    
    bound_match = re.search(r'lb\((\d+),\s*1\)\.', content)
    bound = float(bound_match.group(1)) if bound_match else 0.0
    
    total_base_value = 0
    for match in re.finditer(r'object\(\d+,\s*\d+,\s*(\d+)\)\.', content):
        total_base_value += int(match.group(1))
        
    k_max = 20
    k_min = 1
    
    C1_mps = total_base_value * k_max
    C2_sum = total_base_value * (k_max * (k_max + 1) / 2)
    C3_minps = total_base_value * k_min
    
    return bound, C1_mps, C2_sum, C3_minps

# ==========================================
# MAIN PROCESSING LOOP
# ==========================================

def calculate_features(input_path, output_path, target_type, domains_dirs):
    print(f"   [{target_type}] Loading base dataset: {input_path}")
    if not os.path.exists(input_path):
        print(f"   ⚠️ Error: The file {input_path} does not exist. Skipping.")
        return

    df = pd.read_csv(input_path)
    new_features = []

    print(f"   [{target_type}] Analyzing ASP files for feature extraction...")
    
    for index, row in df.iterrows():
        domain = row['domain']
        filename = row['instance_name']
        
        # Dynamically fetch the instance path based on the generated domains_dirs
        instance_path = os.path.join(domains_dirs.get(domain, ""), filename)
        
        if not os.path.exists(instance_path):
            b, c1, c2, c3 = 0.0, 1.0, 1.0, 0.0
        else:
            if domain == "GraphColouring":
                b, c1, c2, c3 = extract_wgc_features(instance_path, filename)
            elif domain == "Knapsack":
                b, c1, c2, c3 = extract_kc_features(instance_path, filename)
            elif domain == "GroupAssignment":
                b, c1, c2, c3 = extract_ga_features(instance_path, filename)
            else:
                b, c1, c2, c3 = 0.0, 1.0, 1.0, 0.0

        # Base features always calculated
        tightness = b / c2 if c2 > 0 else 0
        
        feature_dict = {
            'bound': b,
            'C2_sum': c2,
            'tightness': tightness
        }
        
        # Add specific features based on the target type
        if target_type == 'AMO':
            rho = c1 / c2 if c2 > 0 else 0
            mps_ratio = b / c1 if c1 > 0 else 0
            feature_dict['C1_mps'] = c1
            feature_dict['rho'] = rho
            feature_dict['mps_ratio'] = mps_ratio
        elif target_type == 'EO':
            span = c1 - c3
            minps_ratio = b / c3 if c3 > 0 else 0
            feature_dict['C3_minps'] = c3
            feature_dict['minps_ratio'] = minps_ratio
            feature_dict['span'] = span

        new_features.append(feature_dict)

    print(f"   [{target_type}] Merging data and saving to {output_path}...")
    df_features = pd.DataFrame(new_features)
    
    # Merge with the features of the original dataframe
    df_final = pd.concat([df, df_features], axis=1)
    df_final.to_csv(output_path, index=False)
    print(f"   [{target_type}] Completed!\n")

def main():
    # Iterate through both 'training' and 'ijcai' datasets
    for dataset in DATASETS:
        print(f"==================================================")
        print(f"🚀 STARTING PIPELINE FOR: {dataset.upper()}")
        print(f"==================================================")
        
        # Generate the correct folder paths for the current dataset
        domains_dirs = get_domain_dirs(dataset)
        
        # Iterate through both target types (AMO and EO)
        for target in TARGETS:
            # Construct dynamic input and output paths
            input_file = os.path.join(BASE_DIR, f"preprocessed/{dataset}_dataset_measp_{target.lower()}.csv")
            output_file = os.path.join(BASE_DIR, f"preprocessed/{dataset}_dataset_{target.lower()}.csv")
            
            calculate_features(input_file, output_file, target, domains_dirs)
            
    print("✅ All feature extractions have been successfully completed.")

if __name__ == "__main__":
    main()