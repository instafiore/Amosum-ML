#!/usr/bin/python3
import re
import os
import sys
import shutil
import random
from typing import List

import numpy as np

# --- PATH DI OUTPUT PER WSL ---
base_dir = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(base_dir, "training-instances-ga")

class InstanceFactory:
    map_roles = {
        "full_professor": (70, 0.17),
        "associate_professor": (50, 0.64),
        "researcher": (30, 0.19),
    }

    SIGMA_P = 0.001
    max_hours_working_per_month = 125
    months = 12

    def __init__(self, num_projects, num_people):
        self.num_projects = num_projects
        self.num_people = num_people
        self.map_person_role = dict()
        self.map_project_duration = dict()
        self.instance = []

    def __create_roles(self):
        start_index_partition = 1
        comulative_percentage = 0
        for role in self.map_roles:
            p = self.map_roles[role][1]
            comulative_percentage += p
            quantile = round(self.num_people * comulative_percentage)
            end_index_partition = quantile 
            for person in range(start_index_partition, end_index_partition+1):
                self.map_person_role[person] = role
                self.instance.append(f"role({person}, {role}).")
            start_index_partition = end_index_partition+1

    def __create_projects_duration(self):
        for pro in range(1, self.num_projects+1):
            value = random.randint(1, 100)
            if value <= 25:
                duration = 3
            elif value <= 60:
                duration = 6
            else:
                duration = 12
            self.map_project_duration[pro] = duration
            max_start = InstanceFactory.months - duration + 1
            sm = random.randint(1, max_start)
            self.instance.append(f"project_month({pro}, {sm}, {sm+duration-1}).")

    def create_instance(self):
        self.instance = []
        self.__create_projects_duration()
        self.__create_roles()
        self.__create_project_bounds()
        return self.instance
    
    def __create_mps(self, pro):
        mps = 0
        for per in range(1, self.num_people+1):
            role = self.map_person_role.get(per, "researcher") # Fallback di sicurezza
            salary = self.map_roles[role][0]
            for _ in range(self.map_project_duration[pro]):
                mps += salary * InstanceFactory.max_hours_working_per_month
        return mps    

    def __create_project_bounds(self):
        for pro in self.map_project_duration:
            mps = self.__create_mps(pro=pro)
            lb = round(self._create_bound(mps))
            ub = lb + round(lb*0.2)
            self.instance.append(f"project({pro}, {lb}, {ub}).")

    def __repr__(self):
        return "middle"
    
    def __str__(self):
        return self.__repr__()
    
    def _create_bound(self, mps):
        mu = mps * 1 / self.num_projects
        sigma = mu * InstanceFactory.SIGMA_P
        return np.random.normal(mu, sigma)

class SatInstanceFactory(InstanceFactory):      
    def _create_bound(self, mps):
        mu = mps * 1 / self.num_projects * 1 / 4
        sigma = mu * InstanceFactory.SIGMA_P
        return np.random.normal(mu, sigma)  
    
    def __repr__(self):
        return "sat"
    
class PossiblyUnsatInstanceFactory(InstanceFactory):
    def _create_bound(self, mps):
        mu = mps * 0.9
        sigma = mu * InstanceFactory.SIGMA_P
        return np.random.normal(mu, sigma)  
    
    def __repr__(self):
        return "punsat"
    
class UnsatInstanceFactory(InstanceFactory):
    def _create_bound(self, mps):
        mu = mps + 1
        sigma = mu * InstanceFactory.SIGMA_P
        return np.random.normal(mu, sigma)  
    
    def __repr__(self):
        return "unsat"
    

class BenchmarkCreator():
    # Applicato uno silanciamento nella generazione verso le istanze 'middle' e 'punsat'
    # perché sono i casi limite dove il classificatore imparerebbe di più.
    instances_type_distribution = {
        "sat": (SatInstanceFactory, 5),
        "middle": (InstanceFactory, 10),
        "possibly_unsat": (PossiblyUnsatInstanceFactory, 10),
        "unsat": (UnsatInstanceFactory, 5),
    }

    def __init__(self, project_configurations, people_configurations, output_dir):
        self.people_configurations = people_configurations
        self.project_configurations = project_configurations
        self.output_dir = output_dir
        self.instances = []

    def print_instance(self, factory: InstanceFactory, instance_id):
        common_lines = []
        common_lines.append(f"person(1..{factory.num_people}).")
        instance = factory.create_instance()
        self.instances.extend(instance)
        
        # Formattazione nome file coerente e pulita
        file_name = f"{instance_id:05d}-group_assignment-p{factory.num_people}-pr{factory.num_projects}-{factory}.asp"
        file_path = os.path.join(self.output_dir, file_name)
        
        with open(file_path, "w") as file:
            file.write("\n".join(instance))
            file.write("\n")
            file.write("\n".join(common_lines))

    def create_benchmark(self):
        print(f"Generazione Training Set in corso in:\n{self.output_dir} ...")
        self.instances = []
        instance_id = 1
        
        for nproj in self.project_configurations:
            for nper in self.people_configurations:
                for instance_type in BenchmarkCreator.instances_type_distribution:
                    Factory, n = BenchmarkCreator.instances_type_distribution[instance_type]
                    for _ in range(n):
                        factory = Factory(num_projects=nproj, num_people=nper)
                        self.print_instance(factory, instance_id)
                        instance_id += 1
                        
        print(f"Generazione completata: creati {instance_id - 1} file.")

def setup_directory(directory_path):
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

def main():
    setup_directory(OUTPUT_DIR)
    
    # Griglia molto più fitta per avere un numero di istanze ideale per il Machine Learning
    # Progetti: da 5 a 40 (salti di 5)
    project_configurations = range(5, 45, 5) 
    # Persone: da 50 a 300 (salti di 50)
    people_configurations = range(50, 350, 50) 
    
    # Questo genererà: 8 (prog) * 6 (pers) * 30 (istanze per config) = 1440 istanze totali.
    
    benchmarkCreator = BenchmarkCreator(project_configurations, people_configurations, OUTPUT_DIR)
    benchmarkCreator.create_benchmark()

if __name__ == "__main__":
    main()