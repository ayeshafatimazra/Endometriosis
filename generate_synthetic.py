import pandas as pd
import numpy as np
import os

# 1. Read biomarker summary statistics from CSVs
def read_biomarker_stats():
    csv_files = {
        'DIE': 'DIE_Mean and Standard Deviation.csv',
        'Ovary': 'Ovary_ Mean and Standard Deviation.csv',
        'Peritoneum': 'Peritoneum_Mean and Standard Deviation.csv',
        'Uterus': 'Uterus_ Mean and Standard Deviation.csv'
    }
    biomarker_stats = {}
    for cls, file in csv_files.items():
        df = pd.read_csv(file)
        biomarker_stats[cls] = df.set_index(df.columns[0])[['Mean', 'SD']]
    return biomarker_stats

# 2. Simulate synthetic patients for each class
def simulate_patients(biomarker_stats, n_patients_per_class=1000):
    synthetic_data = []
    for cls, stats in biomarker_stats.items():
        means = stats['Mean'].values
        sds = stats['SD'].values
        biomarkers = stats.index.tolist()
        samples = np.random.normal(loc=means, scale=sds, size=(n_patients_per_class, len(biomarkers)))
        for i in range(n_patients_per_class):
            patient = {
                'patient_id': f'{cls}_{i+1}',
                'class': cls
            }
            patient.update({biomarker: samples[i, j] for j, biomarker in enumerate(biomarkers)})
            synthetic_data.append(patient)
    return synthetic_data

# 3. Combine all patients into a single DataFrame
def create_patients_df(synthetic_data):
    df_patients = pd.DataFrame(synthetic_data)
    return df_patients

# 4. Add synthetic 'time_to_diagnosis' variable
def add_time_to_diagnosis(df_patients):
    np.random.seed(42)  # For reproducibility
    df_patients['time_to_diagnosis'] = np.random.lognormal(mean=2, sigma=0.5, size=len(df_patients))
    df_patients['time_to_diagnosis'] = df_patients['time_to_diagnosis'].round(2)
    return df_patients

# 5. Save the synthetic dataset
def save_dataset(df_patients, out_path='data/synthetic_endometriosis.csv'):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df_patients.to_csv(out_path, index=False)
    print(f'Saved to {out_path}')

if __name__ == '__main__':
    biomarker_stats = read_biomarker_stats()
    synthetic_data = simulate_patients(biomarker_stats)
    df_patients = create_patients_df(synthetic_data)
    df_patients = add_time_to_diagnosis(df_patients)
    save_dataset(df_patients) 