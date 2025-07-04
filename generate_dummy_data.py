import pandas as pd
import numpy as np

N_PATIENTS = 200
N_BIOMARKERS = 36
CONTROL_RATIO = 0.25  # 25% controls

np.random.seed(42)

patients = [f"P{str(i+1).zfill(3)}" for i in range(N_PATIENTS)]
is_control = np.random.choice([0, 1], size=N_PATIENTS, p=[1 - CONTROL_RATIO, CONTROL_RATIO])

ages = np.random.randint(19, 44, size=N_PATIENTS)
bmis = np.round(np.random.uniform(19, 34, size=N_PATIENTS), 1)

# rASRM for non-controls, empty for controls
rasrm_stages = [
    np.random.choice(['I', 'II', 'III', 'IV'], p=[0.07, 0.26, 0.36, 0.31]) if not c else ""
    for c in is_control
]

# Leiomyoma: ~70% prevalence, regardless of group
leiomyomas = np.random.choice(['Yes', 'No'], size=N_PATIENTS, p=[0.7, 0.3])

# Enzian: core (0-3) and extra (0/1) for cases, all 0 for controls
enzian_core = ['P', 'O', 'T', 'A', 'B', 'C']
enzian_extra = ['Fa', 'Fu', 'Fi', 'Fother']
enzian_data = {}

for cat in enzian_core:
    enzian_data[f'Enzian_{cat}'] = [
        np.random.randint(0, 4) if not c else 0 for c in is_control
    ]
for cat in enzian_extra:
    enzian_data[f'Enzian_{cat}'] = [
        np.random.randint(0, 2) if not c else 0 for c in is_control
    ]

# Biomarkers: generate normally for all
biomarker_names = [f"Cytokine_{i+1}" for i in range(N_BIOMARKERS)]
biomarker_data = np.round(np.abs(np.random.normal(loc=5, scale=2, size=(N_PATIENTS, N_BIOMARKERS))), 2)

# Group label
groups = ["Control" if c else "Endometriosis" for c in is_control]

# Assemble DataFrame
df = pd.DataFrame({
    'Patient_ID': patients,
    'Group': groups,
    'Age': ages,
    'BMI': bmis,
    'rASRM_stage': rasrm_stages,
    'Leiomyoma': leiomyomas
})

# Add Enzian columns
for col in enzian_data:
    df[col] = enzian_data[col]

# Add biomarker columns
for idx, biomarker in enumerate(biomarker_names):
    df[biomarker] = biomarker_data[:, idx]

# Save to CSV
df.to_csv("data/example/dummy_data.csv", index=False)
