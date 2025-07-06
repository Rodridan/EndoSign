import pandas as pd
import numpy as np

N_PATIENTS = 200
N_BIOMARKERS = 36
CONTROL_RATIO = 0.25

np.random.seed(42)

patients = [f"P{str(i+1).zfill(3)}" for i in range(N_PATIENTS)]
is_control = np.random.choice([0, 1], size=N_PATIENTS, p=[1 - CONTROL_RATIO, CONTROL_RATIO])

ages = np.random.randint(19, 44, size=N_PATIENTS)
bmis = np.round(np.random.uniform(19, 34, size=N_PATIENTS), 1)

# Disease progression stages for non-controls
rasrm_choices = ['I', 'II', 'III', 'IV']
rasrm_probs = [0.07, 0.26, 0.36, 0.31]
rasrm_stage_num = []
rasrm_stages = []
for c in is_control:
    if not c:
        stage = np.random.choice(rasrm_choices, p=rasrm_probs)
        rasrm_stages.append(stage)
        rasrm_stage_num.append(rasrm_choices.index(stage) + 1)
    else:
        rasrm_stages.append("")
        rasrm_stage_num.append(0)

leiomyomas = np.random.choice(['Yes', 'No'], size=N_PATIENTS, p=[0.7, 0.3])

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

# --- Biomarker simulation ---

biomarker_names = [f"Cytokine_{i+1}" for i in range(N_BIOMARKERS)]
biomarker_data = np.zeros((N_PATIENTS, N_BIOMARKERS))

# 1. Disease progression biomarkers
for i in range(N_PATIENTS):
    if is_control[i]:
        biomarker_data[i, 0] = np.random.normal(4, 0.8)  # Cytokine_1: low in controls
        biomarker_data[i, 1] = np.random.normal(4, 0.8)  # Cytokine_2: low in controls
        biomarker_data[i, 2] = np.random.normal(6, 0.8)  # Cytokine_3: high in controls
    else:
        biomarker_data[i, 0] = 4 + 0.7 * rasrm_stage_num[i] + np.random.normal(0, 0.7) # increases
        biomarker_data[i, 1] = 3.5 + 0.5 * rasrm_stage_num[i] + np.random.normal(0, 0.7) # increases
        biomarker_data[i, 2] = 7 - 0.7 * rasrm_stage_num[i] + np.random.normal(0, 0.7) # decreases

# 2. Correlated biomarker clusters
for cluster_start in [3, 8, 13]:
    cov = 0.7 * np.ones((5, 5)) + 0.3 * np.eye(5)  # strong intra-cluster correlation
    cluster_values = np.random.multivariate_normal(mean=np.full(5, 5), cov=cov, size=N_PATIENTS)
    biomarker_data[:, cluster_start:cluster_start+5] = cluster_values

# 3. Remaining cytokines
for j in range(18, N_BIOMARKERS):
    biomarker_data[:, j] = np.random.normal(5, 2, N_PATIENTS)

# --- Additional clinically relevant features ---
# Infertility: higher if endometriosis and high stage
infertility = []
for i in range(N_PATIENTS):
    if is_control[i]:
        infertility.append(np.random.choice(['Yes', 'No'], p=[0.05, 0.95]))
    else:
        stage = rasrm_stage_num[i]
        if stage >= 3:
            infertility.append(np.random.choice(['Yes', 'No'], p=[0.7, 0.3]))
        else:
            infertility.append(np.random.choice(['Yes', 'No'], p=[0.3, 0.7]))

# Pain score: higher with stage
pain_score = []
for i in range(N_PATIENTS):
    if is_control[i]:
        pain_score.append(np.clip(np.random.normal(1, 1), 0, 10))
    else:
        pain_score.append(np.clip(np.random.normal(2 + 2 * rasrm_stage_num[i], 1.2), 0, 10))
pain_score = np.round(pain_score, 1)

# Hormonal medication: most common in Endo cases, esp. with higher stage
hormonal_med = []
for i in range(N_PATIENTS):
    if is_control[i]:
        hormonal_med.append(np.random.choice(['Yes', 'No'], p=[0.15, 0.85]))
    else:
        stage = rasrm_stage_num[i]
        if stage >= 3:
            hormonal_med.append(np.random.choice(['Yes', 'No'], p=[0.7, 0.3]))
        else:
            hormonal_med.append(np.random.choice(['Yes', 'No'], p=[0.45, 0.55]))

batch = np.random.choice(['A', 'B', 'C'], size=N_PATIENTS, p=[0.4, 0.3, 0.3])

groups = ["Control" if c else "Endometriosis" for c in is_control]

df = pd.DataFrame({
    'Patient_ID': patients,
    'Group': groups,
    'Age': ages,
    'BMI': bmis,
    'rASRM_stage': rasrm_stages,
    'Leiomyoma': leiomyomas,
    'Infertility': infertility,
    'Pain_Score': pain_score,
    'Hormonal_Medication': hormonal_med,
    'Batch': batch
})

for col in enzian_data:
    df[col] = enzian_data[col]
for idx, biomarker in enumerate(biomarker_names):
    df[biomarker] = biomarker_data[:, idx]

df.to_csv("data/example/dummy_data.csv", index=False)
