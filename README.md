<p align="center">
  <img src="EndoSign_logo.png" alt="EndoSign Logo" width="300"/>
</p>

<h2 align="center">Endometriosis Plasma Biomarker Analysis using dimensionality reduction and classification of plasma biomarkers for endometriosis.</h2>

# Background & Biomedical Problem
Endometriosis is a complex, heterogeneous gynecological disorder characterized by the presence of endometrial-like tissue outside the uterus. Affecting up to 10% of reproductive-age women, it is associated with chronic pelvic pain, infertility, and reduced quality of life. Diagnosis remains challenging, often delayed for years, and current clinical staging systems (such as rASRM) although of high importance for patient follow-up, are unable to capture the whole diversity of the disease.

Emerging evidence shows that inflammatory plasma biomarkers and novel clinical scoring systems, such as the #Enzian classification system, could provide a more granular understanding of endometriosis phenotypes. However, integrating and analyzing these multi-dimensional datasets to uncover clinically meaningful patient subgroups and diagnostic markers remains a significant challenge.

# Project Overview 

EndoSign is an open-source, Python-based framework for the analysis, visualization, and classification of plasma biomarker profiles in endometriosis cohorts. 

## The platform enables:

- Integration of multi-omics biomarker data and detailed clinical phenotypes (including #Enzian and rASRM scores)
- Dimensionality reduction (UMAP, PCA, t-SNE) for data visualization and patient landscape mapping
- Unsupervised clustering to identify data-driven subtypes
- Supervised classification for biomarker-based patient stratification
- Visual analytics and reporting

All pipelines are designed to protect patient confidentiality, allowing for the use of dummy/example datasets and straightforward extension to external cohorts.

# Getting Started (installation, requirements)

# How to Upload and Format Your Own Data

# Pipeline of Analysis
  1. **Data Ingestion and Validation**
    - Import patient and biomarker datasets (CSV, tabular formats)
    - Validate clinical fields (Enzian, rASRM, group, leiomyoma status, etc.)
    - Data Cleaning and Preprocessing

  2. **Missing value handling and imputation**
  - Outlier removal (with traceability and export of outlier records)
  - Normalization and (optionally) log-transformation of biomarker values

  3. **Exploratory Data Analysis (EDA) and Visualization**
  - Cohort summaries: group, rASRM, Enzian category distributions
  - Biomarker value distributions, correlation heatmaps, Venn diagrams of comorbidities
  - Dimensionality reduction plots (UMAP/PCA/t-SNE, 2D and 3D), colored by clinical features
  
  4. **Clustering and Subgroup Identification**
  - UMAP/PCA embedding on Enzian and/or biomarker dimensions
  - Unsupervised clustering (KMeans, DBSCAN, etc.) to reveal novel patient clusters
  - Association of clusters with clinical variables and biomarker signatures
  5. **Supervised Classification and Feature Selection**
  - Machine learning models (Random Forest, Logistic Regression, XGBoost, etc.)
  - Feature importance analysis, ROC/AUC, and cross-validation
  - Support for multi-class classification (e.g., rASRM or Enzian subtypes)
  6. **Reporting and Export**
  - Automated data profiling reports (HTML)
  - Publication-ready plots and tables (png, pdf, csv)
  - Documentation for reproducing the analysis with external datasets

# Example Workflows (link to notebooks)

# Contribution Guidelines

# References and Acknowledgments
