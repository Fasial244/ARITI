# IAU Work — AI vs Human Content Detection

This repository contains my lab work for the Machine Learning course at IAU.  
Project theme: **AI vs Human Content Detection** (text classification using NLP + engineered features).

---

## Dataset

- **File:** `ai_human_detection_v1.csv`
- **Type:** NLP + metadata (tabular dataset with `text` + related fields)
- **Text sources/domains:** essays, emails, blog posts, social media-style posts
- **Labels (`human_or_ai`):** `human`, `ai`, `post_edited_ai`
- **Note:** Some rows contain API error messages (e.g., “Client Error”), which are removed during cleaning.

---

## Labs Overview

### Lab 2 — Identifying ML Problems, Selecting Open Datasets, Methodology Diagram
- Defined the ML task as **supervised classification**.
- Target column: `human_or_ai`.
- Primary input feature: `text`.
- Metadata is used mainly for analysis (some metadata can cause label leakage).

---

### Lab 3 — Data Understanding and Exploration (EDA)
**Key findings from EDA:**
- **Length:** AI-generated text tends to be longer on average than human text.
- **Vocabulary:** Human writing often shows higher lexical diversity.
- **Structure:** AI text tends to be more consistently structured.
- Detected **noise** such as API error rows that must be removed before modeling.

Outputs include:
- label distribution plots
- text length distributions (word/character length) by class
- basic text statistics

---

### Lab 4 — Data Quality & Preprocessing
**Tasks completed:**
1. **Data quality issues:** missing values, duplicates (if present), noisy error rows, class imbalance.
2. **Missing value strategy:** filled missing `prompt` with `"Unknown"`.
3. **Outliers (IQR):** created numeric text features and detected outliers using IQR; handled using capping.
4. **Normalization:** applied both Min-Max scaling and Z-score standardization.
5. **PCA:** applied PCA on standardized numeric features and interpreted explained variance.

---

### Lab 5 — Feature Engineering Experiments
This lab focuses on building and comparing engineered features and feature configurations.

**Task 1 — New engineered feature**
- Added: `punct_ratio` = punctuation_count / character_length  
- Justification: punctuation density helps capture stylistic consistency independent of text length.

**Task 2 — Change `is_peak_hour` rule**
- Tested alternative definitions of `is_peak_hour` using the hour extracted from `generation_date`.
- Compared performance to see if timing information affects classification.

**Task 3 — Change `top_k` vocabulary size**
- Varied TF-IDF `max_features` (e.g., 10, 30, 50, and larger values) and compared:
  - accuracy
  - top text features (via model coefficients)

**Task 4 — Optional feature selection**
- Applied feature selection (SelectKBest with chi-squared on TF-IDF text features).
- Compared performance with and without feature selection.

---

## Additional Labs (ML Projects)

### Lab 6: Linear Regression
Description: Linear Regression model to predict 'Yearly Amount Spent' using the `Ecommerce Customers` dataset.
- **Dataset:** `Lab6/Ecommerce Customers`
- **Notebook:** `Lab6/lab6_faisal.ipynb`

### Lab 7: Logistic Regression
Description: Logistic Regression classification to predict whether a user 'Clicked on Ad' based on user features.
- **Dataset:** `Lab7/advertising.csv`
- **Notebook:** `Lab7/lab_7.ipynb`

### Lab 8: K-Nearest Neighbors (KNN)
Description: K-Nearest Neighbors classification using standardized features on an artificial dataset.
- **Dataset:** `Lab8/KNN_Project_Data`
- **Notebook:** `Lab8/02_K_Nearest_Neighbors_Assignment.ipynb`

### Lab 9: Decision Trees and Random Forest
Description: Classification model using Decision Trees and Random Forest to predict loan repayment.
- **Dataset:** `Lab9/loan_data.csv` (LendingClub data)
- **Notebook:** `Lab9/02-Decision Trees and Random Forest Project.ipynb`

---

## How to Run

1. Open the notebook in Google Colab or Jupyter.
2. Make sure the dataset path matches your environment:
   - Colab: `/content/ai_human_detection_v1.csv` (or relative path if using the organized structure).
   - Local: update `DATA_PATH` in the notebook.
3. Run all cells from top to bottom.

---

## Libraries Used

- Python
- pandas, numpy
- matplotlib, seaborn
- scikit-learn
