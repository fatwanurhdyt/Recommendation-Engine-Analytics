# 🚀 Hybrid Recommendation Engine: Content-Based & Collaborative Filtering

**Author:** Fatwa Nurhidayat  
**Focus:** Exploratory Data Analysis (EDA), Machine Learning, Predictive Analytics, and Large-Scale Data Processing.

---

## 📊 Business Overview
In the digital streaming and content aggregator industry, user engagement heavily relies on personalized content discovery. With thousands of titles available, users often face choice overload, leading to decreased retention. 

This project aims to build a robust **Recommendation Engine** that translates large-scale user interaction data and item metadata into actionable, personalized suggestions. By combining content characteristics and user behavior patterns, this system enhances user experience, solves the *cold-start* problem, and drives data-driven content optimization.

### Strategic Impact
1. **Personalized User Experience:** Delivering highly relevant recommendations to increase user engagement and satisfaction.
2. **Efficient Content Discovery:** Reducing the time users spend searching through extensive catalogs.
3. **Advanced Data Utilization:** Transforming over 7.8 million raw interaction logs and metadata into predictive insights.

---

## 🎯 Project Goals & Solution Approach

### Problem Statement
1. How to effectively recommend new content to users, especially addressing the *cold-start* problem for new users?
2. How to process large-scale datasets (millions of rows) to find hidden patterns in user preferences?

### The Solutions
To provide a comprehensive solution, this project implements a **Hybrid Approach**:

1. **Content-Based Filtering (TF-IDF & FAISS):** Utilizes **TF-IDF (Term Frequency-Inverse Document Frequency)** to extract and weight item metadata (e.g., genres). To ensure highly efficient similarity searches across the vectorized data, **FAISS (Facebook AI Similarity Search)** is implemented using Euclidean (L2) distance. This approach effectively resolves the cold-start problem by recommending items based on content similarity rather than historical user interactions.

2. **Collaborative Filtering (Singular Value Decomposition - SVD):** Leverages matrix factorization to uncover latent features from a massive dataset of user ratings. By mapping both users and items into a shared latent space, the SVD model accurately predicts user ratings for unseen content, delivering deeply personalized recommendations based on global community behavior.

---

## 🔍 Exploratory Data Analysis (EDA) & Data Pipeline

### Dataset Information
The system processes a large-scale public dataset from MyAnimeList, consisting of:
- **Items:** 12,294 titles containing metadata (Genre, Type, Episodes).
- **Interactions:** **7,813,737** rating records from 73,516 unique users.

### Data Integrity & Preparation
To ensure high data integrity and optimize model performance, an extensive data cleaning pipeline was executed:
- **Handling Missing Values & Duplicates:** Purged incomplete metadata and duplicate interaction logs to maintain dataset reliability.
- **Outlier Management:** Identified and processed invalid rating inputs (e.g., `-1` values representing unwatched/unrated statuses) to prevent model bias.
- **Feature Engineering:** Extracted relevant text features using TF-IDF, limiting vocabulary to the top 100 most impactful terms to optimize memory and processing speed.

---

## 🤖 Modeling & Inference

### 1. Content-Based Filtering
- **Mechanism:** Transforms genre metadata into a TF-IDF matrix. Uses FAISS to index the vectors and perform blazing-fast nearest-neighbor searches.
- **Strengths:** Excellent for new items or users (cold-start), providing highly interpretable recommendations based on explicit content features.
- *Output Example:* Successfully retrieves Top-N similar items based on textual input instantly.
*(Note: Refer to `img/output_cbf.png` for output visualization)*

### 2. Collaborative Filtering (SVD)
- **Mechanism:** Builds a predictive model using the Surprise library's SVD algorithm. The dataset is split (80/20) for rigorous training and testing.
- **Strengths:** Captures complex, non-linear relationships and hidden user preferences, making it highly robust against sparse datasets.
- *Output Example:* Predicts explicit ratings for unrated items, sorting them to deliver personalized Top-N lists.
*(Note: Refer to `img/output_cf.png` for output visualization)*

---

## 📈 Evaluation Metrics & Business Value

### 1. Precision@K (Content-Based)
To measure the absolute relevance of the top recommendations, Precision@K was utilized:

$$
\text{Precision@K} = \frac{\text{Relevant Items in Top-K}}{K}
$$

**Result:** The model achieved a Precision score of **1.00 (100%)** on test queries, meaning every single item recommended in the Top-10 list was strictly relevant to the input's characteristics.

### 2. Root Mean Square Error / RMSE (Collaborative Filtering)
To evaluate the predictive accuracy of the SVD model on unseen data:

$$
\text{RMSE} = \sqrt{ \frac{1}{N} \sum_{i=1}^{N} (\hat{r}_i - r_i)^2 }
$$

**Result:** The model achieved an RMSE of **1.1335**. Given the rating scale of 1-10, an error margin of ~1.1 indicates a highly accurate predictive capability, ensuring the recommendations closely align with actual user sentiment.

---
*Developed by Fatwa Nurhidayat as part of an advanced Machine Learning implementation to demonstrate proficiency in handling large-scale datasets, data mining, and building predictive analytics solutions.*