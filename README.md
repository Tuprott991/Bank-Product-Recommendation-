# 1. Overview

This project aims to build a series of recommendation systems to provide customized product offerings to customers in the banking and financial sector. Leveraging several proven algorithms, the project implements three traditional yet powerful techniques:

1. **Memory-based Collaborative Filtering:** User-based and Item-based models
2. **Model-based Collaborative Filtering:** Matrix Factorization (Funk SVD)
3. **Machine Learning:** GBDT + LR (Gradient Boosting Decision Tree + Logistic Regression)

For the *cold start* phase, where new customers have just opened their accounts and have no transaction or product purchase history, the project uses a *popularity-based* model to recommend the most popular products.

The project also evaluates the performance of these models and discusses use cases, risks, and production procedures for deploying a recommendation system in real-world scenarios.

# 2. Project Steps

1. **Data Import and Profiling:** Import all required data and perform an initial quality check.
2. **Data Cleaning:** Impute missing values, drop unused fields, and enhance data integrity.
3. **Feature Engineering:** Transform original features into new representations for model training.
4. **Model Training:** Build and train the models listed above using the prepared data.
5. **Recommendation:** Recommend products to customers based on their purchasing history and features. Evaluate model performance using a test split.
6. **Implementation & Improvement:** Discuss real-life applications, potential risks, and ideas for further improvement.