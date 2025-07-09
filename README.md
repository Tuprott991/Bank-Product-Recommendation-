# I. Overview

This project aims to build a series of recommendation systems to provide customized product offerings to customers in the banking and financial sector. Leveraging several proven algorithms, the project implements three traditional yet powerful techniques:

1. **Memory-based Collaborative Filtering:** User-based and Item-based models
2. **Model-based Collaborative Filtering:** Matrix Factorization (Funk SVD)
3. **Machine Learning:** GBDT + LR (Gradient Boosting Decision Tree + Logistic Regression)

For the *cold start* phase, where new customers have just opened their accounts and have no transaction or product purchase history, the project uses a *popularity-based* model to recommend the most popular products.

The project also evaluates the performance of these models and discusses use cases, risks, and production procedures for deploying a recommendation system in real-world scenarios.

# II. Project Steps

1. **Data Import and Profiling:** Import all required data and perform an initial quality check.
2. **Data Cleaning:** Impute missing values, drop unused fields, and enhance data integrity.
3. **Feature Engineering:** Transform original features into new representations for model training.
4. **Model Training:** Build and train the models listed above using the prepared data.
5. **Recommendation:** Recommend products to customers based on their purchasing history and features. Evaluate model performance using a test split.
6. **Implementation & Improvement:** Discuss real-life applications, potential risks, and ideas for further improvement.

# III. Propose models in project
## 1. Popularity-based Recommendation

- **Principle:** Recommend products based on the highest purchase frequency (item frequency) across the entire dataset.
- **Advantages:**  
  - Simple, easy to implement, does not require complex historical data.
  - Suitable for all users, especially new users (cold-start), helping them discover the most popular products at the current time.
- **Disadvantages:**  
  - Not personalized.
  - Does not consider time factors, which may lead to bias if a product is only popular during a specific period.
- **Solution:**  
  - Apply *time decay adjustment*: each interaction with a product is weighted less as it gets older, so recent interactions have higher weights. This helps the model reflect current trends rather than just total historical purchases.
- **Real-world application:**  
  - Commonly used to initialize recommendation systems or as a baseline for comparison with more complex models.

---

## 2. Memory-based Collaborative Filtering

### 2.1. User-based Collaborative Filtering

- **Principle:** Personalized recommendations based on user similarity. The idea is that users with similar preferences in the past will likely have similar preferences in the future.
- **How it works:**  
  - Build a user-item matrix (M x N), where M is the number of users and N is the number of products.
  - Calculate similarity between users using techniques like cosine similarity, Pearson correlation, or Jaccard coefficient.
  - Identify users similar to the target user, then recommend products they have purchased that the target user has not.
- **Advantages:**  
  - Good personalization, leverages community behavior.
  - Effective with a large user base and rich interaction data.
- **Disadvantages:**  
  - The user-item matrix is often sparse, with many zeros since most users have not interacted with many products.
  - Prediction becomes difficult with sparse data, especially for new users or new products (cold-start).
- **Real-world application:**  
  - Suitable for systems with a large user base, diverse products, and rich interaction data.

### 2.2. Item-based Collaborative Filtering

- **Principle:** Recommendations are based on product similarity rather than user similarity.
- **How it works:**  
  - Calculate similarity between products based on user interaction history.
  - Recommend products similar to those the user has already purchased.
- **Advantages:**  
  - More reliable and personalized than user-based, as it relies on products similar to those already purchased.
  - Often works better with sparse data, since products usually receive more ratings than users give.
  - More widely used in industry due to scalability and efficiency.
- **Disadvantages:**  
  - Still faces the cold-start problem for new products with no interaction history.
- **Real-world application:**  
  - Widely used in e-commerce, banking, and entertainment recommendation systems.

---

## 3. Model-based Collaborative Filtering

- **Principle:** Uses machine learning algorithms to learn patterns from user-item interaction data, rather than relying solely on direct similarity calculations.
- **Advantages over memory-based approaches:**
  1. **Addresses data sparsity:**  
     - Uses techniques like matrix factorization, dimensionality reduction, and latent factor models to uncover hidden factors influencing purchasing behavior.
     - Helps overcome the issue of many zeros in the user-item matrix.
  2. **Good scalability:**  
     - Models can be trained offline with efficient optimization algorithms, enabling fast predictions in production.
  3. **Improved performance:**  
     - Easily integrates new features and combines with other algorithms to boost effectiveness.
- **Algorithms used:**  
  - **Matrix Factorization:**  
    - Decomposes the user-item matrix into two lower-dimensional matrices (user matrix & item matrix), representing relationships between users/items and latent factors.
    - The idea is that hidden factors (preferences, product features) influence user-item interactions.
    - Optimized using gradient descent or alternating least squares to minimize reconstruction error between the original matrix and the product of the two new matrices.
    - Reduces computational cost, improves generalization, and addresses data sparsity.
- **Real-world application:**  
  - Used in large-scale recommendation systems such as Netflix, Amazon, and major banks.

---

## 4. Gradient Boosting Tree + Logistic Regression (GBDT + LR)

- **Principle:** Combines the strengths of GBDT (powerful feature learning) and Logistic Regression (interpretability and flexibility).
- **How it works:**  
  - GBDT captures complex patterns and feature interactions, automatically generating non-linear features.
  - Logistic Regression provides model interpretability and efficiently handles high-dimensional data.
  - After training, the model predicts the probability that a customer will prefer each product based on customer profiles, product features, and user-item interactions.
  - Recommends products with the highest predicted probabilities for each customer.
- **Advantages:**  
  - High recommendation performance, leveraging the strengths of both models.
  - Easily scalable and results are interpretable.
- **Disadvantages:**  
  - Requires more computational resources than simpler models.
  - Needs careful hyperparameter tuning to avoid overfitting.
- **Real-world application:**  
  - Used in modern recommendation systems, especially when interpretability for end-users or managers is required.

---

# IV. Summary

- The project follows these steps: data import, cleaning, feature engineering, model training, evaluation, and improvement.
- Each step is crucial to ensure the quality and effectiveness of the recommendation system.
- The proposed models range from simple (popularity-based) to complex (model-based, hybrid), each with its own strengths and suitable for different stages of system development.
- To optimize performance, it is recommended to combine multiple methods, continuously update data, and improve models based on real-world feedback.
- Special attention should be paid to practical issues such as cold-start, data drift over time, and scalability when deploying in a banking environment.

