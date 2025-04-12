# Logistic Regression

Logistic regression is a supervised learning algorithm used for binary classification problems. Despite its name, it's a classification algorithm rather than a regression algorithm. It models the probability that an input belongs to a certain class using the logistic (sigmoid) function to transform a linear combination of features into a value between 0 and 1.

## Major Technical Points

1. **Mathematical Formulation**

   - Probability model: P(y=1|x) = σ(w^T x + b)
   - Sigmoid function: σ(z) = 1/(1+e^(-z))
   - Decision boundary: w^T x + b = 0 (linear in feature space)
   - Log-odds (logit): log(P(y=1|x)/(1-P(y=1|x))) = w^T x + b

2. **Cost Function**

   - Binary cross-entropy loss: J(w,b) = -1/m Σ[y_i log(p_i) + (1-y_i) log(1-p_i)]
   - Cannot use MSE (non-convex for logistic regression)
   - Convex function with a global minimum
   - [Maximum likelihood estimation perspective](./logistic_regression_mle.md)

3. **Parameter Estimation**

   - No closed-form solution (unlike linear regression)
   - Gradient descent optimization
   - Newton's method/Fisher scoring
   - Maximum likelihood estimation

4. **Gradient Computation (detailed derivation)**

   - ∂J/∂w_j = 1/m Σ(h(x_i) - y_i)x_i,j
   - Similar to linear regression but with h(x) = σ(w^T x + b)
   - Efficient computation with vectorization

5. **Regularization**

   - L1 regularization (Lasso): Adds λΣ|w_j| to cost function
   - L2 regularization (Ridge): Adds λΣw_j² to cost function
   - Elastic Net: Combines L1 and L2
   - Prevents overfitting and feature selection (L1)

6. **Multi-class Extensions**

   - One-vs-Rest: Train K binary classifiers
   - Multinomial logistic regression (Softmax regression)
   - Softmax function: P(y=k|x) = e^(w_k^T x) / Σ e^(w_j^T x)

7. **Evaluation Metrics**

   - Accuracy, precision, recall, F1-score
   - ROC curve and AUC
   - Confusion matrix
   - Log loss (cross-entropy)

8. **Assumptions**

   - Independence of observations
   - No multicollinearity among predictors
   - Linear relationship between log-odds and features
   - Large sample size (rule of thumb: 10 events per predictor)

9. **Advantages**

   - Highly interpretable (coefficients represent log-odds)
   - Efficient training
   - Outputs well-calibrated probabilities
   - Works well with sparse data
   - Less prone to overfitting than complex models

10. **Limitations**

    - Cannot model complex non-linear decision boundaries
    - Assumes linearity in log-odds space
    - Sensitive to outliers
    - May struggle with imbalanced datasets
    - Complete separation issues ("perfect classification paradox")

11. **Implementation**

    - Scikit-learn: `LogisticRegression`
    - Statsmodels: Provides detailed statistical output
    - Gradient descent implementation for large datasets
    - Mini-batch and stochastic variants for scalability

12. **Practical Considerations**

    - Feature scaling important for convergence speed
    - Handling categorical variables with one-hot encoding
    - Feature selection to avoid multicollinearity
    - Threshold tuning for classification
