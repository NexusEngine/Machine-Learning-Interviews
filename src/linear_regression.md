# Linear Regression

Linear regression is a supervised learning algorithm that models the relationship between a dependent variable (target) and one or more independent variables (features) by fitting a linear equation to the observed data. The goal is to find the best-fitting straight line through the points that minimizes the sum of squared differences between observed and predicted values.

## Major Technical Points

1. **Mathematical Formulation**

   - Simple linear regression: y = β₀ + β₁x + ε
   - Multiple linear regression: y = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ + ε
   - Matrix form: y = Xβ + ε
2. **Parameter Estimation**

   - Ordinary Least Squares (OLS): β = (X^T X)^(-1) X^T y
   - Minimizes the sum of squared residuals: min Σ(y_i - ŷ_i)²
   - Closed-form solution exists when X^T X is invertible
3. **Assumptions**

   - Linearity: The relationship between X and y is linear
   - Independence: Observations are independent of each other
   - Homoscedasticity: Constant variance of errors
   - Normality: Errors are normally distributed
   - No multicollinearity: Independent variables are not highly correlated
4. **Evaluation Metrics**

   - R-squared (coefficient of determination): Proportion of variance explained
   - Adjusted R-squared: Accounts for the number of predictors
   - Mean Squared Error (MSE): Average of squared errors
   - Root Mean Squared Error (RMSE): Square root of MSE
   - Mean Absolute Error (MAE): Average of absolute errors
5. **Regularization Techniques**

   - Ridge Regression (L2): Adds penalty term λΣβ²
   - Lasso Regression (L1): Adds penalty term λΣ|β|
   - Elastic Net: Combines L1 and L2 penalties
6. **Extensions and Variants**

   - Polynomial Regression: Fits polynomial functions
   - Weighted Least Squares: Assigns different weights to observations
   - Generalized Linear Models: Extends to non-normal distributions
7. **Practical Considerations**

   - Feature scaling: Standardization or normalization
   - Handling categorical variables: One-hot encoding
   - Outlier detection and treatment
   - Feature selection methods
   - Cross-validation for model evaluation
8. **Limitations**

   - Sensitive to outliers
   - Cannot model complex non-linear relationships
   - Assumes independence of errors
   - Prone to overfitting with many features
9. **Implementation**

   - Scikit-learn: `LinearRegression`, `Ridge`, `Lasso`
   - Statsmodels: Provides detailed statistical output
   - Gradient descent implementation for large datasets
10. **Interpretability**

    - Coefficients represent change in y for unit change in x
    - Statistical significance tests for coefficients (t-tests, p-values)
    - Confidence intervals for parameter estimates
