# Decision Trees

A decision tree is a supervised learning algorithm that creates a flowchart-like tree structure where each internal node represents a feature (or attribute), each branch represents a decision rule, and each leaf node represents an outcome (categorical or numerical value). It's a non-parametric method used for both classification and regression tasks.

## Major Technical Points

1. **Basic Structure**

   - **Root Node**: The topmost node that represents the entire dataset
   - **Decision Nodes**: Internal nodes that test a feature and split the data
   - **Branches**: Connections between nodes representing the decision rules
   - **Leaf Nodes**: Terminal nodes that provide the final prediction value
   - **Paths**: Sequence of nodes from root to leaf representing decision rules

2. **Splitting Criteria**

   - **Classification Trees**:
     - **Gini Impurity**: Measures the probability of incorrect classification
       - Gini = 1 - Σ(p_i)², where p_i is the probability of class i
       - Lower Gini value indicates better splits
     - **Entropy/Information Gain**: Measures the reduction in uncertainty
       - Entropy = -Σ(p_i * log₂(p_i))
       - Information Gain = Entropy(parent) - Weighted Sum of Entropy(children)
   - **Regression Trees**:
     - **Variance Reduction**: Minimizes the variance in target values
     - **Mean Squared Error (MSE)**: Sum of squared differences between actual and predicted values

3. **Tree Construction Algorithm**

   - **Greedy Top-Down Approach (CART, ID3, C4.5, etc.)**:
     1. Select the best feature to split the data using splitting criteria
     2. Create child nodes based on the split
     3. Recursively repeat for each child node until stopping criteria are met
   - **Optimal Split Finding**: For each feature, evaluate all possible split points
   - **Computational Complexity**: O(n_features × n_samples × log(n_samples))

4. **Stop Criteria (Pruning Parameters)**

   - **Pre-pruning (Early stopping)**:
     - Maximum tree depth
     - Minimum samples required to split a node
     - Minimum samples required at a leaf node
     - Maximum number of leaf nodes
     - Minimum impurity decrease required for a split
   - **Post-pruning**:
     - Cost-complexity pruning (Minimal Cost-Complexity Pruning)
     - Reduced Error Pruning
     - Pessimistic Error Pruning

5. **Inference**

   - **Classification**: 
     - Traverse from root to leaf following decision rules
     - Assign majority class of the leaf node (or class probabilities)
   - **Regression**:
     - Traverse from root to leaf following decision rules
     - Assign average target value of the leaf node
   - **Time Complexity**: O(depth of tree) - very fast prediction

6. **Advantages**

   - Highly interpretable ("white box" model)
   - Requires minimal data preprocessing (no normalization/scaling needed)
   - Handles both numerical and categorical features
   - Implicitly performs feature selection
   - Handles non-linear relationships well
   - Robust to outliers

7. **Limitations**

   - Prone to overfitting (especially with deep trees)
   - High variance (small changes in data can result in very different trees)
   - Biased toward features with more levels (cardinality bias)
   - Struggles with imbalanced datasets
   - Cannot extrapolate beyond training data range

8. **Hyperparameters**

   - Maximum depth
   - Minimum samples to split
   - Minimum samples per leaf
   - Maximum features
   - Splitting criterion
   - Minimum impurity decrease

9. **Feature Importance**

   - Based on total reduction in criterion brought by that feature
   - Higher importance for features used closer to the root
   - Calculated as the weighted average of node impurity decreases across all nodes that use the feature

10. **Implementations**

    - **Scikit-learn**: `DecisionTreeClassifier`, `DecisionTreeRegressor`
    - **Computational Complexity**:
      - Training: O(n_features × n_samples × log(n_samples))
      - Prediction: O(log(n_samples))

11. **Extensions and Variants**

    - **Classification And Regression Trees (CART)**: Binary splits, uses Gini impurity or MSE
    - **ID3**: Uses entropy, handles categorical variables with multi-way splits
    - **C4.5**: Extension of ID3 that handles continuous attributes and missing values
    - **CHAID**: Uses chi-square tests for splitting, allows multi-way splits
    - **Conditional Inference Trees**: Uses statistical tests to select variables
