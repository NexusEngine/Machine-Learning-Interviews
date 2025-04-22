# Ensemble Methods

Ensemble methods combine multiple machine learning models to produce better predictive performance than could be obtained from any of the constituent models alone. These methods work by reducing variance (bagging), bias (boosting), or both.

## Bagging and Boosting Methods

### Bagging (Bootstrap Aggregating)

Bagging is an ensemble technique that builds multiple models independently and then combines their predictions through averaging (for regression) or voting (for classification).

**Technical Points:**
1. **Bootstrap Sampling**: Each model is trained on a random subset of the training data, sampled with replacement
2. **Parallel Training**: Models are built independently and can be trained in parallel
3. **Averaging/Voting**: Final prediction is the average (regression) or majority vote (classification) of all models
4. **Variance Reduction**: Primarily reduces variance, making models more stable and robust to noise
5. **Out-of-Bag (OOB) Estimation**: Data points not used in bootstrap samples can estimate model performance without separate validation set
6. **Implementation**: Typically uses decision trees as base learners, but can use any algorithm
7. **Hyperparameters**: Number of estimators, sample size, maximum features

**Mathematical Formulation:**
- For regression: $f_{bag}(x) = \frac{1}{B} \sum_{i=1}^{B} f_i(x)$
- For classification: $f_{bag}(x) = \text{mode}(f_1(x), f_2(x), \ldots, f_B(x))$

Where $B$ is the number of bootstrap samples, and $f_i(x)$ is the prediction of the $i$-th model.

## Random Forest

Random Forest is a specific implementation of bagging that uses decision trees as base learners with an additional layer of randomness.

**Technical Points:**
1. **Feature Randomization**: At each split, only a random subset of features is considered
2. **De-correlation**: The feature randomization helps de-correlate the trees
3. **Bootstrapping**: Each tree is trained on a bootstrap sample of the training data
4. **No Pruning**: Trees are typically grown deep with minimal or no pruning
5. **Feature Importance**: Provides built-in feature importance metrics based on impurity decrease
6. **Computational Efficiency**: Highly parallelizable algorithm
7. **Hyperparameters**:
   - Number of trees (n_estimators)
   - Maximum depth of trees
   - Minimum samples for split/leaf
   - Maximum features to consider per split ($\sqrt{p}$ for classification, $p/3$ for regression where $p$ is total features)

**Algorithm:**
```
RANDOM_FOREST(training_data, n_trees, max_features, min_samples_split)
    forest = []
    
    for i = 1 to n_trees do
        # Create bootstrap sample of size N from training data
        bootstrap_sample = SAMPLE_WITH_REPLACEMENT(training_data, N)
        
        # Train a decision tree on the bootstrap sample
        tree = DECISION_TREE()
        
        # Recursively build the tree
        BUILD_TREE(tree, bootstrap_sample, max_features, min_samples_split)
        
        # Add the tree to our forest
        forest.append(tree)
    end for
    
    return forest

BUILD_TREE(tree, data, max_features, min_samples_split)
    if stopping_criteria_met(data, min_samples_split) then
        return LEAF_NODE(majority_class_or_average_value(data))
    end if
    
    # Consider only a random subset of features at this node
    feature_subset = RANDOM_SUBSET(all_features, max_features)
    
    # Find the best feature and split point among the subset
    best_feature, best_split = FIND_BEST_SPLIT(data, feature_subset)
    
    # Split the data
    left_data, right_data = SPLIT_DATA(data, best_feature, best_split)
    
    # Recursively build left and right subtrees
    left_subtree = BUILD_TREE(tree, left_data, max_features, min_samples_split)
    right_subtree = BUILD_TREE(tree, right_data, max_features, min_samples_split)
    
    return INTERNAL_NODE(best_feature, best_split, left_subtree, right_subtree)

PREDICT(forest, sample)
    predictions = []
    
    for each tree in forest do
        # Get prediction from each tree
        prediction = TREE_PREDICT(tree, sample)
        predictions.append(prediction)
    end for
    
    # For classification: return majority vote
    # For regression: return average
    if classification_task then
        return MODE(predictions)
    else
        return MEAN(predictions)
    end if
```

**Advantages:**
- Robust to overfitting compared to individual decision trees
- Handles high-dimensional data well
- Provides feature importance
- Works well "out of the box" with little tuning
- Handles missing values and maintains accuracy for missing data

**Limitations:**
- Less interpretable than single decision trees
- Can be computationally expensive with many trees and large datasets
- Not well-suited for linear relationships with a small number of features

## Boosting

Boosting builds models sequentially, where each model attempts to correct the errors made by the previous models.

### AdaBoost (Adaptive Boosting)

AdaBoost was one of the first successful boosting algorithms, which adjusts the weights of incorrectly classified instances to focus on difficult cases.

**Technical Points:**
1. **Sequential Training**: Models are built sequentially, each trying to correct errors made by previous models
2. **Instance Weighting**: After each model is trained, misclassified instances get higher weights
3. **Weighted Voting**: Final prediction is a weighted majority vote of all models
4. **Base Learners**: Typically uses decision stumps (one-level decision trees)
5. **Model Weighting**: Each model's contribution to the final prediction is weighted based on its accuracy
6. **Exponential Loss**: Optimizes exponential loss function
7. **Hyperparameters**: Learning rate, number of estimators

**Algorithm:**
```
ADABOOST(training_data, labels, n_estimators)
    # Initialize weights uniformly
    weights = [1/N, 1/N, ..., 1/N]  # N is the number of training examples
    ensemble = []
    alpha_values = []
    
    for m = 1 to n_estimators do
        # Train weak learner (usually a decision stump) on weighted data
        weak_learner = TRAIN_WEAK_LEARNER(training_data, labels, weights)
        
        # Get predictions from this weak learner
        predictions = PREDICT(weak_learner, training_data)
        
        # Calculate weighted error
        error = 0
        for i = 1 to N do
            if predictions[i] != labels[i] then
                error = error + weights[i]
            end if
        end for
        
        # If error is too high (≥ 0.5), discard this model and terminate
        if error >= 0.5 then
            break
        end if
        
        # Calculate model weight
        alpha = 0.5 * ln((1 - error) / error)
        
        # Update instance weights
        for i = 1 to N do
            if predictions[i] == labels[i] then
                weights[i] = weights[i] * exp(-alpha)
            else
                weights[i] = weights[i] * exp(alpha)
            end if
        end for
        
        # Normalize weights to sum to 1
        weights = NORMALIZE(weights)
        
        # Add model to ensemble
        ensemble.append(weak_learner)
        alpha_values.append(alpha)
    end for
    
    return ensemble, alpha_values

PREDICT_ADABOOST(ensemble, alpha_values, sample)
    final_score = 0
    
    for m = 1 to LENGTH(ensemble) do
        prediction = PREDICT(ensemble[m], sample)  # Usually +1 or -1
        final_score = final_score + (alpha_values[m] * prediction)
    end for
    
    # Return sign of the final score
    return SIGN(final_score)
```

**Mathematical Formulation:**
- Final prediction: $F(x) = \sum_{m=1}^{M} \alpha_m h_m(x)$
- Where $\alpha_m$ is the weight of the $m$-th model, and $h_m(x)$ is the prediction of the $m$-th model.
- $\alpha_m = \frac{1}{2} \ln\left(\frac{1-\epsilon_m}{\epsilon_m}\right)$, where $\epsilon_m$ is the weighted error rate of model $m$

### Gradient Boosting Machine (GBM)

GBM builds on the boosting framework but uses gradient descent optimization to minimize a loss function.

**Technical Points:**
1. **Gradient Descent**: Uses gradient descent to minimize a differentiable loss function
2. **Residual Fitting**: Each tree tries to predict the residuals (errors) of the previous trees
3. **Shrinkage**: Uses a learning rate to scale the contribution of each tree
4. **Loss Functions**: Can optimize various loss functions (MSE for regression, log loss for classification)
5. **Full Trees**: Usually uses deeper trees than AdaBoost
6. **Regularization**: Various regularization techniques to prevent overfitting
7. **Hyperparameters**:
   - Number of trees
   - Learning rate (shrinkage)
   - Tree depth
   - Subsampling rate

**Algorithm:**
```
GRADIENT_BOOSTING(training_data, labels, n_estimators, learning_rate, loss_function)
    # Initialize with a constant value (for regression: mean of target values)
    F_0(x) = arg min_γ Σ L(y_i, γ)
    ensemble = [F_0]
    
    for m = 1 to n_estimators do
        # Compute the negative gradient (pseudo-residuals)
        for i = 1 to N do
            r_im = -[∂L(y_i, F(x_i))/∂F(x_i)]_{F=F_{m-1}}
        end for
        
        # Fit a regression tree to the negative gradient values
        tree_m = FIT_REGRESSION_TREE(training_data, pseudo_residuals)
        
        # Find the optimal leaf node predictions by solving
        # γ_jm = arg min_γ Σ_{x_i∈R_jm} L(y_i, F_{m-1}(x_i) + γ)
        # for each leaf region R_jm in the tree
        OPTIMIZE_LEAF_VALUES(tree_m, training_data, labels, F_{m-1}, loss_function)
        
        # Update the model with a shrunken version of the new tree
        F_m(x) = F_{m-1}(x) + learning_rate * tree_m(x)
        
        # Add the tree to the ensemble
        ensemble.append(tree_m)
    end for
    
    return ensemble

PREDICT_GBM(ensemble, learning_rate, sample)
    # Start with the initial prediction
    prediction = ensemble[0](sample)
    
    # Add the contribution of each tree
    for m = 1 to LENGTH(ensemble) - 1 do
        prediction = prediction + learning_rate * ensemble[m](sample)
    end for
    
    return prediction
```

**Mathematical Formulation:**
- Initialize: $F_0(x) = \arg\min_\gamma \sum_{i=1}^{n} L(y_i, \gamma)$
- For $m = 1$ to $M$:
  - Compute negative gradient: $r_{im} = -\left[\frac{\partial L(y_i, F(x_i))}{\partial F(x_i)}\right]_{F=F_{m-1}}$
  - Fit a regression tree to the negative gradient: $h_m(x)$
  - Find multiplier: $\gamma_m = \arg\min_\gamma \sum_{i=1}^{n} L(y_i, F_{m-1}(x_i) + \gamma h_m(x_i))$
  - Update model: $F_m(x) = F_{m-1}(x) + \eta \gamma_m h_m(x)$ (where $\eta$ is the learning rate)

### XGBoost (eXtreme Gradient Boosting)

XGBoost is an optimized implementation of gradient boosting with several improvements for performance and regularization.

**Technical Points:**
1. **Regularization**: L1 and L2 regularization to prevent overfitting
2. **CART with Weighted Quantile Sketch**: More efficient handling of missing values
3. **Sparsity-Aware Split Finding**: Efficiently handles sparse data
4. **Parallelization**: Column block for parallel learning
5. **Cache Optimization**: Optimized for hardware caching
6. **Out-of-Core Computing**: Can handle datasets that don't fit in memory
7. **Built-in Cross-Validation**: Prevents overfitting with early stopping
8. **Hyperparameters**:
   - Number of trees
   - Learning rate
   - Maximum depth
   - Regularization alpha and lambda
   - Subsampling and column sampling rates
   - Gamma (minimum loss reduction for split)
   - Scale_pos_weight (for imbalanced datasets)

**Mathematical Extensions:**
- Objective function: $Obj = \sum_{i=1}^{n} L(y_i, \hat{y}_i) + \sum_{k=1}^{K} \Omega(f_k)$
- Regularization term: $\Omega(f) = \gamma T + \frac{1}{2}\lambda\|w\|^2$
- Second-order approximation for faster convergence

**Advantages:**
- Consistently outperforms standard GBM
- Better handling of missing values
- Built-in regularization
- Highly optimized for speed and performance
- Versatile (supports custom objectives and evaluation metrics)

## Comparative Analysis

| Aspect | Bagging (Random Forest) | Boosting (AdaBoost, GBM, XGBoost) |
|--------|-------------------------|-----------------------------------|
| Training Method | Parallel (independent) | Sequential (dependent) |
| Primary Benefit | Reduces variance | Reduces bias (and variance with regularization) |
| Prone to Overfitting | Less prone | More prone (without proper regularization) |
| Training Speed | Fast (parallelizable) | Slower (sequential) |
| Hyperparameter Sensitivity | Less sensitive | More sensitive |
| Performance on Noisy Data | Better | Can perform poorly with outliers |
| Interpretability | Moderate (feature importance) | Lower (complex interactions) |
| Best Use Case | When base models have high variance | When base models have high bias |

## Implementation Considerations

1. **Base Model Selection**: 
   - Decision trees are most common but any model can be used
   - Weak learners for boosting (shallow trees)
   - Stronger models for bagging

2. **Ensemble Size**: 
   - Random Forest: Often 100-500 trees
   - Boosting: Needs careful tuning (early stopping)

3. **Feature Importance**:
   - Both methods provide feature importance metrics
   - Random Forest: Mean decrease in impurity
   - Boosting: Accumulated improvement in loss function

4. **Computational Resources**:
   - Bagging: Parallelizable (multi-core processing)
   - Boosting: Sequential (less parallelizable)
   - XGBoost: Optimized for both parallel and distributed computing
