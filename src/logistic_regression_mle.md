# Maximum Likelihood Estimation in Logistic Regression

Maximum likelihood estimation (MLE) provides a statistical framework that explains why the binary cross-entropy loss function is the appropriate choice for logistic regression.

## The Statistical Perspective

In logistic regression, we're modeling a binary outcome (0 or 1) based on input features. From a statistical perspective, we're trying to estimate parameters that make our observed data most probable.

## Step-by-Step Derivation

### 1. The Probability Model

For logistic regression, we model the probability of a positive class (y=1) given features x as:

```
P(y=1|x) = σ(w^T x + b)
```

And consequently:

```
P(y=0|x) = 1 - σ(w^T x + b)
```

Where σ is the sigmoid function.

### 2. The Likelihood Function

The likelihood function represents the probability of observing our entire training dataset, given our model parameters. For a dataset with m independent examples, the likelihood is:

```
L(w,b) = P(y^(1), y^(2), ..., y^(m) | x^(1), x^(2), ..., x^(m), w, b)
```

Since the examples are independent, this becomes:

```
L(w,b) = ∏_{i=1}^m P(y^(i) | x^(i), w, b)
```

For each example, we need either P(y=1|x) or P(y=0|x) depending on the actual label. This can be elegantly written as:

```
L(w,b) = ∏_{i=1}^m [P(y=1|x^(i))^{y^(i)} × P(y=0|x^(i))^{(1-y^(i))}]
```

Substituting our model:

```
L(w,b) = ∏_{i=1}^m [σ(w^T x^(i) + b)^{y^(i)} × (1-σ(w^T x^(i) + b))^{(1-y^(i))}]
```

### 3. The Log-Likelihood

Taking the logarithm (which is monotonic and simplifies calculations):

```
log L(w,b) = ∑_{i=1}^m [y^(i) log(σ(w^T x^(i) + b)) + (1-y^(i)) log(1-σ(w^T x^(i) + b))]
```

### 4. Maximizing Log-Likelihood / Minimizing Negative Log-Likelihood

In maximum likelihood estimation, we want to find parameters w and b that maximize this log-likelihood. In optimization contexts, we typically phrase problems as minimizations, so we negate:

```
-log L(w,b) = -∑_{i=1}^m [y^(i) log(σ(w^T x^(i) + b)) + (1-y^(i)) log(1-σ(w^T x^(i) + b))]
```

Dividing by m to get the average:

```
-log L(w,b)/m = -(1/m)∑_{i=1}^m [y^(i) log(σ(w^T x^(i) + b)) + (1-y^(i)) log(1-σ(w^T x^(i) + b))]
```

This is precisely the binary cross-entropy loss function:

```
J(w,b) = -(1/m)∑_{i=1}^m [y^(i) log(p^(i)) + (1-y^(i)) log(1-p^(i))]
```

where p^(i) = σ(w^T x^(i) + b) is our predicted probability.

## Why This Matters

1. **Statistical Foundation**: MLE provides a principled statistical foundation for logistic regression.

2. **Probability Interpretation**: It ensures that outputs can be properly interpreted as probabilities.

3. **Appropriate Loss Function**: It explains why binary cross-entropy is the correct loss function for logistic regression rather than alternatives like mean squared error.

4. **Generalizes to Other Models**: This same MLE approach generalizes to other models, explaining the connection between logistic regression and other generalized linear models.

5. **Theoretical Guarantees**: MLE estimators have desirable statistical properties such as consistency and asymptotic normality under certain conditions.

By understanding the maximum likelihood perspective, you gain insight into why logistic regression works the way it does, beyond just the mechanics of the algorithm.
