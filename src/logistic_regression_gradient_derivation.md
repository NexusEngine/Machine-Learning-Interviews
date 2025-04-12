# Derivation of Binary Cross-Entropy Loss Gradient for Logistic Regression

This document outlines the step-by-step process to derive the gradient of the binary cross-entropy loss function with respect to a weight parameter `w_j` for logistic regression.

**1. Define the Components:**

*   **Hypothesis (Prediction):** `h(x) = σ(z)` where `σ` is the sigmoid function.
*   **Linear Combination:** `z = wᵀx + b = Σ(w_k * x_k) + b`
*   **Sigmoid Function:** `σ(z) = 1 / (1 + e⁻ᶻ)`
*   **Binary Cross-Entropy Loss (for one example):** `L(y, h(x)) = -[y * log(h(x)) + (1-y) * log(1-h(x))]`
    *   `y` is the true label (0 or 1).
    *   `h(x)` is the predicted probability P(y=1|x).
*   **Cost Function (average loss over m examples):** `J(w, b) = (1/m) * Σ L(yᵢ, h(xᵢ))`

**2. Calculate Derivative of Sigmoid Function:**

We need `∂h/∂z = σ'(z)`:
*   `σ'(z) = d/dz [ (1 + e⁻ᶻ)⁻¹ ]`
*   `σ'(z) = -1 * (1 + e⁻ᶻ)⁻² * (-e⁻ᶻ)` (using chain rule)
*   `σ'(z) = e⁻ᶻ / (1 + e⁻ᶻ)²`
*   `σ'(z) = (1 / (1 + e⁻ᶻ)) * (e⁻ᶻ / (1 + e⁻ᶻ))`
*   `σ'(z) = σ(z) * ((1 + e⁻ᶻ - 1) / (1 + e⁻ᶻ))`
*   `σ'(z) = σ(z) * (1 - 1 / (1 + e⁻ᶻ))`
*   **`σ'(z) = σ(z) * (1 - σ(z))`** or **`∂h/∂z = h(x) * (1 - h(x))`**

**3. Apply the Chain Rule to the Loss Function:**

We want to find the derivative of the loss `L` with respect to a specific weight `w_j`: `∂L/∂w_j`.
Using the chain rule:
`∂L/∂w_j = (∂L/∂h) * (∂h/∂z) * (∂z/∂w_j)`

**4. Calculate Each Part of the Chain Rule:**

*   **`∂L/∂h`**:
    *   `∂L/∂h = d/dh [-y*log(h) - (1-y)*log(1-h)]`
    *   `∂L/∂h = - [y/h - (1-y)/(1-h)]`
    *   `∂L/∂h = - [ (y*(1-h) - (1-y)*h) / (h*(1-h)) ]`
    *   `∂L/∂h = - [ (y - y*h - h + y*h) / (h*(1-h)) ]`
    *   `∂L/∂h = - [ (y - h) / (h*(1-h)) ]`
    *   **`∂L/∂h = (h - y) / (h * (1 - h))`**

*   **`∂h/∂z`**: (From step 2)
    *   **`∂h/∂z = h * (1 - h)`**

*   **`∂z/∂w_j`**:
    *   `z = w₀x₀ + w₁x₁ + ... + w_j x_j + ... + w_n x_n + b` (assuming x₀=1 for bias)
    *   **`∂z/∂w_j = x_j`** (The j-th feature value for the current example)

**5. Combine the Parts:**

*   `∂L/∂w_j = [(h - y) / (h * (1 - h))] * [h * (1 - h)] * [x_j]`
*   The `h * (1 - h)` terms cancel out.
*   **`∂L/∂w_j = (h(x) - y) * x_j`**

**6. Gradient of the Cost Function (Average over m examples):**

The gradient of the total cost `J` is the average of the gradients for each example:
*   `∂J/∂w_j = (1/m) * Σ [ (h(xᵢ) - yᵢ) * xᵢ,j ]`

Where:
*   `m` is the number of training examples.
*   `i` iterates through each training example.
*   `h(xᵢ)` is the prediction for the i-th example.
*   `yᵢ` is the true label for the i-th example.
*   `xᵢ,j` is the value of the j-th feature for the i-th example.

This final expression gives the gradient component for a single weight `w_j`, which is used in gradient descent algorithms to update the weights. The derivative with respect to the bias term `b` can be found similarly (where `x_j` would effectively be 1).
