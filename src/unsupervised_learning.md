# Unsupervised Learning

Unsupervised learning is a type of machine learning where algorithms identify patterns in data without labeled outputs or supervision. Unlike supervised learning, which trains on labeled examples, unsupervised learning discovers hidden structures and relationships in unlabeled data.

## Clustering

Clustering algorithms group similar data points together based on their intrinsic properties without prior knowledge of group definitions.

### K-means Clustering

**Technical Points:**
1. **Algorithm**:
   - Initialize k cluster centroids randomly
   - Iterate until convergence:
     - Assign each data point to the nearest centroid
     - Update centroids to the mean of all points in the cluster
   - Minimize the within-cluster sum of squares (WCSS)

2. **Mathematical Formulation**:
   - Objective: minimize J = Σ(Σ||x_i - μ_j||²) over all clusters j and points i in cluster j
   - Where μ_j is the centroid of cluster j

3. **Key Properties**:
   - Requires specifying the number of clusters (k) beforehand
   - Sensitive to initial centroid positions
   - Assumes spherical clusters of similar size
   - Complexity: O(n*k*d*i) where n=samples, k=clusters, d=dimensions, i=iterations

4. **Techniques for Choosing k**:
   - Elbow method: Plot WCSS vs. k and look for the "elbow"
   - Silhouette score: Measure of how similar objects are to their clusters compared to other clusters
   - Gap statistic: Compare intra-cluster dispersion to that expected under null reference

5. **Variants**:
   - K-means++: Smarter initialization to improve convergence
   - Mini-batch K-means: Uses batches of data for faster processing
   - K-medoids: Uses actual data points as centers (more robust to outliers)

### Hierarchical Clustering

**Technical Points:**
1. **Approaches**:
   - Agglomerative (bottom-up): Start with individual points as clusters and merge
   - Divisive (top-down): Start with one cluster and recursively divide

2. **Linkage Criteria**:
   - Single linkage: Minimum distance between points in clusters
   - Complete linkage: Maximum distance between points in clusters
   - Average linkage: Average distance between all pairs of points
   - Ward's method: Minimizes variance increase after merging

3. **Dendrogram**:
   - Tree diagram showing the hierarchical relationship between clusters
   - Height represents the distance at which clusters merge
   - Cutting the dendrogram at different heights produces different clustering granularities

4. **Advantages**:
   - No need to specify number of clusters beforehand
   - Provides hierarchical relationships between clusters
   - Works well with arbitrary distance metrics

5. **Disadvantages**:
   - Computationally expensive O(n³) for naive implementations
   - Cannot scale to large datasets without approximation
   - No global objective function being optimized

### DBSCAN (Density-Based Spatial Clustering of Applications with Noise)

**Technical Points:**
1. **Algorithm Concept**:
   - Define points as core, border, or noise based on neighborhood density
   - Core points have at least minPts within distance ε
   - Clusters form around connected core points
   - Border points are within ε of core points but have fewer than minPts neighbors
   - Noise points are not within ε of any core point

2. **Parameters**:
   - ε (epsilon): Neighborhood radius
   - minPts: Minimum number of points required to form a dense region

3. **Advantages**:
   - Discovers clusters of arbitrary shape
   - Naturally handles noise and outliers
   - Doesn't require specifying number of clusters
   - Works well with spatial data

4. **Disadvantages**:
   - Sensitive to parameter selection
   - Struggles with clusters of varying densities
   - Computationally expensive for large datasets (without optimization)

5. **Extensions**:
   - OPTICS: Addresses varying density clusters
   - HDBSCAN: Hierarchical DBSCAN with adaptive density thresholds

## Gaussian Mixture Models (GMMs)

GMMs model data as a mixture of several Gaussian distributions, providing a probabilistic clustering approach.

**Technical Points:**
1. **Mathematical Model**:
   - Data points are generated from a mixture of k multivariate Gaussian distributions
   - p(x) = Σ(π_i * N(x|μ_i, Σ_i)) for i=1 to k
   - Where π_i are mixture weights (probabilities), μ_i are means, and Σ_i are covariance matrices

2. **Expectation-Maximization (EM) Algorithm**:
   - E-step: Calculate posterior probabilities of each point belonging to each cluster
   - M-step: Update parameters (weights, means, covariances) based on these probabilities
   - Iterate until convergence
   - Guarantees to increase log-likelihood at each step

3. **Advantages**:
   - Provides probabilistic cluster assignments (soft clustering)
   - Flexible cluster shapes through covariance matrices
   - Can model complex distributions
   - Handles uncertainty in clustering

4. **Disadvantages**:
   - Sensitive to initialization
   - May converge to local optima
   - Requires specifying number of components
   - Struggles with high-dimensional data (curse of dimensionality)

5. **Variants**:
   - Diagonal GMM: Restricts covariance matrices to be diagonal
   - Tied GMM: Shares covariance matrix across components
   - Regularized GMM: Adds regularization to covariance estimation

## Dimensionality Reduction

Dimensionality reduction techniques transform high-dimensional data into a lower-dimensional space while preserving important information.

### Principal Component Analysis (PCA)

**Technical Points:**
1. **Mathematical Foundation**:
   - Linear transformation that finds orthogonal axes (principal components) of maximum variance
   - Covariance matrix of data: C = (1/n) * X^T * X (after centering)
   - Eigendecomposition of covariance matrix: C = V * D * V^T
   - Principal components are eigenvectors of the covariance matrix

2. **Algorithm**:
   - Center the data by subtracting the mean
   - Compute covariance matrix
   - Compute eigenvectors and eigenvalues
   - Sort eigenvectors by decreasing eigenvalues
   - Project data onto top k eigenvectors

3. **Variance Explained**:
   - Each component's eigenvalue represents the variance captured
   - Cumulative explained variance ratio guides dimension selection
   - Scree plot: eigenvalues vs. component indices

4. **Advantages**:
   - Reduces dimensionality while maximizing variance retention
   - Removes correlation between features
   - Computationally efficient
   - Helps with visualization and the curse of dimensionality

5. **Limitations**:
   - Only captures linear relationships
   - Sensitive to scaling of features
   - May not preserve important discriminative information
   - Struggles with manifold data

### Linear Discriminant Analysis (LDA)

**Technical Points:**
1. **Key Difference from PCA**:
   - Supervised technique that uses class labels
   - Maximizes separability between known classes
   - Finds axes that maximize between-class variance relative to within-class variance

2. **Mathematical Formulation**:
   - Between-class scatter matrix: S_B = Σ(n_i * (μ_i - μ) * (μ_i - μ)^T)
   - Within-class scatter matrix: S_W = Σ(Σ(x_j - μ_i) * (x_j - μ_i)^T)
   - Objective: maximize J(W) = |W^T * S_B * W| / |W^T * S_W * W|
   - Solution: eigenvectors of S_W^(-1) * S_B

3. **Advantages**:
   - Considers class separability
   - Good for multi-class problems
   - Can be used for both dimensionality reduction and classification
   - Often needs fewer components than PCA for classification tasks

4. **Limitations**:
   - Assumes classes have equal covariance matrices
   - Limited to c-1 dimensions (where c is number of classes)
   - Requires sufficient samples per class
   - Only captures linear decision boundaries

### t-SNE (t-Distributed Stochastic Neighbor Embedding)

**Technical Points:**
1. **Core Concept**:
   - Non-linear dimensionality reduction for visualization
   - Preserves local similarities and cluster structures
   - Converts high-dimensional distances to conditional probabilities
   - Creates similar distribution in low-dimensional space

2. **Algorithm**:
   - Compute pairwise similarities in high-dimensional space using Gaussian kernel
   - Define similar distribution in low-dimensional space using t-distribution
   - Minimize Kullback-Leibler divergence between distributions
   - Gradient descent with momentum for optimization

3. **Parameters**:
   - Perplexity: Balance between local and global structure (typically 5-50)
   - Learning rate: Step size for gradient descent
   - Early exaggeration: Initial multiplication factor for similarities

4. **Advantages**:
   - Excellent for visualizing high-dimensional data in 2D/3D
   - Preserves local structure and clusters
   - Handles non-linear manifolds well
   - More intuitive visualizations than linear methods

5. **Limitations**:
   - Computationally expensive O(n²)
   - Stochastic results (different runs give different results)
   - Not suitable for dimensionality reduction as preprocessing
   - Difficult to project new data points

## Association Rule Learning

Association rule learning identifies interesting relationships or associations between variables in large datasets.

### Apriori Algorithm

**Technical Points:**
1. **Key Concepts**:
   - Support: Frequency of itemset appearance in database
   - Confidence: Conditional probability of Y given X
   - Lift: Ratio of observed support to expected support if X and Y were independent

2. **Algorithm Steps**:
   - Generate frequent itemsets with minimum support
   - Generate rules with minimum confidence
   - Prune using anti-monotonicity: if an itemset is infrequent, its supersets must be infrequent

3. **Mathematical Measures**:
   - Support(X→Y) = P(X∩Y)
   - Confidence(X→Y) = P(Y|X) = Support(X∩Y)/Support(X)
   - Lift(X→Y) = Support(X∩Y)/(Support(X)*Support(Y))

4. **Applications**:
   - Market basket analysis
   - Product recommendations
   - Cross-selling strategies
   - Website navigation analysis

5. **Limitations**:
   - Computationally expensive for large datasets
   - Many rules may be generated (filtering required)
   - Only captures co-occurrence, not causality

## Anomaly Detection

Anomaly detection identifies data points, events, or observations that deviate significantly from the dataset's normal behavior.

**Technical Points:**
1. **Approaches**:
   - Statistical: Assumes normal distribution, flags outliers (z-score, modified z-score)
   - Distance-based: Measures distance to nearest neighbors (k-NN, LOF)
   - Density-based: Identifies points in low-density regions
   - Clustering-based: Points not belonging to any cluster or in small clusters
   - Isolation Forest: Isolates anomalies through random partitioning

2. **Evaluation Metrics**:
   - Precision, recall, F1-score (if labeled data available)
   - AUC-ROC, AUC-PR
   - Average precision score

3. **One-class SVM**:
   - Learns a boundary around normal data
   - New points outside boundary classified as anomalies
   - Uses kernel tricks for non-linear boundaries

4. **Autoencoder-based Detection**:
   - Train autoencoder on normal data
   - Measure reconstruction error for new points
   - High error indicates potential anomaly

5. **Challenges**:
   - Class imbalance (anomalies are rare)
   - Definition of "normal" may evolve over time
   - Curse of dimensionality affects distance-based methods
   - Lack of labeled data for supervised approaches

## Self-Organizing Maps (SOMs)

SOMs are a type of artificial neural network for unsupervised learning that produces a low-dimensional representation of the input space.

**Technical Points:**
1. **Architecture**:
   - Competitive learning neural network
   - Grid of neurons, each with a weight vector of the same dimension as input
   - Topological preservation of input space

2. **Training Algorithm**:
   - Initialize weights randomly
   - For each input vector:
     - Find Best Matching Unit (BMU) with closest weights
     - Update BMU and neighbors to move closer to input
     - Neighborhood size and learning rate decrease over time

3. **Applications**:
   - Visualization of high-dimensional data
   - Document organization
   - Customer segmentation
   - Feature extraction

4. **Advantages**:
   - Preserves topological structure
   - Handles high-dimensional data well
   - Robust to noise
   - Visual interpretation

5. **Limitations**:
   - Fixed network structure must be defined in advance
   - Sensitive to initial conditions
   - Training can be computationally intensive
   - Discrete representation of continuous spaces

## Evaluation of Unsupervised Learning

Without labeled data, evaluating unsupervised learning models requires specialized metrics.

**Technical Points:**
1. **Internal Validation Metrics**:
   - Silhouette coefficient: Measures separation and cohesion
   - Davies-Bouldin index: Ratio of within-cluster to between-cluster distances
   - Calinski-Harabasz index: Ratio of between-cluster to within-cluster dispersion
   - Dunn index: Ratio of minimum inter-cluster to maximum intra-cluster distance

2. **External Validation Metrics** (when labels available):
   - Adjusted Rand Index (ARI)
   - Normalized Mutual Information (NMI)
   - Fowlkes-Mallows score
   - V-measure

3. **Stability Assessment**:
   - Bootstrap sampling to measure clustering stability
   - Consensus clustering
   - Cross-validation for dimensionality reduction

4. **Visual Inspection**:
   - Dimensionality reduction for visualization
   - Cluster visualization
   - Heatmaps of distance/similarity matrices

5. **Domain-specific Evaluation**:
   - Business metric improvements
   - Expert validation
   - A/B testing of applications of unsupervised learning results
