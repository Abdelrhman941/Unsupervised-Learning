# 🔍 Unsupervised Learning - Complete Reference Guide

A comprehensive resource for understanding, selecting, and applying unsupervised learning algorithms across clustering, dimensionality reduction, and anomaly detection tasks.

## 📋 Table of Contents

1. [Overview](#overview)
2. [Algorithm Selection Guide](#algorithm-selection-guide)
3. [Clustering Algorithms](#clustering-algorithms)
4. [Dimensionality Reduction](#dimensionality-reduction)
5. [Anomaly Detection](#anomaly-detection)
6. [Performance Comparison](#performance-comparison)
7. [Evaluation Metrics](#evaluation-metrics)
8. [Decision Framework](#decision-framework)
9. [Further Resources](#further-resources)

## 📚 Documentation
- [Complete Unsupervised Learning Guide](docs/unsupervised.html)

```
2-Unsupervised_Learning/
├── 1-Clustering Algorithms/
│   ├── 1-K_Means/
│   ├── 2-DBSCAN/
│   └── 3-Hierarchical Clustering/
├── 2-Dimensionality Reduction/
│   ├── 1-PCA/
│   ├── 2-t_SNE/
│   └── 3-Autoencoders/
├── 3-Anomaly Detection/
│   ├── 1-Isolation Forest/
│   ├── 2-One Class SVM/
│   └── 3-LOF/
└── docs/
    └── unsupervised.html
```

<div style="width: 100%; height: 10px; background: linear-gradient(to right, gray, white, orange,white, gray); border-radius: 5px; margin: 20px 0;"></div>

## Overview

Unsupervised learning algorithms discover hidden patterns, structures, and relationships in data without labeled examples. These techniques are essential for exploratory data analysis, feature engineering, and understanding complex datasets.

**Key Problem Types:**
- **Clustering**: Group similar data points
- **Dimensionality Reduction**: Compress high-dimensional data
- **Anomaly Detection**: Identify unusual patterns
- **Association Rules**: Find relationships between variables

**Key Considerations:**
- **Data Size**: Small $(< 1K)$, Medium $(1K-100K)$, Large $(> 100K)$
- **Dimensionality**: Low $(< 50)$, Medium $(50-1000)$, High $(> 1000)$
- **Cluster Shape**: Spherical, Arbitrary, Hierarchical
- **Noise Tolerance**: Clean vs Noisy data
- **Interpretability**: High vs Low requirement
- **Computational Resources**: Memory and time constraints

<div style="width: 100%; height: 10px; background: linear-gradient(to right, gray, white, orange,white, gray); border-radius: 5px; margin: 20px 0;"></div>

## Algorithm Selection Guide

### 🎯 Quick Decision Tree

```
Problem Type?
├── Clustering
│   ├── Know number of clusters? → K-Means
│   ├── Arbitrary shapes + noise? → DBSCAN
│   └── Hierarchical structure? → Hierarchical Clustering
├── Dimensionality Reduction
│   ├── Linear relationships? → PCA
│   ├── Non-linear + visualization? → t-SNE
│   └── Complex patterns? → Autoencoders
└── Anomaly Detection
    ├── Tree-based isolation? → Isolation Forest
    ├── Boundary-based? → One-Class SVM
    └── Density-based? → LOF
```

### 📋 Detailed Algorithm Workflows

#### K-Means - Step by Step:
1. **Initialize** k centroids randomly
2. **Assign** each point to nearest centroid  
3. **Update** centroids to cluster means
4. **Repeat** until convergence

#### DBSCAN - Core Concepts:
1. **Define density** as points within epsilon distance
2. **Core points** have minimum neighbors (minPts)
3. **Border points** are within epsilon of core points
4. **Noise points** are neither core nor border

#### Hierarchical Clustering (Agglomerative):
1. **Start** with each point as individual cluster
2. **Merge** closest clusters using linkage criteria
3. **Continue** until single cluster remains
4. **Cut** dendrogram at desired level

#### PCA Process:
1. **Standardize** data (center and scale)
2. **Compute** covariance matrix
3. **Find** eigenvectors and eigenvalues
4. **Sort** by eigenvalues (importance)
5. **Project** data onto top k components

#### t-SNE Workflow:
1. **Compute** pairwise similarities in high-D space
2. **Initialize** random low-D embedding
3. **Compute** similarities in low-D space
4. **Minimize** divergence between distributions
5. **Use** gradient descent for optimization

#### Autoencoders Architecture:
1. **Encoder** compresses input to latent space
2. **Decoder** reconstructs from latent representation
3. **Train** to minimize reconstruction error
4. **Use** encoder for dimensionality reduction

#### Isolation Forest Method:
1. **Build** ensemble of isolation trees
2. **Randomly** select features and split values
3. **Partition** data recursively
4. **Anomalies** have shorter average path lengths
5. **Score** based on path length normalization

#### One-Class SVM Approach:
1. **Map** data to high-dimensional space (kernel)
2. **Find** hyperplane separating data from origin
3. **Maximize** margin around normal data
4. **Points** outside boundary are anomalies

#### LOF Process:
1. **Compute** k-distance for each point
2. **Calculate** reachability distance
3. **Compute** local reachability density
4. **Compare** density to neighbors
5. **Generate** LOF score

### 📊 Data Characteristics Matrix

| Data Condition | Clustering | Dimensionality Reduction | Anomaly Detection |
|---|---|---|---|
| **Small dataset (< 1K)** | K-Means, Hierarchical | PCA, t-SNE | LOF, One-Class SVM |
| **Large dataset (> 100K)** | K-Means, DBSCAN | PCA, Autoencoders | Isolation Forest |
| **High-dimensional** | K-Means (with PCA) | PCA, Autoencoders | Isolation Forest |
| **Non-spherical clusters** | DBSCAN, Hierarchical | t-SNE, Autoencoders | LOF |
| **Noisy data** | DBSCAN | PCA (robust variants) | Isolation Forest, LOF |
| **Need interpretability** | K-Means, Hierarchical | PCA | Isolation Forest |
| **Real-time processing** | K-Means | PCA | Isolation Forest |

<div style="width: 100%; height: 10px; background: linear-gradient(to right, gray, white, orange,white, gray); border-radius: 5px; margin: 20px 0;"></div>

## Clustering Algorithms

Clustering algorithms group similar data points together without prior knowledge of group labels.

<div style="background: linear-gradient(135deg,rgb(139, 162, 80),rgb(50, 165, 136)); 
            color: #ffffff; 
            width: 100%; 
            height: 40px; 
            text-align: center; 
            font-weight: bold; 
            line-height: 40px; 
            margin: 15px 0; 
            font-size: 24px; 
            border-radius: 10px; 
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3);">
    1. K-Means Clustering
</div>

**Core Algorithm:**
1. Initialize k centroids randomly
2. Assign points to nearest centroid
3. Update centroids to cluster means
4. Repeat until convergence

**Objective Function:**
$$
J = \sum_{i=1}^{k} \sum_{x \in C_i} ||x - \mu_i||^2
$$

**Distance Metric:**
$$
d(x, \mu) = \sqrt{\sum_{j=1}^{n} (x_j - \mu_j)^2}
$$

**When to Use:**
- Spherical, well-separated clusters expected
- Know approximate number of clusters
- Fast clustering needed for large datasets
- Feature scaling is possible
- Similar cluster sizes
- Need fast, scalable algorithm

**When NOT to Use:**
- Non-spherical or irregular cluster shapes
- Clusters of very different sizes/densities
- Presence of outliers (sensitive to them)
- Unknown optimal k value
- Vastly different cluster sizes
- Noisy data with outliers

**Pros:**
- Fast and computationally efficient
- Simple to understand and implement
- Works well with spherical clusters
- Scales well to large datasets

**Cons:**
- Requires pre-specifying k
- Sensitive to initialization and outliers
- Assumes spherical clusters
- Struggles with varying cluster sizes

**Variants:**
- **K-Means++**: Smart initialization
- **Mini-batch K-Means**: Faster for large datasets
- **K-Medoids**: More robust to outliers

<div style="background: linear-gradient(135deg,rgb(139, 162, 80),rgb(50, 165, 136)); 
            color: #ffffff; 
            width: 100%; 
            height: 40px; 
            text-align: center; 
            font-weight: bold; 
            line-height: 40px; 
            margin: 15px 0; 
            font-size: 24px; 
            border-radius: 10px; 
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3);">
    2. DBSCAN (Density-Based Spatial Clustering)
</div>

**Core Concepts:**
- **Core Point**: Has ≥ MinPts neighbors within ε distance
- **Border Point**: Within ε of a core point
- **Noise Point**: Neither core nor border

**Parameters:**
- **ε (eps)**: Maximum distance between points
- **MinPts**: Minimum points to form dense region

**Algorithm:**
1. For each unvisited point
2. Find all neighbor points within ε
3. If neighbors ≥ MinPts, start new cluster
4. Recursively add density-reachable points

**When to Use:**
- Irregular cluster shapes expected
- Outlier detection needed
- Unknown number of clusters
- Varying cluster densities
- Arbitrary cluster shapes
- Noisy data with outliers

**When NOT to Use:**
- Clusters with significantly different densities
- High-dimensional data (curse of dimensionality)
- Need deterministic results (order-dependent)
- Very high-dimensional data
- Varying densities across clusters
- Need deterministic results
- Memory constraints (stores all distances)

**Pros:**
- Finds arbitrary cluster shapes
- Automatically determines cluster count
- Robust to outliers
- Identifies noise points

**Cons:**
- Sensitive to hyperparameters (eps, min_samples)
- Struggles with varying densities
- Memory intensive for large datasets
- Difficult parameter tuning

<div style="background: linear-gradient(135deg,rgb(139, 162, 80),rgb(50, 165, 136)); 
            color: #ffffff; 
            width: 100%; 
            height: 40px; 
            text-align: center; 
            font-weight: bold; 
            line-height: 40px; 
            margin: 15px 0; 
            font-size: 24px; 
            border-radius: 10px; 
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3);">
    3. Hierarchical Clustering
</div>

**Two Approaches:**
- **Agglomerative**: Bottom-up (merge clusters)
- **Divisive**: Top-down (split clusters)

**Linkage Criteria:**
- **Single**: $d(A,B) = \min_{a \in A, b \in B} d(a,b)$
- **Complete**: $d(A,B) = \max_{a \in A, b \in B} d(a,b)$
- **Average**: $d(A,B) = \frac{1}{|A||B|} \sum_{a \in A, b \in B} d(a,b)$
- **Ward**: Minimizes within-cluster variance

**When to Use:**
- Need cluster hierarchy/relationships
- Small to medium datasets
- Exploring different cluster granularities
- Deterministic results required
- Need hierarchical structure
- Unknown number of clusters
- Want dendrogram visualization

**When NOT to Use:**
- Very large datasets (O(n³) complexity)
- Need fast clustering
- Clear cluster count known beforehand
- Large datasets (O(n³) complexity)
- Need specific number of clusters
- High-dimensional data
- Real-time applications

**Pros:**
- No need to specify cluster count
- Deterministic results
- Provides cluster hierarchy
- Works with any distance metric

**Cons:**
- Computationally expensive O(n³)
- Sensitive to outliers
- Difficult to handle large datasets
- Merge decisions are irreversible

<div style="width: 100%; height: 10px; background: linear-gradient(to right, gray, white, orange,white, gray); border-radius: 5px; margin: 20px 0;"></div>

## Dimensionality Reduction

Techniques to reduce the number of features while preserving important information.

<div style="background: linear-gradient(135deg,rgb(139, 162, 80),rgb(50, 165, 136)); 
            color: #ffffff; 
            width: 100%; 
            height: 40px; 
            text-align: center; 
            font-weight: bold; 
            line-height: 40px; 
            margin: 15px 0; 
            font-size: 24px; 
            border-radius: 10px; 
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3);">
    1. Principal Component Analysis (PCA)
</div>

**Mathematical Foundation:**
- Find directions of maximum variance
- Project data onto principal components
- Components are orthogonal and ranked by variance

**Eigenvalue Decomposition:**
$$
\mathbf{C} = \mathbf{X}^T\mathbf{X} = \mathbf{V}\mathbf{\Lambda}\mathbf{V}^T
$$

**Variance Explained:**
$$
\text{Explained Variance Ratio} = \frac{\lambda_i}{\sum_{j=1}^{n} \lambda_j}
$$

**When to Use:**
- Linear relationships in data
- Need interpretable components
- Computational efficiency important
- Preprocessing for other algorithms
- Feature engineering and noise reduction

**When NOT to Use:**
- Non-linear relationships dominant
- Need to preserve local structure
- Categorical or discrete data
- Non-linear relationships
- Need exact feature reconstruction
- Very sparse data
- Components must have domain meaning

**Pros:**
- Fast and efficient
- Interpretable components
- Removes multicollinearity
- Good for preprocessing

**Cons:**
- Only captures linear relationships
- Components may lack interpretability
- Sensitive to feature scaling
- May lose important non-linear patterns

**Variants:**
- **Kernel PCA**: Non-linear version
- **Sparse PCA**: Interpretable components
- **Incremental PCA**: Large datasets

<div style="background: linear-gradient(135deg,rgb(139, 162, 80),rgb(50, 165, 136)); 
            color: #ffffff; 
            width: 100%; 
            height: 40px; 
            text-align: center; 
            font-weight: bold; 
            line-height: 40px; 
            margin: 15px 0; 
            font-size: 24px; 
            border-radius: 10px; 
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3);">
    2. t-SNE (t-Distributed Stochastic Neighbor Embedding)
</div>

**Core Idea:**
- Preserve local neighborhoods in low dimensions
- Convert distances to probabilities
- Minimize KL divergence between distributions

**Probability in High Dimensions:**
$$
p_{j|i} = \frac{\exp(-||x_i - x_j||^2 / 2\sigma_i^2)}{\sum_{k \neq i} \exp(-||x_i - x_k||^2 / 2\sigma_i^2)}
$$

**Probability in Low Dimensions:**
$$
q_{ij} = \frac{(1 + ||y_i - y_j||^2)^{-1}}{\sum_{k \neq l}(1 + ||y_k - y_l||^2)^{-1}}
$$

**When to Use:**
- Data visualization (2D/3D)
- Exploring cluster structure
- Non-linear relationships exist
- Local structure preservation needed
- Non-linear dimensionality reduction
- Understanding data manifolds

**When NOT to Use:**
- Need fast/real-time processing
- Interpretable dimensions required
- Large datasets (>10k points)
- Reproducible results needed
- Feature extraction for ML models
- Preserving global structure
- Deterministic results needed
- Very large datasets (slow)

**Pros:**
- Excellent visualization capability
- Preserves local structure
- Reveals non-linear patterns
- Great for cluster exploration

**Cons:**
- Computationally expensive
- Non-deterministic results
- Hyperparameter sensitive
- Not suitable for new data projection

**Key Parameters:**
- **Perplexity**: Local neighborhood size (5-50)
- **Learning Rate**: Step size (10-1000)
- **Iterations**: Usually 1000+

<div style="background: linear-gradient(135deg,rgb(139, 162, 80),rgb(50, 165, 136)); 
            color: #ffffff; 
            width: 100%; 
            height: 40px; 
            text-align: center; 
            font-weight: bold; 
            line-height: 40px; 
            margin: 15px 0; 
            font-size: 24px; 
            border-radius: 10px; 
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3);">
    3. Autoencoders
</div>

**Architecture:**
```
Input → Encoder → Latent Space → Decoder → Reconstruction
```

**Loss Function:**
$$
L = ||x - \hat{x}||^2 + \lambda \cdot R(\theta)
$$

**Types:**
- **Vanilla**: Basic encoder-decoder
- **Denoising**: Learn from corrupted input
- **Variational (VAE)**: Probabilistic latent space
- **Sparse**: Sparse hidden representations

**When to Use:**
- Complex non-linear relationships
- Large datasets available
- Need generative capabilities
- Custom loss functions required
- Non-linear dimensionality reduction
- Complex data (images, sequences)

**When NOT to Use:**
- Small datasets
- Simple linear relationships
- Interpretability crucial
- Limited computational resources
- Linear relationships sufficient
- Need fast training/inference

**Pros:**
- Captures complex patterns
- Flexible architecture
- Can generate new data
- Handles various data types

**Cons:**
- Requires large datasets
- Computationally intensive
- Black box (less interpretable)
- Hyperparameter tuning complex

<div style="width: 100%; height: 10px; background: linear-gradient(to right, gray, white, orange,white, gray); border-radius: 5px; margin: 20px 0;"></div>

## Anomaly Detection

Identify data points that deviate significantly from normal patterns.

<div style="background: linear-gradient(135deg,rgb(139, 162, 80),rgb(50, 165, 136)); 
            color: #ffffff; 
            width: 100%; 
            height: 40px; 
            text-align: center; 
            font-weight: bold; 
            line-height: 40px; 
            margin: 15px 0; 
            font-size: 24px; 
            border-radius: 10px; 
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3);">
    1. Isolation Forest
</div>

**Core Principle:**
- Anomalies are easier to isolate
- Build random binary trees
- Shorter paths = more anomalous

**Anomaly Score:**
$$
s(x,n) = 2^{-\frac{E(h(x))}{c(n)}}
$$

Where:
- $E(h(x))$: Average path length
- $c(n)$: Average path length of BST

**When to Use:**
- Large datasets with few anomalies
- High-dimensional data
- Fast anomaly detection needed
- No labeled anomaly examples
- Large datasets
- No assumptions about normal data

**When NOT to Use:**
- Small datasets
- High anomaly rates expected
- Need interpretable results
- Categorical data dominant
- Need probability estimates
- Interpretable anomaly reasons
- Seasonal or trend data

**Pros:**
- Fast and scalable
- Handles high dimensions well
- No need for labeled data
- Linear time complexity

**Cons:**
- Less effective in very high dimensions
- Hyperparameter sensitivity
- May struggle with normal data variations
- Black box approach

<div style="background: linear-gradient(135deg,rgb(139, 162, 80),rgb(50, 165, 136)); 
            color: #ffffff; 
            width: 100%; 
            height: 40px; 
            text-align: center; 
            font-weight: bold; 
            line-height: 40px; 
            margin: 15px 0; 
            font-size: 24px; 
            border-radius: 10px; 
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3);">
    2. One-Class SVM
</div>

**Objective:**
- Find hyperplane separating normal data from origin
- Maximize margin to origin
- Points far from hyperplane are anomalous

**Decision Function:**
$$
f(x) = \sum_{i} \alpha_i K(x_i, x) - \rho
$$

**When to Use:**
- Clear normal data definition
- Non-linear decision boundaries needed
- Medium-sized datasets
- Few anomalies expected
- Clear boundary between normal/abnormal
- High-dimensional data
- Kernel trick needed
- Robust to outliers in training

**When NOT to Use:**
- Large datasets (computational cost)
- High-dimensional sparse data
- Interpretability required
- Imbalanced normal data
- Large datasets (slow training)
- Need probability estimates
- Normal data has multiple modes
- Real-time detection

**Pros:**
- Effective for non-linear patterns
- Robust to outliers in training
- Solid theoretical foundation
- Flexible kernel choices

**Cons:**
- Computationally expensive
- Sensitive to hyperparameters
- Memory intensive
- Difficult to interpret

<div style="background: linear-gradient(135deg,rgb(139, 162, 80),rgb(50, 165, 136)); 
            color: #ffffff; 
            width: 100%; 
            height: 40px; 
            text-align: center; 
            font-weight: bold; 
            line-height: 40px; 
            margin: 15px 0; 
            font-size: 24px; 
            border-radius: 10px; 
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3);">
    3. Local Outlier Factor (LOF)
</div>

**Core Concept:**
- Compare local density to neighbors
- High LOF = low local density = anomaly

**Local Reachability Density:**
$$
lrd_k(A) = \frac{1}{\frac{\sum_{B \in N_k(A)} reach\text{-}dist_k(A,B)}{|N_k(A)|}}
$$

**LOF Score:**
$$
LOF_k(A) = \frac{\sum_{B \in N_k(A)} \frac{lrd_k(B)}{lrd_k(A)}}{|N_k(A)|}
$$

**When to Use:**
- Varying density regions
- Local anomaly context important
- Interpretable anomaly scores needed
- Moderate dataset sizes
- Local anomalies important
- Varying data densities
- Need to understand anomaly context
- Medium-sized datasets

**When NOT to Use:**
- Very large datasets
- Uniform data density
- Real-time detection needed
- High-dimensional data
- Global anomalies more important
- Real-time requirements

**Pros:**
- Handles varying densities
- Interpretable anomaly scores
- Good for local anomalies
- No assumptions about distribution

**Cons:**
- Computationally expensive O(n²)
- Sensitive to k parameter
- Memory intensive
- Poor scalability

<div style="width: 100%; height: 10px; background: linear-gradient(to right, gray, white, orange,white, gray); border-radius: 5px; margin: 20px 0;"></div>

## Performance Comparison

| **📂 Category**              | **🧠 Algorithm**           | ⚡ **Training Speed** | 🚀 **Prediction Speed** | 🎯 **Accuracy** | 🔍 **Interpretability** | 📊 **Scalability** | 🔧 **Parameter Tuning** |
| ---------------------------- | -------------------------- | -------------------- | ----------------------- | --------------- | ----------------------- | ------------------ | ----------------------- |
| **Clustering**               | **K-Means**                | 🟢 Fast              | 🟢 Very Fast            | 🟡 Medium       | 🟢 High                 | 🟢 Excellent       | 🟡 Medium               |
|                              | **DBSCAN**                 | 🟡 Medium            | 🟡 Medium               | 🟢 High         | 🟡 Medium               | 🟡 Medium          | 🔴 Hard                 |
|                              | **Hierarchical**           | 🔴 Slow              | 🟢 Fast (pre-computed)  | 🟢 High         | 🟢 Very High            | 🔴 Poor            | 🟢 Easy                 |
| **Dimensionality Reduction** | **PCA**                    | 🟢 Fast              | 🟢 Very Fast            | 🟡 Medium       | 🟢 High                 | 🟢 Excellent       | 🟢 Easy                 |
|                              | **t-SNE**                  | 🔴 Slow              | 🔴 N/A (no transform)   | 🟢 High         | 🔴 Low                  | 🔴 Poor            | 🔴 Hard                 |
|                              | **Autoencoders**           | 🔴 Slow              | 🟢 Fast                 | 🟢 Very High    | 🔴 Low                  | 🟢 Good            | 🔴 Very Hard            |
| **Anomaly Detection**        | **Isolation Forest**       | 🟢 Fast              | 🟢 Fast                 | 🟢 High         | 🟡 Medium               | 🟢 Excellent       | 🟢 Easy                 |
|                              | **One-Class SVM**          | 🔴 Slow              | 🟢 Fast                 | 🟢 High         | 🔴 Low                  | 🔴 Poor            | 🔴 Hard                 |
|                              | **LOF**                    | 🟡 Medium            | 🔴 Slow                 | 🟢 High         | 🟢 High                 | 🔴 Poor            | 🟡 Medium               |

### 📊 Clustering Comparison Table

| Algorithm | Time Complexity | Cluster Shapes | Outlier Handling | Parameters | Best For |
|-----------|-----------------|----------------|------------------|------------|----------|
| **K-Means** | O(n*k*i) | Spherical | Poor | k, max_iter | Large datasets, spherical clusters |
| **DBSCAN** | O(n log n) | Arbitrary | Excellent | eps, min_samples | Irregular shapes, outlier detection |
| **Hierarchical** | O(n³) | Any | Moderate | linkage, distance | Small datasets, hierarchy needed |

### 📊 Dimensionality Reduction Comparison

| Method | Type | Speed | Interpretability | Best Use Case | Scalability |
|--------|------|-------|------------------|---------------|-------------|
| **PCA** | Linear | Fast | High | Preprocessing, feature reduction | Excellent |
| **t-SNE** | Non-linear | Slow | Low | Visualization, exploration | Poor |
| **Autoencoders** | Non-linear | Medium | Low | Complex patterns, generation | Good |

### 📊 Anomaly Detection Comparison

| Method | Speed | Scalability | Interpretability | Best For | Memory Usage |
|--------|-------|-------------|------------------|----------|--------------|
| **Isolation Forest** | Fast | High | Low | Large datasets, high dimensions | Low |
| **One-Class SVM** | Slow | Medium | Medium | Non-linear boundaries | High |
| **LOF** | Slow | Low | High | Local anomalies, varying density | High |

<div style="width: 100%; height: 10px; background: linear-gradient(to right, gray, white, orange,white, gray); border-radius: 5px; margin: 20px 0;"></div>

## Evaluation Metrics

### 🔍 Clustering Metrics

**Internal Metrics (No ground truth needed):**
- **Silhouette Score**: $s = \frac{b-a}{\max(a,b)}$ where $a$ = intra-cluster distance, $b$ = nearest-cluster distance
  - Range: -1 to 1, higher is better
  - Measures cluster cohesion and separation
- **Calinski-Harabasz Index**: Ratio of between-cluster to within-cluster variance
  - Higher values indicate better clustering
  - Also known as Variance Ratio Criterion
- **Davies-Bouldin Index**: Average similarity between clusters
  - Lower values indicate better clustering
  - Measures average similarity between each cluster and its most similar cluster

**External Metrics (Ground truth available):**
- **Adjusted Rand Index**: Measures similarity to true clustering
  - Range: 0 to 1, higher is better
  - Adjusted for chance agreement
- **Normalized Mutual Information**: Information shared between clusterings
  - Range: 0 to 1, higher is better
- **Homogeneity & Completeness**: Cluster purity measures
  - Homogeneity: Each cluster contains only members of a single class
  - Completeness: All members of a given class are assigned to the same cluster

```python
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

# Internal evaluation
silhouette_avg = silhouette_score(X, cluster_labels)
ch_score = calinski_harabasz_score(X, cluster_labels)

# External evaluation (if true labels available)
ari_score = adjusted_rand_score(true_labels, cluster_labels)
nmi_score = normalized_mutual_info_score(true_labels, cluster_labels)
```

### 📏 Dimensionality Reduction Metrics

- **Explained Variance Ratio**: Proportion of variance preserved (PCA)
  - Higher values indicate better variance retention
- **Reconstruction Error**: $||X - X_{reconstructed}||^2$
  - Lower values indicate better reconstruction
  - Difference between original and reconstructed data
- **Trustworthiness**: Preservation of local neighborhoods
  - Measures how well local structure is preserved
- **Continuity**: Preservation of local structure
  - Smoothness of the embedding

### 🚨 Anomaly Detection Metrics

- **Precision**: $\frac{TP}{TP + FP}$
  - Proportion of identified anomalies that are truly anomalous
- **Recall**: $\frac{TP}{TP + FN}$
  - Proportion of actual anomalies that were correctly identified
- **F1-Score**: $\frac{2 \cdot Precision \cdot Recall}{Precision + Recall}$
  - Harmonic mean of precision and recall
- **AUC-ROC**: Area under receiver operating characteristic curve
  - Higher values indicate better discrimination
- **Average Precision**: Area under precision-recall curve
  - Better for imbalanced datasets than AUC-ROC
- **Contamination Rate**: Expected proportion of anomalies in the dataset

<div style="width: 100%; height: 10px; background: linear-gradient(to right, gray, white, orange,white, gray); border-radius: 5px; margin: 20px 0;"></div>

## Decision Framework

### 1. Problem Identification
```python
# Key questions to ask:
questions = {
    "What's your goal?": ["Group similar items", "Reduce dimensions", "Find outliers"],
    "Data size?": ["< 1K", "1K-100K", "> 100K"],
    "Dimensionality?": ["< 50", "50-1000", "> 1000"],
    "Prior knowledge?": ["Know # clusters", "Unknown structure", "Have examples"],
    "Computational budget?": ["Real-time", "Minutes", "Hours"]
}
```

### 2. Algorithm Selection Strategy

**For Clustering:**
```python
def select_clustering_algorithm(data_size, cluster_shape, known_k):
    if known_k and cluster_shape == "spherical":
        return "K-Means"
    elif cluster_shape == "arbitrary" and data_size < 50000:
        return "DBSCAN"
    elif need_hierarchy:
        return "Hierarchical"
    else:
        return "K-Means with preprocessing"
```

**For Dimensionality Reduction:**
```python
def select_dim_reduction(purpose, data_size, relationship):
    if purpose == "visualization":
        return "t-SNE" if data_size < 10000 else "PCA then t-SNE"
    elif relationship == "linear":
        return "PCA"
    elif data_size > 100000:
        return "PCA" if relationship == "linear" else "Autoencoders"
    else:
        return "PCA"
```

### 3. Hyperparameter Tuning Guidelines

**K-Means:**
- Use elbow method or silhouette analysis for k
- Try k-means++ initialization
- Set max_iter based on convergence needs

**DBSCAN:**
- Use k-distance graph for eps selection
- MinPts ≈ 2 × dimensions (rule of thumb)
- Test multiple parameter combinations

**PCA:**
- Choose n_components based on explained variance (90-95%)
- Consider standardization for mixed-scale features

<div style="width: 100%; height: 10px; background: linear-gradient(to right, gray, white, orange,white, gray); border-radius: 5px; margin: 20px 0;"></div>

## 🚀 Quick Start Templates

### Clustering Pipeline
```python
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# Preprocessing
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# K-Means with elbow method
inertias = []
silhouette_scores = []
k_range = range(2, 11)

for k in k_range:
    kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42)
    labels = kmeans.fit_predict(X_scaled)
    inertias.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X_scaled, labels))

# Best k based on silhouette score
best_k = k_range[np.argmax(silhouette_scores)]
```

### Dimensionality Reduction Pipeline
```python
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# PCA for variance analysis
pca = PCA()
pca.fit(X_scaled)

# Plot explained variance
plt.figure(figsize=(10, 6))
plt.subplot(1, 2, 1)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Components')
plt.ylabel('Cumulative Explained Variance')

# Apply PCA with optimal components
n_components = np.argmax(np.cumsum(pca.explained_variance_ratio_) >= 0.95) + 1
pca_reduced = PCA(n_components=n_components)
X_pca = pca_reduced.fit_transform(X_scaled)

# t-SNE for visualization
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
X_tsne = tsne.fit_transform(X_pca)  # Apply on PCA results for speed
```

### Anomaly Detection Pipeline
```python
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor

# Multiple algorithms comparison
algorithms = {
    'Isolation Forest': IsolationForest(contamination=0.1, random_state=42),
    'One-Class SVM': OneClassSVM(nu=0.1),
    'LOF': LocalOutlierFactor(n_neighbors=20, contamination=0.1)
}

results = {}
for name, algorithm in algorithms.items():
    # LOF returns labels directly, others need predict
    if name == 'LOF':
        predictions = algorithm.fit_predict(X_scaled)
    else:
        predictions = algorithm.fit(X_scaled).predict(X_scaled)
    
    # Convert to binary (1 = normal, -1 = anomaly)
    anomaly_count = sum(predictions == -1)
    results[name] = {
        'predictions': predictions,
        'anomaly_count': anomaly_count,
        'anomaly_ratio': anomaly_count / len(X)
    }

print("Anomaly Detection Results:")
for name, result in results.items():
    print(f"{name}: {result['anomaly_count']} anomalies ({result['anomaly_ratio']:.2%})")
```

<div style="width: 100%; height: 10px; background: linear-gradient(to right, gray, white, orange,white, gray); border-radius: 5px; margin: 20px 0;"></div>

## 🚀 Quick Start Examples

### Clustering Example
```python
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler

# Prepare data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# K-Means
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

# DBSCAN
dbscan = DBSCAN(eps=0.5, min_samples=5)
clusters = dbscan.fit_predict(X_scaled)
```

### Dimensionality Reduction Example
```python
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# t-SNE
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X_scaled)
```

### Anomaly Detection Example
```python
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM

# Isolation Forest
iso_forest = IsolationForest(contamination=0.1, random_state=42)
anomalies = iso_forest.fit_predict(X_scaled)

# One-Class SVM
oc_svm = OneClassSVM(nu=0.1)
anomalies = oc_svm.fit_predict(X_scaled)
```

---

*This comprehensive guide provides everything you need to master unsupervised learning techniques. Each algorithm section includes detailed theory, practical implementation guidance, and real-world application scenarios.*