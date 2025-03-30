
# Unsupervised Learning (Clustering) - README

## Description:
This section applies unsupervised learning techniques to cluster visitors based on their mode of entry (Air, Land, Sea). The models used include:
- K-Means Clustering
- PCA for dimensionality reduction
- Cluster evaluation metrics (Silhouette Score, Davies-Bouldin Index)

## Dependencies:
- scikit-learn
- pandas
- numpy
- matplotlib

## How to Run:
1. Install the required libraries:
   ```bash
   pip install scikit-learn pandas numpy matplotlib
   ```
2. Run the script `unsupervised_learning.py` to perform clustering and visualize the results.
3. The script will display a 3D scatter plot of the clustered data and provide evaluation metrics for the clustering performance.

## Steps:
1. **Data Preprocessing**: Standardize the data before applying clustering algorithms.
2. **Clustering**: Apply K-Means clustering to segment the visitors into groups.
3. **PCA**: Use PCA for dimensionality reduction and visualization in 2D/3D.
4. **Cluster Evaluation**: Evaluate the quality of clusters using metrics like Silhouette Score and Davies-Bouldin Index.
