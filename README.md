# Deep Learning for Wine Data Clustering
This project applies deep learning techniques to cluster wines based on their chemical properties. The dataset used is the Wine dataset from the UCI Machine Learning Repository.

## Project Overview
The goal of this project is to leverage an autoencoder neural network to extract features from the Wine dataset and then apply K-means clustering on the extracted features to group the wines into distinct clusters.

## Objective

The objective of this project is to:

1. Preprocess the data by standardizing the features.
2. Build and train an autoencoder to learn a compressed representation of the data.
3. Extract features using the encoder part of the autoencoder.
4. Apply K-means clustering on the extracted features to group the wines.
5. Evaluate the clustering performance using the silhouette score.

## Dataset

The Wine dataset contains chemical analysis results of wines grown in the same region in Italy but derived from three different cultivars. The dataset includes the following features:

- Alcohol
- Malic acid
- Ash
- Alcalinity of ash
- Magnesium
- Total phenols
- Flavanoids
- Nonflavanoid phenols
- Proanthocyanins
- Color intensity
- Hue
- OD280/OD315 of diluted wines
- Proline

## Steps

### Step 1: Preprocess the Data

Standardize the chemical properties of the wines to ensure each feature contributes equally to the analysis.

### Step 2: Build and Train an Autoencoder

Construct an autoencoder neural network with an encoder and a decoder. Train the autoencoder to minimize reconstruction error, ensuring the compressed representation retains essential information about the original data.

### Step 3: Extract Features Using the Encoder

Use the trained encoder to transform the wine data into a set of compressed features that capture the underlying structure of the data.

### Step 4: Apply Clustering on Extracted Features

Perform K-means clustering on the compressed features to group the wines into distinct clusters. Determine the optimal number of clusters using the elbow method and silhouette score.

### Step 5: Evaluate the Clustering

Assess the quality of the clustering using the silhouette score, which measures how similar an object is to its own cluster compared to other clusters.

## Results

- **Cluster Visualization**: The scatter plot shows the distribution of wines in the compressed feature space, color-coded by their cluster labels. Each point represents a wine, and the colors indicate different clusters.
- **Silhouette Score**: The silhouette score of 0.343 indicates a moderate level of clustering quality.

## Conclusion

By using a deep learning approach with an autoencoder, we successfully:

- Learned a compressed representation of the wine data that captures essential information about the chemical properties.
- Applied K-means clustering on the compressed features to group the wines into distinct clusters.
- Achieved a moderate level of clustering performance, as indicated by the silhouette score.

This approach demonstrates the effectiveness of combining deep learning techniques with traditional clustering methods to gain deeper insights into complex datasets.


## Requirements

- Python 3.x
- pandas
- numpy
- matplotlib
- scikit-learn
- tensorflow

## Acknowledgments

- UCI Machine Learning Repository for the Wine dataset.
- Scikit-learn and TensorFlow for the machine learning tools.
