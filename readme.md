# PCA and Agglomerative Clustering on Iris Dataset

This repository contains the source code and resources for the **PCA and Agglomerative Clustering on the Iris Dataset** project. This project demonstrates the application of Principal Component Analysis (PCA) for dimensionality reduction and Agglomerative Clustering for hierarchical clustering on the Iris dataset.

## Table of Contents

- [Project Overview](#project-overview)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
  - [Imports](#imports)
  - [Data Loading](#data-loading)
  - [Data Preprocessing](#data-preprocessing)
  - [Modeling](#modeling)
  - [Visualization](#visualization)
- [License](#license)
- [Contact](#contact)

## Project Overview

The **PCA and Agglomerative Clustering on the Iris Dataset** project includes the following key steps:

- **Data Loading**: Importing the Iris dataset from `sklearn`.
- **Data Preprocessing**: Standardizing the data to normalize the feature values.
- **Dimensionality Reduction**: Applying Principal Component Analysis (PCA) to reduce the dataset dimensions from 4 to 2 for easier visualization.
- **Clustering**: Performing Agglomerative Clustering on the reduced dataset to classify the data into clusters.
- **Visualization**: Plotting the dendrogram to visualize the hierarchical clustering and to identify the optimal number of clusters.

## Project Structure

- **notebook.ipynb**: The Jupyter notebook containing the complete code for the analysis, from data loading to clustering and visualization.
- **LICENSE**: The Apache License 2.0 file that governs the use and distribution of this project's code.
- **requirements.txt**: A file listing all the Python libraries and dependencies required to run the project.
- **.gitignore**: A file specifying which files or directories should be ignored by Git.

## Installation

To run the project locally, follow these steps:

1. **Clone the repository**:
   ```bash
   git clone repository
```
2. Navigate to the project directory:

``` bash
cd your-repository-name

```
3.Create a virtual environment (optional but recommended):

``` bash
python3 -m venv venv
source venv/bin/activate   # On Windows use `venv\Scripts\activate`

```

4. Install the required dependencies:

```bash 

pip install -r requirements.txt
```

5. Run the Jupyter notebook:

``` bash
jupyter notebook notebook.ipynb
```

##  Usage

Imports

The notebook begins by importing the necessary libraries:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import scipy.cluster.hierarchy as sc
from sklearn.cluster import AgglomerativeClustering

These libraries are essential for data manipulation (`pandas`, `numpy`), visualization (`matplotlib`, `seaborn`), dimensionality reduction (PCA), and clustering (Agglomerative Clustering and hierarchical clustering).

## Data Loading
The Iris dataset is loaded from the sklearn.datasets module:

``` bash

iris = datasets.load_iris()
iris_data = iris.data
```
## Data Preprocessing
The data is standardized using StandardScaler to normalize the feature values, ensuring that each feature contributes equally to the analysis:

``` bash

scaler = StandardScaler()
X_scaled = scaler.fit_transform(iris_data)

```
## Modeling

Dimensionality Reduction

Principal Component Analysis (PCA) is applied to reduce the dataset's dimensions from 4 to 2, facilitating easier visualization and clustering:

``` bash

pca = PCA(n_components=2)
pca_scaled = pca.fit_transform(X_scaled)

```

Agglomerative Clustering

Agglomerative Clustering is then applied to the reduced dataset to group the data into clusters:

``` bash
cluster = AgglomerativeClustering(n_clusters=2, linkage='ward')
cluster.fit(pca_scaled)
```

Visualization
The dendrogram is plotted to visualize the hierarchical clustering and to help identify the optimal number of clusters:

```bash
plt.figure(figsize=(20, 7))
plt.title('Dendograms')
sc.dendrogram(sc.linkage(pca_scaled, method='ward'))
plt.xlabel('Sample index')
plt.ylabel('Euclidean distance')
plt.show()
```

This visualization aids in understanding the structure of the data and how the clusters are formed.

## License
This project is licensed under the Apache License 2.0. See the `LICENSE` file for more details.

## Contact
For any inquiries or contributions, feel free to reach out or submit an issue or pull request on GitHub.