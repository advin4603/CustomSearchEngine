import numpy as np
from sklearn.decomposition import PCA
from createTermDocumentMatrix import VectorTermDocumentMatrix
import argparse
from pathlib import Path
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(prog="cluster_and_plot",
                                 description="clusters and plots document vectors")
parser.add_argument("vector_term_matrix_path")
parser.add_argument("-k", "--clusters", default="3")
args = parser.parse_args()
k = int(args.clusters)
vector_term_matrix_path = Path(args.vector_term_matrix_path)
vector_matrix = VectorTermDocumentMatrix.load(vector_term_matrix_path)

pca = PCA(2)
reduced_tf_idf = pca.fit_transform(np.array(vector_matrix.matrix))

kmeans = KMeans(n_clusters=k)
labels = kmeans.fit_predict(reduced_tf_idf)
unique_labels = np.unique(labels)

for cluster_num in unique_labels:
    print(cluster_num)
    print("\n".join(doc.name for doc, label in zip(vector_matrix.document_list, labels) if label == cluster_num))
    plt.scatter(reduced_tf_idf[labels == cluster_num, 0], reduced_tf_idf[labels == cluster_num, 1], label=cluster_num)

plt.legend()
plt.show()
