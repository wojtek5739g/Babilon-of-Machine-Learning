import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

calendar = pd.read_csv('./calendar.csv')
sales_train = pd.read_csv('./sales_train_validation.csv')
sell_prices = pd.read_csv('./sell_prices.csv')

d_cols = [col for col in sales_train.columns if "d_" in col]


data = sales_train[d_cols].values
corr = np.corrcoef(data, rowvar=False)

nrows, ncols = corr.shape

# dist_matrix = 1 - corr
# distances = []
# for n_clusters in range(1, 10):
#     kmeans = KMeans(n_clusters=n_clusters)
#     kmeans.fit(dist_matrix)
#     distances.append(kmeans.inertia_)
    
# plt.plot(range(1, 10), distances, marker='o')
# plt.xlabel('Number of clusters')
# plt.ylabel('Inertia')
# plt.show()


np.fill_diagonal(corr, 0)
mean_corr = np.mean(corr, axis=0)


X = np.array(mean_corr).reshape(-1, 1)
n_clusters = 4
kmeans = KMeans(n_clusters=n_clusters)

# Dopasowujemy model do danych
kmeans.fit(X)
# Otrzymujemy przypisania klastrów dla każdej kolumny macierzy
cluster_labels = kmeans.labels_

# Wyświetlamy wyniki
for i in range(n_clusters):
    cluster_i_indices = np.where(cluster_labels == i)[0]
    # print(f"Cluster {i}: {cluster_i_indices}")
    print(f"Cluster {i}: {len(cluster_i_indices)} dni")

cluster_corr_matrices = []
for i in range(n_clusters):
    cluster_i_indices = np.where(cluster_labels == i)[0]
    cluster_i_data = data[:, cluster_i_indices]
    cluster_i_corr = np.corrcoef(cluster_i_data, rowvar=False)
    cluster_corr_matrices.append(cluster_i_corr)

cluster_corr_means = []
for i in range(n_clusters):
    cluster_i_indices = np.where(cluster_labels == i)[0]
    cluster_i_data = data[:, cluster_i_indices]
    cluster_i_corr = np.corrcoef(cluster_i_data, rowvar=False)
    cluster_corr_means.append(np.mean(cluster_i_corr))

max_mean_corr_cluster = np.argmax(cluster_corr_means)
print(f"Cluster {max_mean_corr_cluster} has the highest mean correlation: {cluster_corr_means[max_mean_corr_cluster]}")

for i in range(n_clusters):
    cluster_i_indices = np.where(cluster_labels == i)[0]
    cluster_i_data = data[:, cluster_i_indices]
    cluster_i_corr = np.corrcoef(cluster_i_data, rowvar=False)
    weak_corr_mask = np.abs(cluster_i_corr) < 0.5
    weak_corr_indices = np.argwhere(weak_corr_mask)
    if weak_corr_indices.size > 0:
        print(f"Cluster {i} has weak correlations between days:")
        for index in weak_corr_indices:
            day1 = cluster_i_indices[index[0]]
            day2 = cluster_i_indices[index[1]]
            print(f"{day1} and {day2}")

for i in range(n_clusters):
    cluster_i_indices = np.where(cluster_labels == i)[0]
    cluster_i_heatmap = sns.heatmap(cluster_corr_matrices[i], cmap='coolwarm')
    cluster_i_heatmap.set_title(f'Cluster {i+1} Correlation Matrix')
    new_xticks = cluster_i_indices
    new_yticks = cluster_i_indices
    if (len(cluster_corr_matrices[i]) > 6):
        plt.xticks(np.arange(0, len(cluster_corr_matrices[i]), 20), new_xticks[::20])
        plt.yticks(np.arange(0, len(cluster_corr_matrices[i]), 20), new_yticks[::20])
        plt.xticks(rotation=90)
    else:
        plt.xticks(np.arange(0, len(cluster_corr_matrices[i]), 1), new_xticks[::1])
        plt.yticks(np.arange(0, len(cluster_corr_matrices[i]), 1), new_yticks[::1])
        plt.xticks(rotation=90)
    
    plt.show()