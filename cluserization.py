import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import os

def calculate_features(file_path):
    raw_data = np.loadtxt(file_path)
    node_a = int(raw_data[0])
    node_b = int(raw_data[1])
    data = raw_data[2:]
    return {
        "mean": np.mean(data),
        "std": np.std(data),
        "max": np.max(data),
        "min": np.min(data),
        "pair_id": f"node_{node_a}_to_node_{node_b}"
    }

file_paths = []

folder_path = os.path.join(os.getcwd(), "demands_for_students")
for filename in os.listdir(folder_path):
    file_paths.append(os.path.join(folder_path, filename))

features = [calculate_features(fp) for fp in file_paths]
df_features = pd.DataFrame(features)

print(df_features)

# opcjonalna standaryzacja - zalecane wedlug zrodel
scaler = StandardScaler()
features_scaled = scaler.fit_transform(df_features.drop(columns=["pair_id"]))

n_clusters = 5  # pamietac o dostosowaniu liczby klastrow
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
df_features["cluster"] = kmeans.fit_predict(features_scaled)

print(df_features)

for cluster_id in range(n_clusters):
    print(f"\nCluster {cluster_id}:")
    cluster_data = df_features[df_features["cluster"] == cluster_id]
    print(cluster_data)

# idxmax -> najwieksza srednia ruchu w klastrze, mozna rozwazyc zastosowanie innej reguly
representative_pairs = df_features.loc[df_features.groupby("cluster")["mean"].idxmax()]

print("Reprezentatywne pary dla klastrów:")
print(representative_pairs)


plt.figure(figsize=(10, 6))
for cluster_id in range(n_clusters):
    cluster_data = df_features[df_features["cluster"] == cluster_id]
    plt.scatter(cluster_data["mean"], cluster_data["std"], label=f"Cluster {cluster_id}")

plt.title("Clustering of Node Pairs")
plt.xlabel("Mean Traffic")
plt.ylabel("Traffic Standard Deviation")
plt.legend()
plt.savefig("cluster_plot.png")
plt.show()

pairs = []

for pair_id in representative_pairs["pair_id"]:
    pairs.append(pair_id)

for fp in file_paths:
    with open(fp, "r") as f:
        lines = f.readlines()
        content_to_copy = lines[3:]
        pair = f"node_{lines[1].strip()}_to_node_{lines[2].strip()}"
        if pair in pairs:
            cluster = representative_pairs[representative_pairs["pair_id"] == pair]["cluster"].iloc[0]
            new_file_path = os.path.join(os.getcwd(), "dataset_after_clusterization", f"{os.path.splitext(os.path.basename(fp))[0]}_{pair}_"
                                                                                      f"{cluster}_{os.path.splitext(os.path.basename(fp))[1]}")
            with open(new_file_path, "w") as new_f:
                new_f.writelines(content_to_copy)
                new_f.close()
        f.close()