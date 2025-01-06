import pandas as pd
import matplotlib.pyplot as plt
import os
from itertools import cycle

cycol = cycle('bgrcmk')

folder_path = os.path.join(os.getcwd(), "dataset_after_clusterization")
file_paths = []

for filename in os.listdir(folder_path):
    file_paths.append(os.path.join(folder_path, filename))

plt.figure(figsize=(110, 70))

for file_path in file_paths:

    data = pd.read_csv(file_path, header=None)
    values = data[0].values

    time = range(1, len(values) + 1)

    file_name = os.path.splitext(os.path.basename(file_path))[0].split('_')
    label = f"Klaster {file_name[-1]} - node {file_name[2]} to node {file_name[5]}"
    plt.plot(time, values, label=label, color=next(cycol))

plt.xlabel("Czas (sekundy)")
plt.ylabel("Natężenie ruchu")
plt.title("Reprezentatywne klastry")
plt.legend()
# plt.savefig("cluster.png")
plt.show()
