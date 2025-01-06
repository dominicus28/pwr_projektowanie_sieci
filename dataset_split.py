import numpy as np
import pandas as pd
import os


time_idx = np.arange(19296)
split_idx = int(19296 * 0.8)  # 80% próbek na trening
train_data = []
test_data = []

folder_path = os.path.join(os.getcwd(), "dataset_after_clusterization")
for filename in os.listdir(folder_path):
    file_path = os.path.join(folder_path, filename)
    traffic_data = np.loadtxt(file_path)
    cluster = filename.split('_')[-2]

    df = pd.DataFrame({
        "time_idx": time_idx,
        "value": traffic_data,
        "group": cluster,  # Identyfikator klastra jako grupa
        "static_feature": cluster,  # Cecha statyczna (klaster)
    })
    train_part = df.iloc[:split_idx].reset_index(drop=True)
    test_part = df.iloc[split_idx:].reset_index(drop=True)
    train_data.append(train_part)
    test_data.append(test_part)

# Łączenie danych w jedną ramkę danych
final_train_data = pd.concat(train_data, ignore_index=True)
final_test_data = pd.concat(test_data, ignore_index=True)

final_train_data = final_train_data.sort_values(by=["group", "time_idx"])
final_test_data = final_test_data.sort_values(by=["group", "time_idx"])

final_train_data.to_csv("training_data.csv", index=False)
final_test_data.to_csv("testing_data.csv", index=False)