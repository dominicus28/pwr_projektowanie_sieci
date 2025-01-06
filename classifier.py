import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.metrics import MAPE
from pytorch_forecasting.data import GroupNormalizer
from lightning.pytorch import Trainer

def train():
    train_data = pd.read_csv("training_data.csv")
    train_data["static_feature"] = train_data["static_feature"].astype(str)

    # Przygotowanie zbioru danych
    max_encoder_length = 50  # długość sekwencji wejściowej
    max_prediction_length = 10  # długość prognozy

    training = TimeSeriesDataSet(
        train_data,
        time_idx="time_idx",
        target="value",
        group_ids=["group"],
        max_encoder_length=max_encoder_length,
        max_prediction_length=max_prediction_length,
        static_categoricals=["static_feature"],  # cechy statyczne kategoryczne
        time_varying_known_reals=["time_idx"],  # znane cechy dynamiczne
        time_varying_unknown_reals=["value"],  # cechy dynamiczne do przewidywania
        target_normalizer=GroupNormalizer(groups=["group"], transformation="softplus"),
    )

    train_dataloader = training.to_dataloader(train=True, batch_size=64, num_workers=4, persistent_workers=True)  # Włączenie wielowątkowości

    tft = TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=0.03,
        hidden_size=16,  # liczba neuronów w warstwach ukrytych
        attention_head_size=1,  # liczba głów mechanizmu atencji
        dropout=0.1,  # współczynnik odrzucania
        hidden_continuous_size=8,  # rozmiar dla cech ciągłych
        output_size=1,  # liczba wyjść (np. jednowymiarowe prognozy)
        loss=MAPE(),
        log_interval=0,
    )

    trainer = Trainer(
        max_epochs=10,  # maksymalna liczba epok
        gradient_clip_val=0.1,  # ograniczenie gradientu
        log_every_n_steps=5,  # loguj co 5 kroków
        accelerator="gpu" if torch.cuda.is_available() else "cpu",  # Wykorzystanie GPU, jeśli dostępne
        devices=1  # Liczba GPU/CPU
    )

    trainer.fit(tft, train_dataloaders=train_dataloader)

    trainer.save_checkpoint('tft_model.ckpt')

def predict():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loaded_model = TemporalFusionTransformer.load_from_checkpoint('tft_model.ckpt', map_location=device)

    test_data = pd.read_csv("testing_data.csv")
    test_data["static_feature"] = test_data["static_feature"].astype(str)

    #kontekst wejsciowy w zaleznosci od max_encoder_length
    train_data = pd.read_csv("training_data.csv")
    train_data["static_feature"] = train_data["static_feature"].astype(str)
    max_encoder_length = 50
    max_prediction_length = 10
    last_samples = train_data.groupby("group").apply(
        lambda group: group.iloc[-max_encoder_length:]
    ).reset_index(drop=True)

    new_data = pd.concat([last_samples, test_data], ignore_index=True)

    all_predictions = []

    for group_id, group_data in new_data.groupby("group"):
        print(f"Processing group {group_id}")

        current_group_data = group_data.reset_index(drop=True)

        current_predictions = []

        # petla iterujaca - mechanizm przesuwnego okna - do dostosowania parametry
        for start_idx in range(0, len(group_data) - max_encoder_length, max_prediction_length):
            input_data = current_group_data.iloc[start_idx:start_idx + max_encoder_length + max_prediction_length]

            test_dataset = TimeSeriesDataSet(
                input_data,
                time_idx="time_idx",
                target="value",
                group_ids=["group"],
                max_encoder_length=max_encoder_length,
                max_prediction_length=max_prediction_length,
                static_categoricals=["static_feature"],
                time_varying_known_reals=["time_idx"],
                time_varying_unknown_reals=["value"],
                target_normalizer=GroupNormalizer(groups=["group"], transformation="softplus"),
            )

            test_dataloader = test_dataset.to_dataloader(train=False, batch_size=1, num_workers=0)

            predictions = loaded_model.predict(test_dataloader)

            current_predictions.append(predictions)

        all_predictions.append(torch.cat(current_predictions).detach().numpy())

    all_predictions = np.concatenate(all_predictions, axis=0)

    np.savetxt("raw_predictions.csv", all_predictions, delimiter=",", fmt="%.5f")

def plot_and_evaluate_metrics(predictions_file="raw_predictions.csv", true_data="testing_data.csv"):
    with open('metrics_values.txt', 'w') as file:
        pass
        file.close()

    predictions_data = pd.read_csv(predictions_file, header=None)

    test_data = pd.read_csv(true_data)

    all_groups_predictions = {}

    num_groups = test_data["group"].nunique()
    points_per_group = len(test_data) // num_groups
    prediction_points = predictions_data.shape[1]

    for group_id in range(num_groups):
        start_idx = (group_id * points_per_group) // prediction_points
        end_idx = start_idx + (points_per_group // prediction_points)
        group_predictions = predictions_data.iloc[start_idx:end_idx].values  # Predykcje dla danej grupy

        all_groups_predictions[group_id] = group_predictions.flatten()

    real_data_per_group = {}

    for group_id in range(num_groups):
        group_data = test_data[test_data["group"] == group_id]
        real_data_per_group[group_id] = group_data

    for group_id in range(num_groups):
        group_predictions = all_groups_predictions[group_id]
        group_real_data = real_data_per_group[group_id]

        df = pd.DataFrame({
            'y_true': group_real_data["value"],
            'y_pred': group_predictions
        })

        df['error'] = abs((df['y_true'] - df['y_pred']) / df['y_true']) * 100
        mape_value = df['error'].mean()

        with open('metrics_values.txt', 'a') as file:
            file.write(f"MAPE dla grupy {group_id}: {mape_value}\n")
            file.close()

        plt.figure(figsize=(10, 6))

        plt.plot(group_real_data["time_idx"], group_real_data["value"], label="Rzeczywiste dane", color='blue')

        plt.plot(group_real_data["time_idx"], group_predictions, label=f"Predykcje - Grupa {group_id}", linestyle='--',
                 color='red')

        plt.xlabel("Czas (time_idx)")
        plt.ylabel("Natężenie ruchu")
        plt.title(f"Predykcje vs Rzeczywiste dane dla grupy {group_id}")
        plt.legend()
        plt.savefig(f"predictions_group_{group_id}.png")
        plt.show()


if __name__ == "__main__":
    train()
    predict()
    plot_and_evaluate_metrics()

# TODO
# zbior walidujacy i wykorzystanie go do optymalizacji hiperparametrow
# dynamiczna optymalizacja learning rate
# dostosowanie parametrow przesuwnego okna
# dodanie walidacji krzyzowej dla szeregow czasowych

