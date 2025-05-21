# Temporal Fusion Transformer
import numpy as np

def create_sequences(df, sequence_length=7, forecast_horizon=1):
    X, y = [], []
    
    for i in range(len(df) - sequence_length - forecast_horizon + 1):
        seq_input = df.iloc[i:i + sequence_length][[
            'Toplam_Devamsızlık_Saati', 'lag_1', 'lag_7', 'day', 'month', 'weekday',
            'is_saturday', 'dayofyear', 'weekofyear', 'year'
        ]]
        seq_output = df.iloc[i + sequence_length:i + sequence_length + forecast_horizon]
        
        X.append(seq_input.values)
        y.append(seq_output['Toplam_Devamsızlık_Saati'].values)
    
    return np.array(X), np.array(y)

clean_data = daily_absence_reset.dropna()
X_seq, y_seq = create_sequences(clean_data)

from pytorch_forecasting import TimeSeriesDataSet
import pandas as pd

# First: create a time_idx
clean_data = daily_absence_reset.dropna().copy()
clean_data['time_idx'] = range(len(clean_data))  # simple increasing index

clean_data['group_id'] = "absence_series"

clean_data = clean_data.rename(columns={
    'Toplam_Devamsızlık_Saati': 'target'
})

cols = ['Tarih', 'time_idx', 'group_id', 'target', 'lag_1', 'lag_7',
        'day', 'month', 'weekday', 'is_saturday', 'dayofyear', 'weekofyear', 'year']
clean_data = clean_data[cols]

max_encoder_length = 30
max_prediction_length = 7

training_cutoff = clean_data["time_idx"].max() - max_prediction_length

training = TimeSeriesDataSet(
    clean_data[clean_data.time_idx <= training_cutoff],
    time_idx="time_idx",
    target="target",
    group_ids=["group_id"],
    max_encoder_length=max_encoder_length,
    max_prediction_length=max_prediction_length,
    time_varying_known_reals=["time_idx", "day", "month", "weekday", "is_saturday", "dayofyear", "weekofyear", "year"],
    time_varying_unknown_reals=["target", "lag_1", "lag_7"],
    static_categoricals=["group_id"],
)

from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.data import NaNLabelEncoder
from pytorch_forecasting.data.encoders import TorchNormalizer

# Add a group column since TFT expects one
daily_absence_reset['group'] = "absence"  # single group

# Define max encoder & prediction lengths
max_encoder_length = 30  # past 30 days
max_prediction_length = 7  # predict next 7 days

daily_absence_reset["lag_1"] = daily_absence_reset.groupby("group")["lag_1"].transform(lambda x: x.ffill().fillna(0))
daily_absence_reset["lag_7"] = daily_absence_reset.groupby("group")["lag_7"].transform(lambda x: x.ffill().fillna(0))

# Create TimeSeriesDataSet
tft_dataset = TimeSeriesDataSet(
    daily_absence_reset,
    time_idx="dayofyear",  # Use continuous daily index
    target="Toplam_Devamsızlık_Saati",
    group_ids=["group"],
    max_encoder_length=max_encoder_length,
    max_prediction_length=max_prediction_length,
    time_varying_known_reals=["day", "month", "weekday", "is_saturday", "dayofyear", "weekofyear", "year"],
    time_varying_unknown_reals=["Toplam_Devamsızlık_Saati", "lag_1", "lag_7"],
    target_normalizer=TorchNormalizer(method="standard"),
    add_relative_time_idx=True,
    add_target_scales=True,
    add_encoder_length=True,
    allow_missing_timesteps=True,  # ✅ fixes the gap-in-time issue
)

daily_absence_reset['time_idx'] = (daily_absence_reset['Tarih'] - daily_absence_reset['Tarih'].min()).dt.days

training_cutoff = daily_absence_reset["Tarih"].max() - pd.Timedelta(days=max_prediction_length)
training_cutoff_time_idx = (training_cutoff - daily_absence_reset["Tarih"].min()).days

train_dataset = tft_dataset.filter(lambda x: x["time_idx_first_prediction"] <= training_cutoff_time_idx)
val_dataset = tft_dataset.filter(lambda x: x["time_idx_first_prediction"] > training_cutoff_time_idx)


from torch.utils.data import DataLoader

batch_size = 32

train_dataloader = train_dataset.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
val_dataloader = val_dataset.to_dataloader(train=False, batch_size=batch_size, num_workers=0)

from pytorch_forecasting.metrics import QuantileLoss
import torch
from pytorch_forecasting import TemporalFusionTransformer

tft = TemporalFusionTransformer.from_dataset(
    train_dataset,
    learning_rate=0.03,
    hidden_size=16,
    attention_head_size=1,
    dropout=0.1,
    loss=QuantileLoss(),
    log_interval=10,
    reduce_on_plateau_patience=4,
)
from pytorch_lightning import Trainer

if torch.cuda.is_available():
    accelerator = "gpu"
    devices = 1
else:
    accelerator = "cpu"
    devices = 1  # Must be an integer > 0

trainer = Trainer(
    max_epochs=30,
    accelerator=accelerator,
    devices=devices,
    gradient_clip_val=0.1,
)

# Raw predictions
raw_output = tft.predict(val_dataloader, mode="raw", return_x=True)

# Correct unpacking
raw_predictions, x = raw_output[0], raw_output[1]

# Plot
tft.plot_prediction(x, raw_predictions, idx=0)

# Graph output screenshot is provided as a file


