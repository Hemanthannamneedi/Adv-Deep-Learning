
import os
import time
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# --------------------------------------------------
# 1. Reproducibility and folders
# --------------------------------------------------

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

os.makedirs("data", exist_ok=True)
os.makedirs("models", exist_ok=True)
os.makedirs("plots", exist_ok=True)

# --------------------------------------------------
# 2. Configuration
# --------------------------------------------------

WINDOW_SIZE = 14
FORECAST_DAYS = 7
EPOCHS = 20
BATCH_SIZE = 32

FEATURE_COLUMNS = [
    "temperature",
    "product_price",
    "promotion",
    "website_visits",
    "sales"
]

TARGET_COLUMN = "sales"
TARGET_INDEX = FEATURE_COLUMNS.index(TARGET_COLUMN)

# --------------------------------------------------
# 3. Generate sample multivariate time series data
# --------------------------------------------------

def generate_sales_data(number_of_days=1000):

    dates = pd.date_range(
        start="2023-01-01",
        periods=number_of_days,
        freq="D"
    )

    day_number = np.arange(number_of_days)

    # Seasonal temperature pattern
    temperature = (
        25
        + 8 * np.sin(2 * np.pi * day_number / 365)
        + np.random.normal(0, 1.5, number_of_days)
    )

    # Product price with small variations
    product_price = (
        100
        + 5 * np.sin(2 * np.pi * day_number / 90)
        + np.random.normal(0, 1, number_of_days)
    )

    # Promotion indicator
    promotion = np.random.binomial(
        1,
        0.25,
        number_of_days
    )

    # Website visits with weekly and seasonal patterns
    website_visits = (
        500
        + 100 * np.sin(2 * np.pi * day_number / 7)
        + 50 * np.sin(2 * np.pi * day_number / 365)
        + 100 * promotion
        + np.random.normal(0, 25, number_of_days)
    )

    # Sales depend on visits, promotion, price, and temperature
    sales = (
        50
        + 0.12 * website_visits
        + 20 * promotion
        - 0.25 * product_price
        + 0.8 * temperature
        + 0.015 * day_number
        + np.random.normal(0, 5, number_of_days)
    )

    sales = np.maximum(sales, 1)

    df = pd.DataFrame({
        "date": dates,
        "temperature": temperature,
        "product_price": product_price,
        "promotion": promotion,
        "website_visits": website_visits,
        "sales": sales
    })

    return df


df = generate_sales_data()

df.to_csv("data/sales_data.csv", index=False)

print("Dataset created successfully.")
print(df.head())
print("\nDataset shape:", df.shape)

# --------------------------------------------------
# 4. Chronological train-test split
# --------------------------------------------------

data = df[FEATURE_COLUMNS].values.astype(np.float32)

split_index = int(len(data) * 0.8)

train_data = data[:split_index]
test_data = data[split_index:]

print("\nTraining rows:", len(train_data))
print("Testing rows:", len(test_data))

# --------------------------------------------------
# 5. Feature scaling
# Fit scaler only on training data
# --------------------------------------------------

scaler = StandardScaler()

train_scaled = scaler.fit_transform(train_data)
all_scaled = scaler.transform(data)

# --------------------------------------------------
# 6. Sliding Window preparation
# --------------------------------------------------

def create_windows(data_array, window_size):

    X = []
    y = []

    for i in range(window_size, len(data_array)):

        X.append(
            data_array[i - window_size:i]
        )

        # Predict all five features at the next time step
        y.append(data_array[i])

    return (
        np.array(X, dtype=np.float32),
        np.array(y, dtype=np.float32)
    )


X_train, y_train = create_windows(
    train_scaled,
    WINDOW_SIZE
)

# Test windows include the last 14 training rows
# so the first test prediction has historical context.
X_test, y_test = create_windows(
    all_scaled[split_index - WINDOW_SIZE:],
    WINDOW_SIZE
)

print("\nSliding Window Results")
print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)

# Expected input shape:
# [number of samples, 14 time steps, 5 features]

# --------------------------------------------------
# 7. Build stacked LSTM model
# --------------------------------------------------

model = Sequential([
    Input(shape=(WINDOW_SIZE, len(FEATURE_COLUMNS))),

    LSTM(
        64,
        return_sequences=True
    ),

    Dropout(0.2),

    LSTM(
        32,
        return_sequences=False
    ),

    Dense(16, activation="relu"),

    # Predict the next value of all five features
    Dense(len(FEATURE_COLUMNS))
])

model.compile(
    optimizer="adam",
    loss="mse",
    metrics=["mae"]
)

model.summary()

# --------------------------------------------------
# 8. Train the LSTM model
# --------------------------------------------------

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

start_time = time.time()

history = model.fit(
    X_train,
    y_train,
    validation_split=0.2,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[early_stopping],
    shuffle=False,
    verbose=1
)

training_time = time.time() - start_time

print(f"\nTraining time: {training_time:.2f} seconds")

# Save model
model.save("models/time_series_lstm.keras")

# Save scaler parameters
np.savez(
    "models/scaler_parameters.npz",
    mean=scaler.mean_,
    scale=scaler.scale_
)

# --------------------------------------------------
# 9. One-step test predictions
# --------------------------------------------------

predictions_scaled = model.predict(
    X_test,
    verbose=0
)

# Convert all predicted features back to original scale
predictions = scaler.inverse_transform(
    predictions_scaled
)

actual_values = scaler.inverse_transform(
    y_test
)

# Evaluate sales only
actual_sales = actual_values[:, TARGET_INDEX]
predicted_sales = predictions[:, TARGET_INDEX]

mae = mean_absolute_error(
    actual_sales,
    predicted_sales
)

rmse = np.sqrt(
    mean_squared_error(
        actual_sales,
        predicted_sales
    )
)

print("\nONE-STEP TEST EVALUATION")
print(f"Sales MAE: {mae:.4f}")
print(f"Sales RMSE: {rmse:.4f}")

# --------------------------------------------------
# 10. Seven-day iterative forecasting
# --------------------------------------------------

# Start with the last 14 known training observations
current_window = train_scaled[-WINDOW_SIZE:].copy()

future_predictions_scaled = []

for day in range(FORECAST_DAYS):

    # Model expects 3D input
    input_window = current_window.reshape(
        1,
        WINDOW_SIZE,
        len(FEATURE_COLUMNS)
    )

    # Predict the next row of five features
    next_row_scaled = model.predict(
        input_window,
        verbose=0
    )[0]

    future_predictions_scaled.append(next_row_scaled)

    # Slide the window forward
    current_window = np.vstack([
        current_window[1:],
        next_row_scaled
    ])

future_predictions_scaled = np.array(
    future_predictions_scaled
)

# Inverse transform all five predicted features
future_predictions = scaler.inverse_transform(
    future_predictions_scaled
)

# Actual first seven days after training split
actual_next_seven = data[
    split_index:split_index + FORECAST_DAYS
]

forecast_dates = df["date"].iloc[
    split_index:split_index + FORECAST_DAYS
].values

forecast_df = pd.DataFrame(
    future_predictions,
    columns=FEATURE_COLUMNS
)

forecast_df.insert(
    0,
    "date",
    forecast_dates
)

forecast_df["actual_sales"] = actual_next_seven[:, TARGET_INDEX]

forecast_df.to_csv(
    "seven_day_forecast.csv",
    index=False
)

print("\nSEVEN-DAY SALES FORECAST")
print(
    forecast_df[
        ["date", "sales", "actual_sales"]
    ].to_string(index=False)
)

# --------------------------------------------------
# 11. Evaluate seven-day forecast
# --------------------------------------------------

seven_day_mae = mean_absolute_error(
    actual_next_seven[:, TARGET_INDEX],
    future_predictions[:, TARGET_INDEX]
)

seven_day_rmse = np.sqrt(
    mean_squared_error(
        actual_next_seven[:, TARGET_INDEX],
        future_predictions[:, TARGET_INDEX]
    )
)

print("\nSEVEN-DAY FORECAST EVALUATION")
print(f"7-day MAE: {seven_day_mae:.4f}")
print(f"7-day RMSE: {seven_day_rmse:.4f}")

# --------------------------------------------------
# 12. Plot training and validation loss
# --------------------------------------------------

plt.figure(figsize=(10, 5))

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.title("LSTM Training and Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Mean Squared Error")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig(
    "plots/training_loss.png"
)

plt.close()

# --------------------------------------------------
# 13. Plot actual vs predicted sales
# --------------------------------------------------

plt.figure(figsize=(12, 5))

plt.plot(
    actual_sales,
    label="Actual Sales"
)

plt.plot(
    predicted_sales,
    label="Predicted Sales",
    alpha=0.8
)

plt.title("Actual vs Predicted Sales")
plt.xlabel("Test Sample")
plt.ylabel("Sales")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig(
    "plots/actual_vs_predicted.png"
)

plt.close()

# --------------------------------------------------
# 14. Plot seven-day forecast
# --------------------------------------------------

plt.figure(figsize=(10, 5))

plt.plot(
    forecast_dates,
    actual_next_seven[:, TARGET_INDEX],
    marker="o",
    label="Actual Sales"
)

plt.plot(
    forecast_dates,
    future_predictions[:, TARGET_INDEX],
    marker="o",
    linestyle="--",
    label="Forecast Sales"
)

plt.title("Seven-Day Sales Forecast")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.xticks(rotation=30)
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig(
    "plots/seven_day_forecast.png"
)

plt.close()

# --------------------------------------------------
# 15. Save evaluation report
# --------------------------------------------------

with open(
    "evaluation_report.txt",
    "w",
    encoding="utf-8"
) as file:

    file.write("DAY 63 - LSTM TIME SERIES FORECASTING\n")
    file.write("=" * 45 + "\n\n")

    file.write("Dataset: Generated multivariate sales data\n")
    file.write(f"Window size: {WINDOW_SIZE} days\n")
    file.write(f"Forecast horizon: {FORECAST_DAYS} days\n")
    file.write(f"Number of features: {len(FEATURE_COLUMNS)}\n")
    file.write(f"Training time: {training_time:.2f} seconds\n\n")

    file.write("ONE-STEP TEST METRICS\n")
    file.write(f"MAE: {mae:.4f}\n")
    file.write(f"RMSE: {rmse:.4f}\n\n")

    file.write("SEVEN-DAY FORECAST METRICS\n")
    file.write(f"MAE: {seven_day_mae:.4f}\n")
    file.write(f"RMSE: {seven_day_rmse:.4f}\n")

print("\nAll Day 63 experiments completed.")
print("Model saved in models/")
print("Forecast saved in seven_day_forecast.csv")
print("Evaluation saved in evaluation_report.txt")
print("Plots saved in plots/")
