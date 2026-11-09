# Day 63 Assignment — Multi-Step Time Series Forecasting

## Objective

To build an LSTM-based time series forecasting model and implement multi-step ahead prediction using the sliding window technique.

## Dataset

A generated multivariate daily sales dataset containing:

- Temperature
- Product Price
- Promotion
- Website Visits
- Sales

## Data Preparation

- Applied StandardScaler to normalize the features.
- Used the first 80% of observations for training and the remaining 20% for testing.
- Created sliding windows using 14 previous time steps.
- Prepared LSTM input in three-dimensional format: samples, time steps, and features.

## Model Architecture

- Input layer
- LSTM layer with 64 units and `return_sequences=True`
- Dropout layer
- LSTM layer with 32 units
- Dense layer with 16 neurons
- Output layer predicting five features

## Multi-Step Forecasting

Implemented iterative forecasting to predict the next seven days.

For each prediction:

1. Pass the latest 14-day window into the trained model.
2. Predict the next time step.
3. Append the predicted values to the window.
4. Remove the oldest time step.
5. Repeat until seven future values are generated.

## Evaluation Metrics

Used:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)

Evaluated both one-step test predictions and the seven-day recursive forecast using the original sales scale.

## Output

- Trained LSTM model
- Seven-day forecast CSV
- Training loss plot
- Actual-versus-predicted plot
- Seven-day forecast visualization
- Evaluation report

## Conclusion

This project demonstrates how LSTM networks can learn patterns from sequential data using sliding windows. It also demonstrates multivariate forecasting, feature scaling, inverse transformation, and iterative multi-step prediction.

_Note: Actual MAE and RMSE values should be taken from the generated evaluation report after running the project._
