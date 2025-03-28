import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import logging
from datetime import datetime, timedelta

class LSTMNet(nn.Module):
    def __init__(self, input_size=6, hidden_size=50, num_layers=2):
        super(LSTMNet, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.input_size = input_size

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        if x.size(-1) != self.input_size:
            raise ValueError(f"Expected input_size: {self.input_size}, got: {x.size(-1)}")

        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        return out

class LSTMModel:
    def __init__(self, sequence_length=60, n_features=6):
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.device = torch.device('cpu')
        self.model = LSTMNet(input_size=n_features).to(self.device)
        self.scaler = MinMaxScaler()
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.metrics = {}
        logging.info(f"Initialized LSTM model with {n_features} features")

    def prepare_data(self, data):
        """Prepare data for LSTM model"""
        try:
            if data.shape[1] != self.n_features:
                raise ValueError(f"Expected {self.n_features} features, got {data.shape[1]}")

            scaled_data = self.scaler.fit_transform(data)
            X, y = [], []

            for i in range(self.sequence_length, len(scaled_data)):
                X.append(scaled_data[i-self.sequence_length:i])
                y.append(scaled_data[i, 0])

            X = torch.FloatTensor(np.array(X)).to(self.device)
            y = torch.FloatTensor(np.array(y)).to(self.device)

            logging.info(f"Prepared data shapes - X: {X.shape}, y: {y.shape}")
            return X, y

        except Exception as e:
            logging.error(f"Error preparing data: {str(e)}")
            raise

    def calculate_metrics(self, y_true, y_pred):
        """Calculate model performance metrics"""
        try:
            # Convert tensors to numpy arrays
            if torch.is_tensor(y_true):
                y_true = y_true.cpu().numpy()
            if torch.is_tensor(y_pred):
                y_pred = y_pred.cpu().numpy()

            # Calculate metrics
            mae = mean_absolute_error(y_true, y_pred)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            r2 = r2_score(y_true, y_pred)

            # Calculate directional accuracy
            direction_true = np.diff(y_true) > 0
            direction_pred = np.diff(y_pred) > 0
            directional_accuracy = np.mean(direction_true == direction_pred) * 100

            return {
                'MAE': mae,
                'RMSE': rmse,
                'R²': r2,
                'Directional Accuracy': directional_accuracy
            }

        except Exception as e:
            logging.error(f"Error calculating metrics: {str(e)}")
            raise

    def train(self, X_train, y_train, epochs=50, batch_size=32):
        """Train the LSTM model"""
        try:
            self.model.train()
            train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

            for epoch in range(epochs):
                total_loss = 0
                for batch_X, batch_y in train_loader:
                    self.optimizer.zero_grad()
                    outputs = self.model(batch_X)
                    loss = self.criterion(outputs.squeeze(), batch_y)
                    loss.backward()
                    self.optimizer.step()
                    total_loss += loss.item()

                if (epoch + 1) % 10 == 0:
                    logging.info(f'Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(train_loader):.4f}')

            # Calculate final training metrics
            with torch.no_grad():
                y_pred = self.model(X_train).squeeze().cpu().numpy()
                y_true = y_train.cpu().numpy()
                self.metrics = self.calculate_metrics(y_true, y_pred)
                logging.info(f"Training metrics: {self.metrics}")

        except Exception as e:
            logging.error(f"Error training model: {str(e)}")
            raise

    def get_metrics(self):
        """Return current model metrics"""
        return self.metrics

    def predict_future(self, X, last_sequence, periods):
        """Generate predictions for multiple future periods"""
        try:
            self.model.eval()
            predictions = []
            current_sequence = last_sequence.clone()

            for _ in range(periods):
                with torch.no_grad():
                    pred = self.model(current_sequence.unsqueeze(0))
                    predictions.append(pred.item())
                    current_sequence = torch.roll(current_sequence, -1, dims=0)
                    current_sequence[-1, 0] = pred.item()

            dummy_array = np.zeros((len(predictions), self.n_features))
            dummy_array[:, 0] = predictions
            return self.scaler.inverse_transform(dummy_array)[:, 0]

        except Exception as e:
            logging.error(f"Error generating predictions: {str(e)}")
            raise

    def get_price_predictions(self, X, current_price):
        """Get price predictions for different time periods"""
        try:
            last_sequence = X[-1]
            periods = {
                'Next Day': 1,
                '1 Week': 5,
                '1 Month': 21,
                '3 Months': 63,
                '6 Months': 126,
                '1 Year': 252
            }

            predictions = {}
            for period_name, days in periods.items():
                future_prices = self.predict_future(X, last_sequence, days)
                predictions[period_name] = future_prices[-1]

            return predictions

        except Exception as e:
            logging.error(f"Error generating price predictions: {str(e)}")
            raise

    def predict(self, X):
        """Generate predictions"""
        try:
            self.model.eval()
            with torch.no_grad():
                predictions = self.model(torch.FloatTensor(X).to(self.device))
                predictions = predictions.cpu().numpy()

                dummy_array = np.zeros((len(predictions), self.n_features))
                dummy_array[:, 0] = predictions.squeeze()
                return self.scaler.inverse_transform(dummy_array)[:, 0]

        except Exception as e:
            logging.error(f"Error generating predictions: {str(e)}")
            raise

    def generate_signals(self, predictions, actual_prices, threshold=0.01):
        """Generate trading signals based on predictions"""
        signals = []
        for i in range(len(predictions)-1):
            predicted_return = (predictions[i+1] - predictions[i]) / predictions[i]

            if predicted_return > threshold:
                signals.append('BUY')
            elif predicted_return < -threshold:
                signals.append('SELL')
            else:
                signals.append('HOLD')

        signals.append('HOLD')  # For the last point
        return signals