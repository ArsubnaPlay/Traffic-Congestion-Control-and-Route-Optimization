import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

# Load the data
data = pd.read_csv('final_traffic_data.csv')

# Preprocess the data
data['datetime'] = pd.to_datetime(data['datetime'])
data['hour'] = data['datetime'].dt.hour
data['day_of_week'] = data['datetime'].dt.dayofweek

# Encode categorical variables
le_road = LabelEncoder()
le_peak = LabelEncoder()
le_congestion = LabelEncoder()

data['road_segment'] = le_road.fit_transform(data['road_segment'])
data['is_peakhour'] = le_peak.fit_transform(data['is_peakhour'])
data['congestion_index'] = le_congestion.fit_transform(data['congestion_index'])

# Scale numerical features
scaler = MinMaxScaler()
numerical_features = ['speed', 'hour', 'day_of_week']
data[numerical_features] = scaler.fit_transform(data[numerical_features])

# Prepare features and targets
X = data[['road_segment', 'is_peakhour', 'speed', 'hour', 'day_of_week']].values
y_congestion = to_categorical(data['congestion_index'].values)
y_arrival = data['arrival_time'].values

# Split the data
X_train, X_test, y_congestion_train, y_congestion_test, y_arrival_train, y_arrival_test = train_test_split(
    X, y_congestion, y_arrival, test_size=0.2, random_state=42)

# Reshape input for LSTM [samples, time steps, features]
X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

# Define and train the congestion index model
congestion_model = Sequential([
    LSTM(64, input_shape=(1, X_train.shape[2]), return_sequences=True),
    Dropout(0.2),
    LSTM(32),
    Dropout(0.2),
    Dense(y_congestion.shape[1], activation='softmax')
])

congestion_model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
congestion_history = congestion_model.fit(X_train, y_congestion_train, epochs=50, batch_size=32, validation_split=0.2, verbose=1)

# Define and train the arrival time model
arrival_model = Sequential([
    LSTM(64, input_shape=(1, X_train.shape[2]), return_sequences=True),
    Dropout(0.2),
    LSTM(32),
    Dropout(0.2),
    Dense(1)
])

arrival_model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
arrival_history = arrival_model.fit(X_train, y_arrival_train, epochs=50, batch_size=32, validation_split=0.2, verbose=1)


congestion_loss, congestion_accuracy = congestion_model.evaluate(X_test, y_congestion_test)
print(f"Congestion Index Model - Test Loss: {congestion_loss:.4f}, Test Accuracy: {congestion_accuracy:.4f}")

arrival_loss, arrival_mae = arrival_model.evaluate(X_test, y_arrival_test)
print(f"Arrival Time Model - Test Loss: {arrival_loss:.4f}, Test MAE: {arrival_mae:.4f}")