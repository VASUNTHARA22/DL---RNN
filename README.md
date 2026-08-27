# DL- Developing a Recurrent Neural Network Model for Stock Prediction

## AIM
To develop a Recurrent Neural Network (RNN) model for predicting stock prices using historical closing price data.

## Problem Statement and Dataset

## DESIGN STEPS

STEP 1:

Collect historical stock price data and select the Closing Price column for prediction.

STEP 2:

Preprocess the data by handling missing values and normalize the closing prices using MinMaxScaler.

STEP 3:

Create input-output sequences using a fixed time window, where previous stock prices are used to predict the next day's closing price.

STEP 4:

Split the sequential data into training and testing datasets, maintaining the chronological order of the stock data.

STEP 5:

Build and train an RNN model using recurrent layers followed by a Dense output layer to predict the next closing price.

STEP 6:

Evaluate the model on the test data, inverse-transform the predictions to the original price scale, and visualize the actual vs. predicted stock prices.
## PROGRAM

### Name: S VASUNTHARA SAI 

### Register Number: 212224230297

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Step 1: Load and Preprocess Data
df_train = pd.read_csv("trainset.csv")
df_test = pd.read_csv("testset.csv")

train_prices = df_train["Close"].values.reshape(-1, 1)
test_prices = df_test["Close"].values.reshape(-1, 1)

scaler = MinMaxScaler()
scaled_train = scaler.fit_transform(train_prices)
scaled_test = scaler.transform(test_prices)

def create_sequences(data, seq_length):
    x = []
    y = []
    for i in range(len(data) - seq_length):
        x.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    return np.array(x), np.array(y)

seq_length = 60

x_train, y_train = create_sequences(scaled_train, seq_length)
x_test, y_test = create_sequences(scaled_test, seq_length)

print("x_train shape:", x_train.shape)
print("y_train shape:", y_train.shape)
print("x_test shape:", x_test.shape)
print("y_test shape:", y_test.shape)

# Convert to PyTorch tensors
x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
x_test_tensor = torch.tensor(x_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

# Create Dataset and DataLoader
train_dataset = TensorDataset(x_train_tensor, y_train_tensor)

train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True
)

# Step 2: Define RNN Model
class RNNModel(nn.Module):
    def __init__(
        self,
        input_size=1,
        hidden_size=64,
        num_layers=2,
        output_size=1
    ):
        super(RNNModel, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(
            self.num_layers,
            x.size(0),
            self.hidden_size
        ).to(x.device)

        out, _ = self.rnn(x, h0)

        out = out[:, -1, :]

        out = self.fc(out)

        return out


model = RNNModel()

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model = model.to(device)

print("Device:", device)
print(model)

# Model Summary
try:
    from torchinfo import summary
    summary(model, input_size=(64, 60, 1))
except ImportError:
    print("torchinfo not installed. Run: pip install torchinfo")

# Loss Function and Optimizer
criterion = nn.MSELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

# Step 3: Train the Model
num_epochs = 30
train_losses = []

for epoch in range(num_epochs):

    model.train()

    running_loss = 0.0

    for batch_x, batch_y in train_loader:

        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        optimizer.zero_grad()

        outputs = model(batch_x)

        loss = criterion(outputs, batch_y)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

    epoch_loss = running_loss / len(train_loader)

    train_losses.append(epoch_loss)

    print(
        f"Epoch [{epoch + 1}/{num_epochs}], "
        f"Loss: {epoch_loss:.6f}"
    )

print()
plt.figure(figsize=(10, 5))

plt.plot(
    train_losses,
    label="Training Loss"
)

plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("Training Loss Over Epochs")
plt.legend()
plt.show()
print()

# Step 4: Make Predictions on Test Set
model.eval()

with torch.no_grad():

    predicted = model(
        x_test_tensor.to(device)
    ).cpu().numpy()

    actual = y_test_tensor.cpu().numpy()

# Inverse Transform
predicted_prices = scaler.inverse_transform(predicted)
actual_prices = scaler.inverse_transform(actual)

plt.figure(figsize=(10, 6))

plt.plot(
    actual_prices,
    label="Actual Price"
)

plt.plot(
    predicted_prices,
    label="Predicted Price"
)

plt.xlabel("Time")
plt.ylabel("Price")
plt.title("Stock Price Prediction using RNN")
plt.legend()
plt.show()
print()

print(
    f"Predicted Price: {predicted_prices[-1][0]:.2f}"
)

print(
    f"Actual Price: {actual_prices[-1][0]:.2f}"
)

```

### OUTPUT

## Training Loss Over Epochs Plot

<img width="855" height="470" alt="image" src="https://github.com/user-attachments/assets/32f8213a-ac45-4d4a-b08e-a377231dc0a3" />


## True Stock Price, Predicted Stock Price vs time

<img width="859" height="547" alt="image" src="https://github.com/user-attachments/assets/64247dcc-309e-4587-9591-cda7454aaf23" />


### Predictions

<img width="332" height="60" alt="image" src="https://github.com/user-attachments/assets/f307422f-a115-4134-8a2e-4c12c36b3a5b" />


## RESULT

Thus, the RNN model was successfully implemented for stock price prediction using historical closing price data.
