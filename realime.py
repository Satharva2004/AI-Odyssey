import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import time

# Step 1: Load the synthetic dataset
file_path = "synthetic_log_data.csv"
df = pd.read_csv(file_path)

# Step 2: Convert 'timestamp' to Unix timestamp
df["timestamp"] = pd.to_datetime(df["timestamp"]).astype("int64") // 10**9

# Step 3: Encode categorical variables
df = pd.get_dummies(
    df,
    columns=[
        "source_ip",
        "event_type",
        "message",
        "user_agent",
        "url",
        "response_code",
    ],
    drop_first=True,
)

# Step 4: Separate features and labels
X = df.drop("label", axis=1)
y = df["label"]

# Step 5: Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 6: Train the Isolation Forest model
model = IsolationForest(contamination=0.05, random_state=42)
model.fit(X_scaled)

# Step 7: Simulate real-time log processing
plt.ion()  # Enable interactive mode for live plotting
fig, ax = plt.subplots(figsize=(10, 6))

for i in range(0, len(X_scaled), 100):  # Process in batches of 100 logs
    batch = X_scaled[i : i + 100]
    predictions = model.predict(batch)

    # Identify anomalies
    normal = batch[predictions == 1]
    anomalies = batch[predictions == -1]

    ax.clear()
    ax.scatter(normal[:, 0], normal[:, 1], label="Normal Data", alpha=0.6)
    ax.scatter(
        anomalies[:, 0], anomalies[:, 1], color="red", label="Anomalies", alpha=0.6
    )
    ax.set_title("Real-Time Anomaly Detection")
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    ax.legend()
    plt.pause(0.2)  # Pause for 0.5 seconds to simulate real-time processing

plt.ioff()  # Turn off interactive mode
plt.show()
