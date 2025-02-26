import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy.special import expit as sigmoid

# Load dataset
file_path = r"C:\Users\sushs\OneDrive\Рабочий стол\caps-final\data\Supplementary Materials (7-variant)-20250226\Supplementary Materials (7-variant)-20250226\Question3_Final_CP.csv"
df = pd.read_csv(file_path)

# Extract features and target variable
X = df.iloc[:, :-1].values  # First 3 columns as features
y = df.iloc[:, -1].values.reshape(-1, 1)  # Last column as target

# Normalize features using Z = (X - mu) / std
scaler = StandardScaler()
X_normalized = scaler.fit_transform(X)

# Add bias term (column of ones)
X_bias = np.c_[np.ones(X_normalized.shape[0]), X_normalized]

# Logistic Regression with Regularization
def compute_cost(theta, X, y, lambda_):
    """Compute cost for logistic regression with regularization."""
    m = len(y)
    h = sigmoid(X.dot(theta))
    cost = (-1 / m) * np.sum(y * np.log(h) + (1 - y) * np.log(1 - h))
    reg_term = (lambda_ / (2 * m)) * np.sum(np.square(theta[1:]))
    return cost + reg_term

def gradient_descent(X, y, theta, alpha, lambda_, iterations):
    """Perform gradient descent with regularization."""
    m = len(y)
    cost_history = []

    for _ in range(iterations):
        h = sigmoid(X.dot(theta))
        gradient = (1 / m) * X.T.dot(h - y)
        gradient[1:] += (lambda_ / m) * theta[1:]  # Regularization (excluding bias term)
        theta -= alpha * gradient
        cost_history.append(compute_cost(theta, X, y, lambda_))

    return theta, cost_history

# Run logistic regression for different configurations
configs = [
    {"iterations": 100, "alpha": 0.1, "lambda": 0.1},
    {"iterations": 1000, "alpha": 0.2, "lambda": 1},
    {"iterations": 10000, "alpha": 0.3, "lambda": 10},
]

results = []
for config in configs:
    theta_init = np.zeros((X_bias.shape[1], 1))
    theta_opt, cost_history = gradient_descent(
        X_bias, y, theta_init, config["alpha"], config["lambda"], config["iterations"]
    )
    max_theta = np.max(np.abs(theta_opt))  # Maximum theta value
    final_cost = round(cost_history[-1], 2)
    max_theta_rounded = round(max_theta, 2)
    results.append((final_cost, max_theta_rounded))

# Predict after 10,000 iterations, lambda=10, alpha=0.3
theta_final, _ = gradient_descent(X_bias, y, np.zeros((X_bias.shape[1], 1)), 0.3, 10, 10000)
predictions = sigmoid(X_bias.dot(theta_final)) >= 0.5  # Threshold at 0.5
num_ones = np.sum(predictions[:10])  # Number of ones in first 10 predictions

# Display results
df_results = pd.DataFrame(results, columns=["Cost Function", "Max Theta"], index=[
    "N=100, alpha=0.1, lambda=0.1",
    "N=1000, alpha=0.2, lambda=1",
    "N=10000, alpha=0.3, lambda=10",
])

# Print results
print(df_results)
print("\nNumber of ones in first 10 predictions:", num_ones)


# N=100, alpha=0.1, lambda=0.1            0.28       1.61
# N=1000, alpha=0.2, lambda=1             0.16       4.59
# N=10000, alpha=0.3, lambda=10           0.33       2.02

# Number of ones in first 10 predictions: 6
