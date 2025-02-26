import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Загрузка данных
file_path = r"C:\Users\sushs\OneDrive\Рабочий стол\caps-final\data\Supplementary Materials (7-variant)-20250226\Supplementary Materials (7-variant)-20250226\Question2_Dataset.csv"
df = pd.read_csv(file_path)

# Извлечение признаков и целевой переменной
X = df[['X1', 'X2', 'X1^2', 'X1^3', 'X2^2', 'X2^3', 'X1*X2', 'X1^2*X2']].values
y = df['Y'].values.reshape(-1, 1)

# Нормализация признаков
scaler = StandardScaler()
X_normalized = scaler.fit_transform(X)

# Добавление столбца единиц для bias
X_bias = np.c_[np.ones(X_normalized.shape[0]), X_normalized]

# Функция для вычисления функции стоимости
def compute_cost(X, y, theta):
    m = len(y)
    predictions = X.dot(theta)
    cost = (1 / (2 * m)) * np.sum((predictions - y) ** 2)
    return cost

# Функция для градиентного спуска
def gradient_descent(X, y, theta, alpha, iterations):
    m = len(y)
    cost_history = []
    for i in range(iterations):
        predictions = X.dot(theta)
        error = predictions - y
        theta -= (alpha / m) * (X.T.dot(error))
        cost_history.append(compute_cost(X, y, theta))
    return theta, cost_history

# Параметры
alpha = 0.1  # Learning rate (из вопроса)
theta = np.zeros((X_bias.shape[1], 1))

# Запуск градиентного спуска для разных значений n
results = {}
for n in [10, 100, 1000]:
    theta_final, cost_history = gradient_descent(X_bias, y, theta, alpha, n)
    final_cost = cost_history[-1]
    results[n] = (final_cost, theta_final)

# Вывод результатов
print("{:<12} {:<20} {:<30}".format("# iterations", "Cost Function", "Optimal Theta (Max)"))
print("-" * 70)
for n, (cost, theta_vals) in results.items():
    cost_rounded = round(cost)  # Округление стоимости до целого
    theta_max = round(np.max(theta_vals))  # Максимальное значение theta, округленное до целого
    print(f"n = {n:<10} {cost_rounded:<20} {theta_max:<30}")

# # iterations Cost Function        Optimal Theta (Max)
# ----------------------------------------------------------------------
# n = 10         895241               3328
# n = 100        36623                3328
# n = 1000       1232                 3328
