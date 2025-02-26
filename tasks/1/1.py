# Пересчет вручную без округлений
import numpy as np
import pandas as pd

data = pd.DataFrame({
    "Case": np.arange(1, 17),  # Номера случаев
    "Stratum": [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4],  # Страты
    "Cluster": [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8],  # Кластеры
    "Variable": [11.5, 29, 35.6, 64.7, 19.2, 20.5, 37.1, 92, 88.3, 78.4, 65.3, 55.2, 85.3, 15.7, 44.5, 67.3]  # Значения переменной
})
# 1. Среднее значение (Mean) для SRS
mean_srs = data["Variable"].mean()

# 2. Стандартное отклонение для SRS
std_dev_srs = data["Variable"].std(ddof=1)

# 3. Стандартная ошибка для SRS
n = len(data)
se_srs = std_dev_srs / np.sqrt(n)

# 4. Доверительный интервал (95% CI)
t_value = 2.04
upper_limit_srs = mean_srs + t_value * se_srs
lower_limit_srs = mean_srs - t_value * se_srs

# 5. Среднее значение для кластерной выборки
mean_clustered = data.groupby("Cluster")["Variable"].mean().mean()

# 6. Стандартная ошибка для кластерной выборки
cluster_means = data.groupby("Cluster")["Variable"].mean()
std_dev_clustered = cluster_means.std(ddof=1)
n_clusters = data["Cluster"].nunique()
se_clustered = std_dev_clustered / np.sqrt(n_clusters)

# 7. D-value
d_value = se_clustered / se_srs

# 8. D-squared
d_squared = d_value ** 2

# 9. Коэффициент интра-кластерной корреляции (ρ)
M = data.groupby("Cluster").size().mean()
roh = (d_squared - 1) / (M - 1)

# 10. Эффективный размер выборки (Neff)
N = len(data)
Neff = N / (1 + (M - 1) * roh)

# Вывод результатов
results_precise = pd.DataFrame({
    "Metric": ["Mean (SRS)", "SE (SRS)", "Upper Limit (SRS)", "Lower Limit (SRS)",
               "Mean (Clustered)", "SE (Clustered)", "D-value", "D-squared", "Roh", "Neff"],
    "Value": [mean_srs, se_srs, upper_limit_srs, lower_limit_srs,
              mean_clustered, se_clustered, d_value, d_squared, roh, Neff]
})

print(results_precise)


#               Metric      Value
# 0         Mean (SRS)  50.600000
# 1           SE (SRS)   6.885050
# 2  Upper Limit (SRS)  64.645502
# 3  Lower Limit (SRS)  36.554498
# 4   Mean (Clustered)  50.600000
# 5     SE (Clustered)   7.624297
# 6            D-value   1.107370
# 7          D-squared   1.226268
# 8                Roh   0.226268
# 9               Neff  13.047718