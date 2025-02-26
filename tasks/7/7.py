import pandas as pd
import numpy as np

# Load the dataset
file_path = r"C:\Users\sushs\OneDrive\Рабочий стол\caps-final\data\Supplementary Materials (7-variant)-20250226\Supplementary Materials (7-variant)-20250226\Question7_Final_CP.csv"
df = pd.read_csv(file_path)

# Given parameters for age groups
category_specific_params_age = {
    "18-24": {"rho": 0.020, "m": 5},
    "55-64": {"rho": 0.018, "m": 5},
}

# Compute statistics
results = []

for age_group, params in category_specific_params_age.items():
    n = df[df["Age Group"] == age_group].shape[0]  # Count of occurrences
    p = n / df.shape[0]  # Estimated proportion

    # Standard error (SE)
    SE = np.sqrt((p * (1 - p)) / n)

    # Confidence Interval (95% CI)
    Z = 1.96  # For 95% confidence level
    CI_lower = p - Z * SE
    CI_upper = p + Z * SE

    # Design Effect (DEFF)
    rho = params["rho"]
    m = params["m"]
    DEFF = 1 + (m - 1) * rho

    # Adjusted Standard Error (SE Adjusted)
    SE_adjusted = SE * np.sqrt(DEFF)

    # Adjusted Confidence Interval (95% CI Adjusted)
    CI_adj_lower = p - Z * SE_adjusted
    CI_adj_upper = p + Z * SE_adjusted

    results.append({
        "Age Group": age_group,
        "n": n,
        "Estimated Proportion": round(p, 3),
        "Standard Error": round(SE, 3),
        "Adjusted SE (Clustering)": round(SE_adjusted, 3),
        "95% CI Lower": round(CI_lower, 3),
        "95% CI Upper": round(CI_upper, 3),
        "95% CI Adjusted Lower": round(CI_adj_lower, 3),
        "95% CI Adjusted Upper": round(CI_adj_upper, 3),
        "Design Effect": round(DEFF, 3)
    })

# Convert results to DataFrame
results_df = pd.DataFrame(results)

# Display results
print(results_df)

# Age Group    n  Estimated Proportion  Standard Error  Adjusted SE (Clustering)  95% CI Lower  95% CI Upper  95% CI Adjusted Lower  95% CI Adjusted Upper  Design Effect
# 0     18-24  227                 0.151           0.024                     0.025         0.105         0.198                  0.103                   0.20          1.080
# 1     55-64  212                 0.141           0.024                     0.025         0.094         0.188                  0.093                   0.19          1.072
