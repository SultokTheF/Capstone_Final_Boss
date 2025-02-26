import numpy as np
import pandas as pd


# Confusion matrix from the image
conf_matrix = np.array([[30, 20, 10],  # Predicted as class 'a'
                        [50, 60, 10],  # Predicted as class 'b'
                        [20, 20, 80]]) # Predicted as class 'c'

# Total correct predictions (diagonal elements)
correct_predictions = np.trace(conf_matrix)

# Total number of predictions
total_predictions = np.sum(conf_matrix)

# Accuracy calculation
accuracy = correct_predictions / total_predictions

# Precision, Recall, and F1-score calculation for each class
precision = []
recall = []
f1_score = []

for i in range(3):  # Three classes (a, b, c)
    TP = conf_matrix[i, i]  # True Positives
    FP = np.sum(conf_matrix[i, :]) - TP  # False Positives
    FN = np.sum(conf_matrix[:, i]) - TP  # False Negatives

    prec = TP / (TP + FP) if (TP + FP) > 0 else 0
    rec = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0

    precision.append(round(prec, 3))
    recall.append(round(rec, 3))
    f1_score.append(round(f1, 3))

# Creating a DataFrame for display
results_df = pd.DataFrame({
    'Class': ['a', 'b', 'c'],
    'Precision': precision,
    'Recall': recall,
    'F1-score': f1_score
})

# Display accuracy and results table
print(results_df)

#   Class  Precision  Recall  F1-score
# 0     a      0.500     0.3     0.375
# 1     b      0.500     0.6     0.545
# 2     c      0.667     0.8     0.727
