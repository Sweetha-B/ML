import pandas as pd
import numpy as np
import math
from scipy import stats
import matplotlib.pyplot as plt

# Sample Chocolate Dataset
data = {
    'Company': ['Lindt', 'Godiva', 'Ghirardelli', 'Valrhona', 'Amedei', 'Nestle', 'Hershey', 'Green & Black'],
    'Cocoa_Percent': [60, 72, 68, 85, 70, 65, 55, 80],
    'Rating': [2.75, 3.5, 3.25, 4.0, 3.75, 2.5, 2.0, 3.75]
}

df = pd.DataFrame(data)
df['Cocoa_Fraction'] = df['Cocoa_Percent'] / 100

# Step 1: Statistics
mean_rating = np.mean(df['Rating'])
std_rating = np.std(df['Rating'])
mode_rating = stats.mode(df['Rating'], keepdims=True).mode[0]

# Step 2: Classify as Above Average
df['Above_Avg'] = df['Rating'].apply(lambda x: 1 if x >= mean_rating else 0)

# Step 3: Correlation
correlation = np.corrcoef(df['Cocoa_Fraction'], df['Rating'])[0, 1]

# Step 4: Prediction Function
def predict_rating(cocoa_percent):
    cocoa_fraction = cocoa_percent / 100
    predicted = mean_rating + (cocoa_fraction - np.mean(df['Cocoa_Fraction'])) * correlation
    return "Above Average" if predicted >= mean_rating else "Below Average"

# Example Prediction
test_cocoa = 78
prediction = predict_rating(test_cocoa)
print(f"Prediction for chocolate with {test_cocoa}% cocoa: {prediction}\n")

# Step 5: Print Statistics
print(f"Mean Rating: {mean_rating:.2f}")
print(f"Standard Deviation: {std_rating:.2f}")
print(f"Most Frequent Rating: {mode_rating}")
print(f"Correlation (Cocoa % vs Rating): {correlation:.2f}")

# Step 6: Final Dataset
print("\nFinal DataFrame:\n")
print(df)

# ------------------- Graphs -------------------



# Scatter Plot: Cocoa % vs Rating
plt.figure(figsize=(6, 4))
plt.scatter(df['Cocoa_Percent'], df['Rating'], c=df['Above_Avg'], cmap='coolwarm', s=80, edgecolors='black')
plt.xlabel('Cocoa Percentage')
plt.ylabel('Rating')
plt.title('Cocoa % vs Chocolate Rating')
plt.grid(True)
plt.tight_layout()
plt.show()


