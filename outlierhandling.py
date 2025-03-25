import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
file_path = "data_encode.csv"
df = pd.read_csv(file_path)

# Identifikasi fitur numerik
numerical_features = df.select_dtypes(include=["int64", "float64"]).columns

# Visualisasi outlier dengan boxplot dalam satu gambar
plt.figure(figsize=(12, 6))
sns.boxplot(data=df[numerical_features])
plt.xticks(rotation=90)
plt.title("Visualisasi Outlier pada Fitur Numerik")
plt.savefig("outlier_visualization.png")  # Simpan gambar visualisasi
plt.show()

# Menggunakan metode IQR untuk mendeteksi outlier
Q1 = df[numerical_features].quantile(0.25)
Q3 = df[numerical_features].quantile(0.75)
IQR = Q3 - Q1

# Menentukan batas bawah dan atas
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Menandai outlier
outlier_mask = (df[numerical_features] < lower_bound) | (df[numerical_features] > upper_bound)

# Dataset dengan outlier (asli)
df_with_outliers = df.copy()
df_with_outliers.to_csv("data_with_outliers.csv", index=False)

# Dataset tanpa outlier
df_without_outliers = df[~outlier_mask.any(axis=1)]
df_without_outliers.to_csv("data_without_outliers.csv", index=False)

print("✅ Proses selesai. Dataset telah disimpan:")
print("- data_with_outliers.csv (dengan outlier)")
print("- data_without_outliers.csv (tanpa outlier)")
print("- outlier_visualization.png (visualisasi outlier)")

