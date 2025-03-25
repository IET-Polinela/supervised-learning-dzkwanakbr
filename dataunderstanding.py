import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("train.csv")

# Hitung statistik deskriptif
stats = df.describe(percentiles=[0.25, 0.5, 0.75]).T
stats = stats.rename(columns={"50%": "Q2 (Median)", "25%": "Q1 (25%)", "75%": "Q3 (75%)"})
stats["count"] = df.count()

# Visualisasi distribusi fitur numerik
plt.figure(figsize=(12, 6))
sns.boxplot(data=df.select_dtypes(include=['number']))
plt.xticks(rotation=45)
plt.title("Boxplot Fitur Numerik")
plt.show()

# Visualisasi distribusi mean, median, dan standar deviasi
plt.figure(figsize=(12, 6))
stats[['mean', 'Q2 (Median)', 'std']].plot(kind='bar', figsize=(12, 6))
plt.title("Statistik Deskriptif (Mean, Median, Std Dev)")
plt.ylabel("Nilai")
plt.xticks(rotation=45)
plt.legend()
plt.show()

# Tampilkan hasil statistik
print(stats)
