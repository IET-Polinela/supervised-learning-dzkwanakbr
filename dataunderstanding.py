import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
file_path = "train.csv"
df = pd.read_csv(file_path)

# Hitung statistik deskriptif
stats = df.describe(percentiles=[0.25, 0.5, 0.75]).T
stats = stats.rename(columns={"50%": "Q2 (Median)", "25%": "Q1 (25%)", "75%": "Q3 (75%)"})
stats["count"] = df.count()

# Visualisasi distribusi fitur numerik (boxplot)
plt.figure(figsize=(12, 6))
sns.boxplot(data=df.select_dtypes(include=['number']))
plt.xticks(rotation=45)
plt.title("Boxplot Fitur Numerik")
plt.savefig("boxplot_features.png")  # Simpan sebagai PNG
plt.close()

# Visualisasi distribusi mean, median, dan standar deviasi
plt.figure(figsize=(12, 6))
stats[['mean', 'Q2 (Median)', 'std']].plot(kind='bar', figsize=(12, 6))
plt.title("Statistik Deskriptif (Mean, Median, Std Dev)")
plt.ylabel("Nilai")
plt.xticks(rotation=45)
plt.legend()
plt.savefig("statistik_deskriptif.png")  # Simpan sebagai PNG
plt.close()

# Simpan hasil statistik deskriptif ke CSV
stats.to_csv("statistik_deskriptif.csv")

print("Visualisasi dan hasil statistik telah disimpan sebagai PNG dan CSV.")

