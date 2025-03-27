import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
file_with_outliers = "data_with_outliers.csv"
file_without_outliers = "dataset_clean_proces.csv"

try:
    df_with_outliers = pd.read_csv(file_with_outliers)
    df_without_outliers = pd.read_csv(file_without_outliers)
    print("Dataset berhasil dimuat!")
except Exception as e:
    print(f"Error saat membaca dataset: {e}")
    exit()

# Menghapus missing values
df_with_outliers.dropna(inplace=True)
df_without_outliers.dropna(inplace=True)

# Memastikan semua kolom numerik
df_with_outliers = df_with_outliers.select_dtypes(include=[np.number])
df_without_outliers = df_without_outliers.select_dtypes(include=[np.number])

# Memisahkan fitur dan target
if "SalePrice" in df_with_outliers.columns and "SalePrice" in df_without_outliers.columns:
    X_with_outliers = df_with_outliers.drop(columns=["SalePrice"])
    Y_with_outliers = df_with_outliers["SalePrice"]

    X_without_outliers = df_without_outliers.drop(columns=["SalePrice"])
    Y_without_outliers = df_without_outliers["SalePrice"]
else:
    print("Kolom 'SalePrice' tidak ditemukan dalam dataset.")
    exit()

# Split dataset (80:20)
X_train_wo, X_test_wo, Y_train_wo, Y_test_wo = train_test_split(X_with_outliers, Y_with_outliers, test_size=0.2, random_state=42)
X_train_wo_no, X_test_wo_no, Y_train_wo_no, Y_test_wo_no = train_test_split(X_without_outliers, Y_without_outliers, test_size=0.2, random_state=42)

# Fungsi untuk menerapkan Polynomial Regression
def polynomial_regression(degree, X_train, X_test, Y_train, Y_test, label):
    poly = PolynomialFeatures(degree=degree)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)

    model = LinearRegression()
    model.fit(X_train_poly, Y_train)
    Y_pred = model.predict(X_test_poly)

    mse = mean_squared_error(Y_test, Y_pred)
    r2 = r2_score(Y_test, Y_pred)

    print(f"{label} - Degree {degree} - MSE: {mse:.2f}, R²: {r2:.4f}")

    # Scatter plot untuk melihat hasil prediksi
    plt.figure(figsize=(6, 4))
    plt.scatter(Y_test, Y_pred, alpha=0.5, label=f"Degree {degree}")
    plt.xlabel("Actual SalePrice")
    plt.ylabel("Predicted SalePrice")
    plt.title(f"Polynomial Regression {label} - Degree {degree}")
    plt.legend()

    # Simpan hasil dalam file PNG
    filename = f"polynomial_{label.lower()}_degree_{degree}.png"
    plt.savefig(filename)
    plt.show()

    return mse, r2

# Evaluasi Polynomial Regression
print("\nEvaluasi dengan Outlier:")
mse_wo_d2, r2_wo_d2 = polynomial_regression(2, X_train_wo, X_test_wo, Y_train_wo, Y_test_wo, "With_Outliers")
mse_wo_d3, r2_wo_d3 = polynomial_regression(3, X_train_wo, X_test_wo, Y_train_wo, Y_test_wo, "With_Outliers")

print("\nEvaluasi tanpa Outlier:")
mse_wo_no_d2, r2_wo_no_d2 = polynomial_regression(2, X_train_wo_no, X_test_wo_no, Y_train_wo_no, Y_test_wo_no, "Without_Outliers")
mse_wo_no_d3, r2_wo_no_d3 = polynomial_regression(3, X_train_wo_no, X_test_wo_no, Y_train_wo_no, Y_test_wo_no, "Without_Outliers")

