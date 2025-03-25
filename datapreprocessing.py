import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load dataset
file_path = "train.csv"
df = pd.read_csv(file_path)

# Identifikasi fitur numerik dan nonnumerik
numerical_features = df.select_dtypes(include=["int64", "float64"]).columns
categorical_features = df.select_dtypes(include=["object"]).columns

# 1️⃣ Encoding fitur nonnumerik dengan LabelEncoder
df_encoded = df.copy()
label_encoders = {}

for col in categorical_features:
    le = LabelEncoder()
    df_encoded[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le  # Simpan encoder untuk referensi jika diperlukan

# 2️⃣ Pisahkan fitur independent (X) dan target (Y)
X = df_encoded.drop(columns=["SalePrice", "Id"])  # Hapus "Id" karena bukan fitur penting
Y = df_encoded["SalePrice"]

# 3️⃣ Membagi dataset menjadi training dan testing (80:20)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# 4️⃣ Gabungkan kembali training & testing ke dalam satu dataset dengan label "Train" dan "Test"
df_encoded["DataType"] = "Train"
df_encoded.loc[X_test.index, "DataType"] = "Test"

# 5️⃣ Simpan dataset hasil preprocessing
df_encoded.to_csv("data_encode.csv", index=False)

# 🔍 Menampilkan informasi dataset setelah preprocessing
print("📌 Dataset setelah preprocessing disimpan sebagai 'data_encode.csv'")
print(f"Total data: {df_encoded.shape}")
print(f"Fitur numerik: {len(numerical_features)} | Fitur kategori: {len(categorical_features)}")
print(f"Training Data: {X_train.shape} | Testing Data: {X_test.shape}")

# Tampilkan beberapa data pertama
print("\n📊 Contoh data setelah preprocessing:")
print(df_encoded.head())
