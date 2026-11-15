import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.datasets import load_diabetes

# 1. Chuẩn bị dữ liệu
diabetes = load_diabetes(as_frame=True)
df = diabetes.data.copy()
df["target"] = diabetes.target.copy()

# 2. Tìm Top 5 features có độ tương quan mạnh nhất với target
abs_corr = df.corr(numeric_only=True)["target"].drop("target").abs()
top_5_features = abs_corr.sort_values(ascending=False).head(5).index.tolist()

# 3. Định nghĩa các bộ feature cần huấn luyện
feature_sets = {
    "1. bmi": ["bmi"],
    "2. bmi + bp": ["bmi", "bp"],
    "3. Top 5 features": top_5_features,
    "4. Toàn bộ 10 features": diabetes.data.columns.tolist()
}

results = []

# 4. Huấn luyện và đánh giá từng bộ feature
for name, features in feature_sets.items():
    X = df[features]
    y = df["target"]

    # Tách dữ liệu Train/Test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Khai báo và huấn luyện mô hình Linear Regression
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Dự đoán
    preds = model.predict(X_test)

    # Tính toán các chỉ số đánh giá
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)

    results.append({
        "Bộ Features": name,
        "MAE": mae,
        "RMSE": rmse,
        "R²": r2
    })

# 5. Hiển thị bảng kết quả so sánh
comparison_df = pd.DataFrame(results)
print(comparison_df)