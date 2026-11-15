import pandas as pd
import numpy as np
from PIL._imaging import display
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression

# 1. Chuẩn bị dữ liệu và Data Cleaning
df = pd.read_csv(r'C:\Users\ADMIN\Downloads\Buổi 2\kc_house_data.csv')
if 'id' in df.columns and 'date' in df.columns:
    df = df.drop(columns=['id', 'date'])

# 2. Xây dựng Baseline (Không có Feature Engineering/Selection)
X_base = df.drop(columns=['price'])
y_base = df['price']
X_train_base, X_test_base, y_train_base, y_test_base = train_test_split(X_base, y_base, test_size=0.2, random_state=42)

baseline_model = LinearRegression()
baseline_model.fit(X_train_base, y_train_base)
baseline_pred = baseline_model.predict(X_test_base)

baseline_mae = mean_absolute_error(y_test_base, baseline_pred)
baseline_rmse = np.sqrt(mean_squared_error(y_test_base, baseline_pred))
baseline_r2 = r2_score(y_test_base, baseline_pred)

# 3. Feature Engineering & Selection
df_fe = df.copy()
df_fe['house_age'] = 2015 - df_fe['yr_built']
df_fe['total_sqft'] = df_fe['sqft_living'] + df_fe['sqft_basement']

abs_corr = df_fe.corr(numeric_only=True)["price"].drop("price").abs()
top_features = abs_corr.sort_values(ascending=False).head(10).index.tolist()

X_fe = df_fe[top_features]
y_fe = df_fe['price']
X_train_fe, X_test_fe, y_train_fe, y_test_fe = train_test_split(X_fe, y_fe, test_size=0.2, random_state=42)

fe_model = LinearRegression()
fe_model.fit(X_train_fe, y_train_fe)
fe_pred = fe_model.predict(X_test_fe)

fe_mae = mean_absolute_error(y_test_fe, fe_pred)
fe_rmse = np.sqrt(mean_squared_error(y_test_fe, fe_pred))
fe_r2 = r2_score(y_test_fe, fe_pred)

# 4. So sánh kết quả theo cấu trúc của tài liệu
comparison = pd.DataFrame({
    "Model": ["Baseline", "Feature Engineering & Selection"],
    "MAE": [baseline_mae, fe_mae],
    "RMSE": [baseline_rmse, fe_rmse],
    "R2": [baseline_r2, fe_r2]
})
print(comparison)