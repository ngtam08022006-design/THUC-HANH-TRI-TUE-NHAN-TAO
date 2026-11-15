# ==============================================================================
# BÀI 2: HUẤN LUYỆN MÔ HÌNH PHÂN LOẠI CHO BỘ DỮ LIỆU WINE_QUALITY_MERGED
# ==============================================================================
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    PrecisionRecallDisplay
)

from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline

RANDOM_STATE = 42
DATA_PATH = r"C:\Users\ADMIN\Downloads\Buổi 3\wine_quality_merged.csv"
BINARY_MODE = True  # True: Phân loại nhị phân (Chất lượng cao >= 7 vs Thường < 7)
# False: Phân loại đa lớp theo từng mức điểm quality gốc

# ==============================================================================
# 1. CHUẨN BỊ & TIỀN XỬ LÝ DỮ LIỆU BAN ĐẦU
# ==============================================================================
print("=" * 80)
print("1. CHUẨN BỊ DỮ LIỆU")
print("=" * 80)

df_raw = pd.read_csv(DATA_PATH)
print("Kích thước dữ liệu gốc:", df_raw.shape)
display(df_raw.head())

# Chuẩn hóa tên cột (xóa khoảng trắng thừa nếu có)
df_raw.columns = df_raw.columns.str.strip()

# ==============================================================================
# 2. KHÁM PHÁ DỮ LIỆU (EDA)
# ==============================================================================
print("\n" + "=" * 80)
print("2. KHÁM PHÁ DỮ LIỆU (EDA)")
print("=" * 80)

# Bước 1: Kiểm tra cấu trúc, giá trị thiếu, trùng lặp
print("Số giá trị thiếu trên từng cột:")
print(df_raw.isna().sum()[df_raw.isna().sum() > 0])
print(f"Tổng giá trị thiếu: {df_raw.isna().sum().sum()}")
print(f"Số dòng trùng lặp: {df_raw.duplicated().sum()}")
print("\nKiểu dữ liệu:")
display(df_raw.dtypes.value_counts())

# Xử lý giá trị thiếu (nếu có) và loại bỏ dòng trùng lặp
df = df_raw.dropna().drop_duplicates().reset_index(drop=True)
print(f"Kích thước sau khi làm sạch: {df.shape}")

# Xác định cột mục tiêu (mặc định là 'quality')
target_col = "quality" if "quality" in df.columns else df.columns[-1]

# Chuyển đổi nhãn theo chế độ bài toán
if BINARY_MODE:
    # 1: Chất lượng cao (quality >= 7), 0: Rượu thường (quality < 7)
    df["target"] = (df[target_col] >= 7).astype(int)
    class_names = ["Thường (<7) [0]", "Chất lượng cao (>=7) [1]"]
else:
    df["target"] = df[target_col]
    class_names = [str(c) for c in sorted(df["target"].unique())]

if target_col != "target":
    df = df.drop(columns=[target_col])

# Bước 2: Phân tích phân bố lớp
print("\n--- PHÂN BỐ CÁC LỚP ---")
class_count = df["target"].value_counts().sort_index()
class_percent = df["target"].value_counts(normalize=True).sort_index() * 100
class_dist = pd.DataFrame({
    "Số mẫu": class_count,
    "Tỷ lệ (%)": class_percent.round(2)
})
class_dist.index = class_names
display(class_dist)

plt.figure(figsize=(6, 4))
class_count.plot(kind="bar", color="steelblue")
plt.title("Phân bố lớp mục tiêu (Wine Quality)")
plt.xlabel("Lớp")
plt.ylabel("Số mẫu")
plt.xticks(ticks=range(len(class_names)), labels=class_names, rotation=0 if BINARY_MODE else 45)
plt.tight_layout()
plt.show()

# Bước 3: Kiểm tra thang đo của các đặc trưng số
num_cols = df.drop(columns=["target"]).select_dtypes(include=[np.number]).columns.tolist()
cat_cols = df.drop(columns=["target"]).select_dtypes(exclude=[np.number]).columns.tolist()

print("\n--- THANG ĐO CÁC ĐẶC TRƯNG SỐ (SẮP XẾP THEO ĐỘ LỆCH CHUẨN) ---")
feature_scale = pd.DataFrame({
    "mean": df[num_cols].mean(),
    "std": df[num_cols].std(),
    "min": df[num_cols].min(),
    "max": df[num_cols].max()
}).sort_values("std", ascending=False)
display(feature_scale)

# ==============================================================================
# 3. CHIA DỮ LIỆU & THIẾT LẬP PIPELINE TIỀN XỬ LÝ
# ==============================================================================
print("\n" + "=" * 80)
print("3. CHIA DỮ LIỆU TRAIN / TEST & TIỀN XỬ LÝ")
print("=" * 80)

X = df.drop(columns=["target"])
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)
print(f"X_train: {X_train.shape} | X_test: {X_test.shape}")

# Chuẩn hóa biến số bằng StandardScaler & mã hóa biến phân loại (nếu có cột 'type'/'color')
transformers = [("num", StandardScaler(), num_cols)]
if len(cat_cols) > 0:
    transformers.append(("cat", OneHotEncoder(drop="first", sparse_output=False), cat_cols))

preprocessor = ColumnTransformer(transformers=transformers)

# ==============================================================================
# 4. HUẤN LUYỆN, XỬ LÝ IMBALANCE & SO SÁNH MÔ HÌNH QUA CROSS-VALIDATION
# ==============================================================================
print("\n" + "=" * 80)
print("4. SO SÁNH MÔ HÌNH & KỸ THUẬT SAMPLING (5-FOLD STRATIFIED CV)")
print("=" * 80)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

if BINARY_MODE:
    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc",
        "pr_auc": "average_precision"
    }
    sort_metric = "f1_mean"
else:
    scoring = {
        "accuracy": "accuracy",
        "precision": "precision_weighted",
        "recall": "recall_weighted",
        "f1": "f1_weighted"
    }
    sort_metric = "f1_mean"

# Danh sách các tổ hợp (Thuật toán + Kỹ thuật xử lý mất cân bằng)
experiments = {
    "Logistic Regression (Baseline)": ImbPipeline([
        ("prep", preprocessor),
        ("model", LogisticRegression(max_iter=5000, random_state=RANDOM_STATE))
    ]),
    "Logistic Regression + OverSampling": ImbPipeline([
        ("prep", preprocessor),
        ("sampler", RandomOverSampler(random_state=RANDOM_STATE)),
        ("model", LogisticRegression(max_iter=5000, random_state=RANDOM_STATE))
    ]),
    "Logistic Regression + UnderSampling": ImbPipeline([
        ("prep", preprocessor),
        ("sampler", RandomUnderSampler(random_state=RANDOM_STATE)),
        ("model", LogisticRegression(max_iter=5000, random_state=RANDOM_STATE))
    ]),
    "Logistic Regression + SMOTE": ImbPipeline([
        ("prep", preprocessor),
        ("sampler", SMOTE(random_state=RANDOM_STATE)),
        ("model", LogisticRegression(max_iter=5000, random_state=RANDOM_STATE))
    ]),
    "KNN + SMOTE": ImbPipeline([
        ("prep", preprocessor),
        ("sampler", SMOTE(random_state=RANDOM_STATE)),
        ("model", KNeighborsClassifier(n_neighbors=5))
    ]),
    "SVM + SMOTE": ImbPipeline([
        ("prep", preprocessor),
        ("sampler", SMOTE(random_state=RANDOM_STATE)),
        ("model", SVC(probability=True, random_state=RANDOM_STATE))
    ]),
    "Decision Tree + SMOTE": ImbPipeline([
        ("prep", preprocessor),
        ("sampler", SMOTE(random_state=RANDOM_STATE)),
        ("model", DecisionTreeClassifier(random_state=RANDOM_STATE))
    ]),
    "Random Forest (Baseline)": ImbPipeline([
        ("prep", preprocessor),
        ("model", RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE))
    ]),
    "Random Forest + SMOTE": ImbPipeline([
        ("prep", preprocessor),
        ("sampler", SMOTE(random_state=RANDOM_STATE)),
        ("model", RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE))
    ])
}

results = []
for name, pipe in experiments.items():
    cv_res = cross_validate(pipe, X_train, y_train, cv=skf, scoring=scoring)
    row = {"Model": name}
    for m in scoring.keys():
        row[f"{m}_mean"] = cv_res[f"test_{m}"].mean()
    results.append(row)

cv_comparison = pd.DataFrame(results).set_index("Model").sort_values(sort_metric, ascending=False)
display(cv_comparison.round(4))

# ==============================================================================
# 5. ĐÁNH GIÁ MÔ HÌNH TỐT NHẤT TRÊN TẬP TEST
# ==============================================================================
print("\n" + "=" * 80)
print("5. ĐÁNH GIÁ MÔ HÌNH TỐT NHẤT TRÊN TẬP TEST")
print("=" * 80)

best_name = cv_comparison.index[0]
best_model = experiments[best_name]
print(f"--> Mô hình được chọn (cao nhất theo {sort_metric}): {best_name}\n")

best_model.fit(X_train, y_train)
y_pred = best_model.predict(X_test)

# Báo cáo phân loại chi tiết
print("--- CLASSIFICATION REPORT ---")
print(classification_report(y_test, y_pred, target_names=class_names, digits=4, zero_division=0))

# Trực quan hóa Confusion Matrix (và Precision-Recall Curve nếu là nhị phân)
if BINARY_MODE:
    y_prob = best_model.predict_proba(X_test)[:, 1]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred, display_labels=class_names, ax=axes[0], colorbar=False
    )
    axes[0].set_title(f"Confusion Matrix\n({best_name})")

    PrecisionRecallDisplay.from_predictions(y_test, y_prob, ax=axes[1])
    axes[1].set_title(f"Precision-Recall Curve\n({best_name})")

    plt.tight_layout()
    plt.show()
else:
    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred, display_labels=class_names, ax=ax, colorbar=False
    )
    ax.set_title(f"Confusion Matrix - {best_name}")
    plt.tight_layout()
    plt.show()