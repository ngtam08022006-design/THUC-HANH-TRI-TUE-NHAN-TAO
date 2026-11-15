# ==============================================================================
# CHƯƠNG TRÌNH TỔNG HỢP: XÂY DỰNG & ĐÁNH GIÁ MÔ HÌNH PHÂN LOẠI (IMBALANCED DATA)
# ==============================================================================
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    PrecisionRecallDisplay
)

from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline

RANDOM_STATE = 42

# ==============================================================================
# PHẦN 0: NẠP DỮ LIỆU & HÀM TẠO TẬP DỮ LIỆU MẤT CÂN BẰNG
# ==============================================================================
cancer = load_breast_cancer(as_frame=True)
X_full = cancer.data.copy()
y_full = (cancer.target == 0).astype(int)  # 0: Benign (lành tính), 1: Malignant (ác tính)[cite: 2]

df_full = X_full.copy()
df_full["target"] = y_full


def make_imbalanced_df(n_malignant=60):
    """Giữ nguyên toàn bộ lớp Benign (357 mẫu) và lấy n_malignant mẫu lớp Malignant."""
    df_benign = df_full[df_full["target"] == 0]
    df_malig = df_full[df_full["target"] == 1]
    if n_malignant is not None:
        df_malig = df_malig.sample(n=n_malignant, random_state=RANDOM_STATE)
    return (
        pd.concat([df_benign, df_malig])
        .sample(frac=1, random_state=RANDOM_STATE)
        .reset_index(drop=True)
    )


# Bộ dữ liệu mặc định cho thực hành (Benign=357, Malignant=60)
df = make_imbalanced_df(n_malignant=60)

# Bộ độ đo đánh giá chuẩn
SCORING_FULL = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc",
    "pr_auc": "average_precision"
}


# ==============================================================================
# PHẦN 1: QUY TRÌNH THỰC HÀNH CƠ BẢN (EDA -> TRAIN/TEST -> EVALUATION)
# ==============================================================================
def run_main_workflow(df_input):
    print("=" * 80)
    print("PHẦN 1: QUY TRÌNH THỰC HÀNH CƠ BẢN (EDA & MÔ HÌNH MẪU)")
    print("=" * 80)

    # 1. Kiểm tra cấu trúc & phân bố lớp
    print(
        f"Kích thước dữ liệu: {df_input.shape} | Giá trị thiếu: {df_input.isna().sum().sum()} | Trùng lặp: {df_input.duplicated().sum()}")

    class_count = df_input["target"].value_counts().sort_index()
    class_percent = df_input["target"].value_counts(normalize=True).sort_index() * 100
    class_dist = pd.DataFrame({"Số mẫu": class_count, "Tỷ lệ (%)": class_percent.round(2)})
    class_dist.index = ["Benign (0)", "Malignant (1)"]
    display(class_dist)

    # Trực quan hóa phân bố lớp
    plt.figure(figsize=(5, 3.5))
    class_count.plot(kind="bar", color=["steelblue", "salmon"])
    plt.title("Phân bố hai lớp (Dữ liệu thực hành)")
    plt.xlabel("Target")
    plt.ylabel("Số mẫu")
    plt.xticks(ticks=[0, 1], labels=["Benign (0)", "Malignant (1)"], rotation=0)
    plt.tight_layout()
    plt.show()

    # 2. Tách Train/Test (giữ nguyên tỷ lệ lớp)
    X = df_input.drop(columns="target")
    y = df_input["target"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    # 3. Đánh giá các pipeline cơ bản + sampling trên tập Train bằng Cross-Validation
    pipelines = {
        "Baseline (Scaled)": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=5000, random_state=RANDOM_STATE))
        ]),
        "Random Over-Sampling": ImbPipeline([
            ("scaler", StandardScaler()),
            ("sampler", RandomOverSampler(random_state=RANDOM_STATE)),
    ("model", LogisticRegression(max_iter=5000, random_state=RANDOM_STATE))
    ]),
    "Random Under-Sampling": ImbPipeline([
        ("scaler", StandardScaler()),
        ("sampler", RandomUnderSampler(random_state=RANDOM_STATE)),
    ("model", LogisticRegression(max_iter=5000, random_state=RANDOM_STATE))
    ]),
    "SMOTE": ImbPipeline([
        ("scaler", StandardScaler()),
        ("sampler", SMOTE(random_state=RANDOM_STATE)),
    ("model", LogisticRegression(max_iter=5000, random_state=RANDOM_STATE))
    ])
    }

    cv_rows = []
    for name, pipe in pipelines.items():
        cv_res = cross_validate(pipe, X_train, y_train, cv=skf, scoring=SCORING_FULL)
        row = {"Model": name}
        for m in SCORING_FULL:
            row[f"{m}_mean"] = cv_res[f"test_{m}"].mean()
        cv_rows.append(row)

    sorted_comp = pd.DataFrame(cv_rows).set_index("Model").sort_values("f1_mean", ascending=False)
    print("\nBảng tổng hợp Cross-Validation (sắp xếp theo F1_mean):")
    display(sorted_comp.round(4))

    # 4. Đánh giá mô hình tốt nhất trên tập Test
    best_name = sorted_comp.index[0]
    best_model = pipelines[best_name]
    best_model.fit(X_train, y_train)
    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)[:, 1]

    print(f"\nClassification Report trên tập Test ({best_name}):")
    print(
        classification_report(y_test, y_pred, target_names=["Benign (0)", "Malignant (1)"], digits=4, zero_division=0))


    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred, display_labels=["Benign (0)", "Malignant (1)"], ax=axes[0]
    )
    axes[0].set_title(f"Confusion Matrix - {best_name}")

    PrecisionRecallDisplay.from_predictions(y_test, y_prob, ax=axes[1])
    axes[1].set_title(f"Precision-Recall Curve - {best_name}")
    plt.tight_layout()
    plt.show()

    return X_train, X_test, y_train, y_test


# ==============================================================================
# PHẦN 2: GIẢI CHI TIẾT 8 BÀI TẬP NHỎ (BÀI 1 -> BÀI 7)
# ==============================================================================
def run_all_exercises(X_train, y_train):
    skf5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    base_lr = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=5000, random_state=RANDOM_STATE))
    ])

    # --- BÀI 1: MỨC ĐỘ IMBALANCE ---
    print("\n" + "=" * 80)
    print("BÀI 1 — SO SÁNH MỨC ĐỘ IMBALANCE (40, 60, 120, TOÀN BỘ DỮ LIỆU GỐC)")
    print("=" * 80)
    configs_b1 = {
        "Malignant = 40": 40,
        "Malignant = 60 (Mặc định)": 60,
        "Malignant = 120": 120,
        "Toàn bộ dữ liệu gốc (212)": None
    }
    res_b1 = []
    for label, n_mal in configs_b1.items():
        df_tmp = make_imbalanced_df(n_mal)
        X_tmp, y_tmp = df_tmp.drop(columns="target"), df_tmp["target"]
        X_tr, _, y_tr, _ = train_test_split(X_tmp, y_tmp, test_size=0.2, stratify=y_tmp, random_state=RANDOM_STATE)
        cv = cross_validate(base_lr, X_tr, y_tr, cv=skf5, scoring=SCORING_FULL)
        res_b1.append({
            "Cấu hình dữ liệu": label,
            "Accuracy": cv["test_accuracy"].mean(),
            "Recall": cv["test_recall"].mean(),
            "F1": cv["test_f1"].mean(),
            "PR-AUC": cv["test_pr_auc"].mean()
        })
    display(pd.DataFrame(res_b1).set_index("Cấu hình dữ liệu").round(4))

    # --- BÀI 2: KFOLD VS STRATIFIED KFOLD ---
    print("\n" + "=" * 80)
    print("BÀI 2 — SO SÁNH KFOLD VÀ STRATIFIED KFOLD")
    print("=" * 80)
    cv_methods = {
        "KFold (k=5)": KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
        "StratifiedKFold (k=5)": skf5
    }
    res_b2 = []
    for name, cv_obj in cv_methods.items():
        cv = cross_validate(base_lr, X_train, y_train, cv=cv_obj, scoring=SCORING_FULL)
        res_b2.append({
            "Phương pháp CV": name,
            "Accuracy_mean": cv["test_accuracy"].mean(),
            "Accuracy_std": cv["test_accuracy"].std(),
            "F1_mean": cv["test_f1"].mean(),
            "F1_std": cv["test_f1"].std(),
            "PR_AUC_mean": cv["test_pr_auc"].mean(),
            "PR_AUC_std": cv["test_pr_auc"].std()
        })
    display(pd.DataFrame(res_b2).set_index("Phương pháp CV").round(4))

    # --- BÀI 3: THAY ĐỔI SỐ FOLD (K = 3, 5, 10) ---
    print("\n" + "=" * 80)
    print("BÀI 3 — THAY ĐỔI SỐ FOLD (K = 3, 5, 10)")
    print("=" * 80)
    res_b3 = []
    for k in [3, 5, 10]:
        skf_k = StratifiedKFold(n_splits=k, shuffle=True, random_state=RANDOM_STATE)
        cv = cross_validate(base_lr, X_train, y_train, cv=skf_k, scoring=SCORING_FULL)
        res_b3.append({
            "Số Fold (K)": k,
            "Accuracy_mean": cv["test_accuracy"].mean(),
            "Accuracy_std": cv["test_accuracy"].std(),
            "F1_mean": cv["test_f1"].mean(),
            "F1_std": cv["test_f1"].std(),
            "PR_AUC_mean": cv["test_pr_auc"].mean(),
            "PR_AUC_std": cv["test_pr_auc"].std()
        })
    display(pd.DataFrame(res_b3).set_index("Số Fold (K)").round(4))

    # --- BÀI 4: THAY ĐỔI MỨC OVER-SAMPLING ---
    print("\n" + "=" * 80)
    print("BÀI 4 — THAY ĐỔI MỨC OVER-SAMPLING (0.5, 0.75, 1.0)")
    print("=" * 80)
    res_b4 = []
    for ratio in [0.5, 0.75, 1.0]:
        pipe_ros = ImbPipeline([
            ("scaler", StandardScaler()),
            ("ros", RandomOverSampler(sampling_strategy=ratio, random_state=RANDOM_STATE)),
            ("model", LogisticRegression(max_iter=5000, random_state=RANDOM_STATE))
        ])
        cv = cross_validate(pipe_ros, X_train, y_train, cv=skf5, scoring=SCORING_FULL)
        res_b4.append({
            "sampling_strategy": ratio,
            "Precision": cv["test_precision"].mean(),
            "Recall": cv["test_recall"].mean(),
            "F1": cv["test_f1"].mean(),
            "PR-AUC": cv["test_pr_auc"].mean()
        })
    display(pd.DataFrame(res_b4).set_index("sampling_strategy").round(4))

    # --- BÀI 5: SO SÁNH BA KỸ THUẬT SAMPLING ---
    print("\n" + "=" * 80)
    print("BÀI 5 — SO SÁNH BA KỸ THUẬT SAMPLING")
    print("=" * 80)
    samplers = {
        "Không Sampling (Baseline)": None,
        "Random Over-Sampling": RandomOverSampler(random_state=RANDOM_STATE),
    "Random Under-Sampling": RandomUnderSampler(random_state=RANDOM_STATE),
    "SMOTE": SMOTE(random_state=RANDOM_STATE)
    }
    res_b5 = []
    for name, smp in samplers.items():
        steps = [("scaler", StandardScaler())]
        if smp is not None:
            steps.append(("sampler", smp))
        steps.append(("model", LogisticRegression(max_iter=5000, random_state=RANDOM_STATE)))
        pipe = ImbPipeline(steps)
        cv = cross_validate(pipe, X_train, y_train, cv=skf5, scoring=SCORING_FULL)
        res_b5.append({
            "Kỹ thuật": name,
            "Accuracy": cv["test_accuracy"].mean(),
            "Precision": cv["test_precision"].mean(),
            "Recall": cv["test_recall"].mean(),
            "F1": cv["test_f1"].mean(),
            "PR-AUC": cv["test_pr_auc"].mean()
        })
    display(pd.DataFrame(res_b5).set_index("Kỹ thuật").round(4))

    # --- BÀI 6: THAY CLASSIFIER & ĐỘ NHẠY VỚI STANDARDSCALER ---
    print("\n" + "=" * 80)
    print("BÀI 6 — THAY CLASSIFIER (KNN, SVM, DECISION TREE) & TÁC ĐỘNG CỦA STANDARDSCALER")
    print("=" * 80)
    classifiers = {
        "Logistic Regression": LogisticRegression(max_iter=5000, random_state=RANDOM_STATE),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "SVM": SVC(probability=True, random_state=RANDOM_STATE),
    "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE)
    }
    res_b6 = []
    for clf_name, clf in classifiers.items():
        for use_scaler in [False, True]:
            steps = []
            if use_scaler:
                steps.append(("scaler", StandardScaler()))
            steps.append(("smote", SMOTE(random_state=RANDOM_STATE)))
            steps.append(("model", clf))
            pipe = ImbPipeline(steps)
            cv = cross_validate(pipe, X_train, y_train, cv=skf5, scoring=SCORING_FULL)
            res_b6.append({
                "Classifier": clf_name,
                "StandardScaler": "Có" if use_scaler else "Không",
                "Accuracy": cv["test_accuracy"].mean(),
                "Precision": cv["test_precision"].mean(),
                "Recall": cv["test_recall"].mean(),
                "F1": cv["test_f1"].mean(),
                "PR-AUC": cv["test_pr_auc"].mean()
            })
    display(pd.DataFrame(res_b6).set_index(["Classifier", "StandardScaler"]).round(4))

    # --- BÀI 7: CÂU HỎI SUY LUẬN ---
    print("\n" + "=" * 80)
    print("BÀI 7 — TÓM TẮT CÂU TRẢ LỜI SUY LUẬN")
    print("=" * 80)
    qa_list = [
        ("1. Tại sao không nên sampling tập test?",
         "Tập test đại diện cho phân phối thực tế để đánh giá khả năng tổng quát hóa cuối cùng; sampling tập test làm sai lệch phân phối thật và khiến kết quả đánh giá không còn khách quan[cite: 2]."),
        ("2. Tại sao không nên fit StandardScaler trên toàn bộ dữ liệu?",
         "Gây rò rỉ dữ liệu (Data Leakage) vì thông tin mean/std của tập validation/test lọt vào quá trình huấn luyện, làm kết quả đánh giá bị cao giả tạo."),
        ("3. Khi nào Recall quan trọng hơn Precision?",
         "Khi bỏ sót ca dương tính (False Negative) gây hậu quả nghiêm trọng hơn báo động nhầm (VD: sàng lọc ung thư, phát hiện gian lận)."),
        ("4. Khi nào Precision quan trọng hơn Recall?",
         "Khi báo động nhầm (False Positive) gây thiệt hại hoặc phiền toái lớn (VD: lọc email spam tránh chặn nhầm email công việc)."),
        ("5. Vì sao PR-AUC thường hữu ích với dữ liệu imbalance?",
         "PR-AUC tập trung đánh giá hiệu năng trên lớp thiểu số (Positive) và không bị làm loãng bởi số lượng lớn mẫu True Negative của lớp đa số[cite: 2]."),
        ("6. Vì sao StratifiedKFold phù hợp hơn KFold trong bài này?",
         "Giữ nguyên tỷ lệ phân bố hai lớp ở mọi fold, tránh tình trạng fold có quá ít mẫu lớp thiểu số gây biến động điểm số."),
        ("7. Random Under-Sampling có thể làm mất thông tin như thế nào?",
         "Xóa ngẫu nhiên nhiều mẫu của lớp đa số, vô tình loại bỏ các mẫu mang đặc trưng quan trọng hoặc nằm sát ranh giới phân tách lớp."),
        ("8. SMOTE khác Random Over-Sampling ở điểm nào?",
         "Random Over-Sampling nhân bản y hệt mẫu có sẵn (dễ overfitting), còn SMOTE nội suy giữa các điểm láng giềng gần nhất để tạo mẫu tổng hợp mới.")
    ]
    for q, a in qa_list:
        print(f"- {q}\n  -> {a}")


# ==============================================================================
# CHẠY TOÀN BỘ CHƯƠNG TRÌNH
# ==============================================================================
if __name__ == "__main__":
    X_train_main, X_test_main, y_train_main, y_test_main = run_main_workflow(df)
    run_all_exercises(X_train_main, y_train_main)