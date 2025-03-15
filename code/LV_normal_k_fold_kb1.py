
import pandas as pd
import numpy as np
import time
import os

import matplotlib.pyplot as plt
# %matplotlib inline
import seaborn as sns

from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc





# Lấy đường dẫn đến thư mục hiện tại
data_dir = os.getcwd()

"""### Load data"""

# Loading training dataset
df_train = pd.read_csv(os.path.join(data_dir, 'data_set_ALL_AML_train.csv'))

# Loading testing dataset

df_test = pd.read_csv(os.path.join(data_dir, 'data_set_ALL_AML_independent.csv'))

print("df_train: df_train.shape", df_train.shape)
print("df_test: df_test.shape", df_test.shape)


"""Cleaning data"""

# Remove call columns in training dataset
columns_to_remove_train = [col for col in df_train if 'call' in col]
train = df_train.drop(columns_to_remove_train, axis=1)

# Remove call columns in testing dataset
columns_to_remove_test = [col for col in df_test if 'call' in col]
test = df_test.drop(columns_to_remove_test, axis=1)

"""Chuyển vị các cột và hàng sao cho 7129 gen trở thành các đặc điểm và mỗi trường hợp bệnh nhân chiếm một hàng duy nhất."""

# Transpose row and columns in training set
X_train = train.T

# Transpose row and columns in testing set
X_test = test.T

# print(X_train.shape)
# X_train.head()
# print(X_test.shape)

# X_train_tr.head(10)

"""Chỉ định hàng thứ hai chứa số lượng gen gia nhập làm tên cột. Sau đó, xóa cả hai hàng đầu tiên của Mô tả gen và Số gia nhập gen."""

# Đặt hàng thứ 2 (Gene Accession Number) làm tên cột
X_train.columns = X_train.iloc[1]  # for training set
X_test.columns = X_test.iloc[1]  # for testing set

# Bỏ 2 hàng đầu tiên (Gene Description and Gene Accession Number) và lập lại chỉ mục
X_train = X_train.iloc[2:].reset_index(drop=True)  # for training set
X_test = X_test.iloc[2:].reset_index(drop=True)  # for testing set

print("X_train:", X_train.shape)
print("X_test: ", X_test.shape)

X_train.head()

"""Chuyển đổi giá trị dữ liệu thành số."""

# Convert data values to numeric for training set
X_train = X_train.astype(float, 64)
# Convert data values to numeric for testing set
X_test =  X_test.astype(float, 64)

"""Load labels"""

# Load labels
labels = pd.read_csv(os.path.join(data_dir, 'actual.csv'))


# Gộp tất cả các dữ liệu
merged_X = pd.concat([X_train, X_test], ignore_index=True, axis=0)
merged_XY = pd.concat([merged_X, labels], axis=1)

print("merged_X:",merged_X.shape)
print("merged_XY:",merged_XY.shape)


# Check for nulls
null_counts = merged_XY.isnull().sum().max()

print('Columns with Null Values:')
print(null_counts)

# Kiểm tra sự mất cân bằng nhãn
merged_XY['cancer'].value_counts()

"""Có nhiều bệnh nhân TẤT CẢ hơn bệnh nhân AML. Chúng ta cũng hãy chuyển đổi nhãn thành giá trị số với ALL là 0 và AML là 1."""

# Replace values in the 'cancer' column with 0 for 'ALL' and 1 for 'AML'
cancer_mapping = {'ALL': 0, 'AML': 1}
merged_XY['cancer'] = merged_XY['cancer'].map(cancer_mapping)

"""Splitting the data into train and test sets"""

# Chia dữ liệu thành đặc trưng và nhãn
X = merged_XY.drop(columns=['cancer'])  # Loại bỏ cột nhãn
y = merged_XY['cancer']

# Splitting the data into training and testing sets

# Khởi tạo k-fold cross-validation với k=5
k = 5
kf = KFold(n_splits=k, shuffle=True, random_state=42)

# Lưu kết quả đánh giá
results = {'NB': [], 'LR': [], 'SVM': [], 'KNN': [], 'DT': [], 'RF': []}
auc_scores = {'NB': [], 'LR': [], 'SVM': [], 'KNN': [], 'DT': [], 'RF': []}
auc_labels = {'NB': [], 'LR': [], 'SVM': [], 'KNN': [], 'DT': [], 'RF': []}  # Lưu AUC từng nhãn
confusion_matrices = {'NB': [], 'LR': [], 'SVM': [], 'KNN': [], 'DT': [], 'RF': []}
classification_reports = {'NB': [], 'LR': [], 'SVM': [], 'KNN': [], 'DT': [], 'RF': []}

# Khởi tạo các mô hình
models = {
    'NB': GaussianNB(),
    'LR': LogisticRegression(),
    'SVM': SVC(probability=True),
    'KNN': KNeighborsClassifier(),
    'DT': DecisionTreeClassifier(random_state=42),
    'RF': RandomForestClassifier(random_state=42)
}

# Lặp qua từng fold
for train_idx, test_idx in kf.split(X, y):
    X_train_fold, X_test_fold = X.iloc[train_idx], X.iloc[test_idx]
    y_train_fold, y_test_fold = y.iloc[train_idx], y.iloc[test_idx]

    for model_name, model in models.items():
        # Huấn luyện mô hình
        model.fit(X_train_fold, y_train_fold)

        # Dự đoán trên tập huấn luyện
        y_train_pred = model.predict(X_train_fold)
        y_train_prob = model.predict_proba(X_train_fold)[:, 1] if hasattr(model, 'predict_proba') else None

        # Dự đoán trên tập kiểm tra
        y_test_pred = model.predict(X_test_fold)
        y_test_prob = model.predict_proba(X_test_fold) if hasattr(model, 'predict_proba') else None

        # Độ chính xác
        acc_train = accuracy_score(y_train_fold, y_train_pred)
        acc_test = accuracy_score(y_test_fold, y_test_pred)

        # Ma trận nhầm lẫn
        cm_train = confusion_matrix(y_train_fold, y_train_pred)
        cm_test = confusion_matrix(y_test_fold, y_test_pred)

        # Báo cáo phân loại
        report_train = classification_report(y_train_fold, y_train_pred, output_dict=True)
        report_test = classification_report(y_test_fold, y_test_pred, output_dict=True)

        # Tính AUC cho từng nhãn
        auc_per_label = {}
        for label in [0, 1]:  # Nhãn 0 (ALL), 1 (AML)
            fpr, tpr, _ = roc_curve(y_test_fold == label, y_test_prob[:, label]) if y_test_prob is not None else (None, None, None)
            auc_label = auc(fpr, tpr) if fpr is not None else None
            auc_per_label[label] = auc_label

        # Tính AUC trung bình các nhãn
        avg_auc = np.mean([v for v in auc_per_label.values() if v is not None])

        # Lưu kết quả vào các danh sách
        results[model_name].append({'train_accuracy': acc_train, 'test_accuracy': acc_test})
        auc_scores[model_name].append(avg_auc)
        auc_labels[model_name].append(auc_per_label)
        confusion_matrices[model_name].append({'train_cm': cm_train, 'test_cm': cm_test})
        classification_reports[model_name].append({'train_report': report_train, 'test_report': report_test})

        # In ra các giá trị cho từng fold
        print(f"===== {model_name} - Fold {len(results[model_name])} =====")
        print(f"Train Accuracy: {acc_train:.4f}, Test Accuracy: {acc_test:.4f}")
        print(f"AUC per label (Test): {auc_per_label}")
        print(f"Average AUC (Test): {avg_auc:.4f}")
        print("Train Confusion Matrix:")
        print(cm_train)
        print("Test Confusion Matrix:")
        print(cm_test)
        print("Train Classification Report:")
        print(pd.DataFrame(report_train).T)
        print("Test Classification Report:")
        print(pd.DataFrame(report_test).T)
        print("\n")
        print("\n")

# Tạo một dictionary để ánh xạ nhãn số thành tên lớp
label_map = {0: 'ALL', 1: 'AML'}

# Khởi tạo các danh sách để lưu trữ các chỉ số cho từng model
accuracy_train = {'NB': [], 'LR': [], 'SVM': [], 'KNN': [], 'DT': [], 'RF': []}
accuracy_test = {'NB': [], 'LR': [], 'SVM': [], 'KNN': [], 'DT': [], 'RF': []}

precision_train = {'NB': {'ALL': [], 'AML': []}, 'LR': {'ALL': [], 'AML': []}, 'SVM': {'ALL': [], 'AML': []},
                   'KNN': {'ALL': [], 'AML': []}, 'DT': {'ALL': [], 'AML': []}, 'RF': {'ALL': [], 'AML': []}}
recall_train = {'NB': {'ALL': [], 'AML': []}, 'LR': {'ALL': [], 'AML': []}, 'SVM': {'ALL': [], 'AML': []},
                'KNN': {'ALL': [], 'AML': []}, 'DT': {'ALL': [], 'AML': []}, 'RF': {'ALL': [], 'AML': []}}
f1_score_train = {'NB': {'ALL': [], 'AML': []}, 'LR': {'ALL': [], 'AML': []}, 'SVM': {'ALL': [], 'AML': []},
                  'KNN': {'ALL': [], 'AML': []}, 'DT': {'ALL': [], 'AML': []}, 'RF': {'ALL': [], 'AML': []}}

precision_test = {'NB': {'ALL': [], 'AML': []}, 'LR': {'ALL': [], 'AML': []}, 'SVM': {'ALL': [], 'AML': []},
                  'KNN': {'ALL': [], 'AML': []}, 'DT': {'ALL': [], 'AML': []}, 'RF': {'ALL': [], 'AML': []}}
recall_test = {'NB': {'ALL': [], 'AML': []}, 'LR': {'ALL': [], 'AML': []}, 'SVM': {'ALL': [], 'AML': []},
               'KNN': {'ALL': [], 'AML': []}, 'DT': {'ALL': [], 'AML': []}, 'RF': {'ALL': [], 'AML': []}}
f1_score_test = {'NB': {'ALL': [], 'AML': []}, 'LR': {'ALL': [], 'AML': []}, 'SVM': {'ALL': [], 'AML': []},
                 'KNN': {'ALL': [], 'AML': []}, 'DT': {'ALL': [], 'AML': []}, 'RF': {'ALL': [], 'AML': []}}

auc_train = {'NB': [], 'LR': [], 'SVM': [], 'KNN': [], 'DT': [], 'RF': []}
auc_test = {'NB': [], 'LR': [], 'SVM': [], 'KNN': [], 'DT': [], 'RF': []}

# Lặp qua từng fold
for train_idx, test_idx in kf.split(X, y):
    X_train_fold, X_test_fold = X.iloc[train_idx], X.iloc[test_idx]
    y_train_fold, y_test_fold = y.iloc[train_idx], y.iloc[test_idx]

    for model_name, model in models.items():
        # Huấn luyện mô hình
        model.fit(X_train_fold, y_train_fold)

        # Dự đoán trên tập huấn luyện và kiểm tra
        y_train_pred = model.predict(X_train_fold)
        y_test_pred = model.predict(X_test_fold)

        # Độ chính xác
        acc_train = accuracy_score(y_train_fold, y_train_pred)
        acc_test = accuracy_score(y_test_fold, y_test_pred)

        # AUC
        y_train_prob = model.predict_proba(X_train_fold)[:, 1] if hasattr(model, 'predict_proba') else None
        y_test_prob = model.predict_proba(X_test_fold)[:, 1] if hasattr(model, 'predict_proba') else None

        auc_train_fold = roc_auc_score(y_train_fold, y_train_prob) if y_train_prob is not None else None
        auc_test_fold = roc_auc_score(y_test_fold, y_test_prob) if y_test_prob is not None else None

        # Báo cáo phân loại
        report_train = classification_report(y_train_fold, y_train_pred, output_dict=True)
        report_test = classification_report(y_test_fold, y_test_pred, output_dict=True)

        # Lưu trữ các chỉ số precision, recall, f1-score cho từng lớp
        for label in ['ALL', 'AML']:
            label_idx = 0 if label == 'ALL' else 1

            precision_train[model_name][label].append(report_train[str(label_idx)]['precision'])
            recall_train[model_name][label].append(report_train[str(label_idx)]['recall'])
            f1_score_train[model_name][label].append(report_train[str(label_idx)]['f1-score'])

            precision_test[model_name][label].append(report_test[str(label_idx)]['precision'])
            recall_test[model_name][label].append(report_test[str(label_idx)]['recall'])
            f1_score_test[model_name][label].append(report_test[str(label_idx)]['f1-score'])

        # Lưu trữ độ chính xác và AUC của từng fold
        accuracy_train[model_name].append(acc_train)
        accuracy_test[model_name].append(acc_test)
        if auc_train_fold is not None:
            auc_train[model_name].append(auc_train_fold)
        if auc_test_fold is not None:
            auc_test[model_name].append(auc_test_fold)

# Tính trung bình các chỉ số quan trọng theo từng nhãn cho các model
for model_name in models.keys():
    print(f"===== {model_name} - Summary =====")

    # Tính trung bình và độ lệch chuẩn của Accuracy
    train_acc_mean = np.mean([res['train_accuracy'] for res in results[model_name]])
    test_acc_mean = np.mean([res['test_accuracy'] for res in results[model_name]])

    # Tính trung bình và độ lệch chuẩn của AUC
    mean_auc = np.mean(auc_scores[model_name])
    std_auc = np.std(auc_scores[model_name])

    # Tính trung bình AUC cho từng nhãn
    label_aucs = {label: np.mean([fold_auc[label] for fold_auc in auc_labels[model_name] if fold_auc[label] is not None]) for label in [0, 1]}
    # Tính trung bình các chỉ số cho từng lớp
    avg_precision_train = {label: np.mean(precision_train[model_name][label]) for label in ['ALL', 'AML']}
    avg_recall_train = {label: np.mean(recall_train[model_name][label]) for label in ['ALL', 'AML']}
    avg_f1_score_train = {label: np.mean(f1_score_train[model_name][label]) for label in ['ALL', 'AML']}

    avg_precision_test = {label: np.mean(precision_test[model_name][label]) for label in ['ALL', 'AML']}
    avg_recall_test = {label: np.mean(recall_test[model_name][label]) for label in ['ALL', 'AML']}
    avg_f1_score_test = {label: np.mean(f1_score_test[model_name][label]) for label in ['ALL', 'AML']}

    std_precision_train = {label: np.std(precision_train[model_name][label]) for label in ['ALL', 'AML']}
    std_recall_train = {label: np.std(recall_train[model_name][label]) for label in ['ALL', 'AML']}
    std_f1_score_train = {label: np.std(f1_score_train[model_name][label]) for label in ['ALL', 'AML']}

    std_precision_test = {label: np.std(precision_test[model_name][label]) for label in ['ALL', 'AML']}
    std_recall_test = {label: np.std(recall_test[model_name][label]) for label in ['ALL', 'AML']}
    std_f1_score_test = {label: np.std(f1_score_test[model_name][label]) for label in ['ALL', 'AML']}

    # In các giá trị trung bình
    print(f"Train Accuracy mean = {train_acc_mean:.5f}, std = {np.std(accuracy_train[model_name]):.4f}")
    print(f"Test Accuracy mean = {test_acc_mean:.5f}, std = {np.std(accuracy_test[model_name]):.4f}")
    print(f"Test AUC mean: {np.mean(auc_test[model_name]):.4f}, std = {np.std(auc_test[model_name]):.4f}" )

        

    print(f"Train Precision mean (ALL) = {avg_precision_train['ALL']:.4f}, std = {std_precision_train['ALL']:.4f}, (AML) = {avg_precision_train['AML']:.4f} std = {std_precision_train['AML']:.4f}")
    print(f"Test Precision mean (ALL) = {avg_precision_test['ALL']:.4f} , std = {std_precision_test['ALL']:.4f}, (AML) = {avg_precision_test['AML']:.4f} std = {std_precision_test['AML']:.4f}")

    print(f"Train Recall mean (ALL) = {avg_recall_train['ALL']:.4f}, std =  {std_recall_train['ALL']:.4f}, (AML) = {avg_recall_train['AML']:.4f} std = {std_recall_train['AML']:.4f}")
    print(f"Test Recall mean (ALL) = {avg_recall_test['ALL']:.4f}, std = {std_recall_test['ALL']:.4f}, (AML) = {avg_recall_test['AML']:.4f} std = {std_recall_test['AML']:.4f}")

    print(f"Train F1-Score mean (ALL) = {avg_f1_score_train['ALL']:.4f}, std = {std_f1_score_train['ALL']:.4f}, (AML) = {avg_f1_score_train['AML']:.4f} std = {std_f1_score_train['AML']:.4f}")
    print(f"Test F1-Score mean (ALL) = {avg_f1_score_test['ALL']:.4f} , std = {std_f1_score_test['ALL']:.4f}, (AML) = {avg_f1_score_test['AML']:.4f} std = {std_f1_score_test['AML']:.4f}")

    print("\n")



# """# Lưu Models vào file txt"""


# output_dir = "results_Kfold"
# os.makedirs(output_dir, exist_ok=True)
# output_file = os.path.join(output_dir, "results_kb1_kfold_final.txt")
# # Tạo file và ghi kết quả
# with open(output_file, "w") as file:
#     for model_name in models.keys():
#         file.write(f"===== Model: {model_name} =====\n\n")
        
#         # Ghi kết quả từng fold
#         for i in range(k):
#             file.write(f"--- Fold {i + 1} ---\n")
#             file.write(f"Train Accuracy: {accuracy_train[model_name][i]:.4f}\n")
#             file.write(f"Test Accuracy: {accuracy_test[model_name][i]:.4f}\n")
            
#             if auc_train[model_name]:
#                 file.write(f"Train AUC: {auc_train[model_name][i]:.4f}\n")
#             if auc_test[model_name]:
#                 file.write(f"Test AUC: {auc_test[model_name][i]:.4f}\n")
            
#             file.write("Train Precision:\n")
#             file.write(f"  ALL: {precision_train[model_name]['ALL'][i]:.4f}, AML: {precision_train[model_name]['AML'][i]:.4f}\n")
#             file.write("Test Precision:\n")
#             file.write(f"  ALL: {precision_test[model_name]['ALL'][i]:.4f}, AML: {precision_test[model_name]['AML'][i]:.4f}\n")
            
#             file.write("Train Recall:\n")
#             file.write(f"  ALL: {recall_train[model_name]['ALL'][i]:.4f}, AML: {recall_train[model_name]['AML'][i]:.4f}\n")
#             file.write("Test Recall:\n")
#             file.write(f"  ALL: {recall_test[model_name]['ALL'][i]:.4f}, AML: {recall_test[model_name]['AML'][i]:.4f}\n")
            
#             file.write("Train F1-Score:\n")
#             file.write(f"  ALL: {f1_score_train[model_name]['ALL'][i]:.4f}, AML: {f1_score_train[model_name]['AML'][i]:.4f}\n")
#             file.write("Test F1-Score:\n")
#             file.write(f"  ALL: {f1_score_test[model_name]['ALL'][i]:.4f}, AML: {f1_score_test[model_name]['AML'][i]:.4f}\n\n")
            
#             # Ghi ma trận nhầm lẫn
#             file.write("Train Confusion Matrix:\n")
#             train_cm = confusion_matrices[model_name][i]['train_cm']
#             for row in train_cm:
#                 file.write("  " + " ".join(map(str, row)) + "\n")
            
#             file.write("Test Confusion Matrix:\n")
#             test_cm = confusion_matrices[model_name][i]['test_cm']
#             for row in test_cm:
#                 file.write("  " + " ".join(map(str, row)) + "\n")
            
#             file.write("\n")
        
#         # Ghi kết quả trung bình
#         file.write("=== Summary ===\n")
#         file.write(f"Train Accuracy mean: {np.mean(accuracy_train[model_name]):.4f}, std: {np.std(accuracy_train[model_name]):.4f}\n")
#         file.write(f"Test Accuracy mean: {np.mean(accuracy_test[model_name]):.4f}, std: {np.std(accuracy_test[model_name]):.4f}\n")
        
#         if auc_train[model_name]:
#             file.write(f"Train AUC mean: {np.mean(auc_train[model_name]):.4f}\n")
#         if auc_test[model_name]:
#             file.write(f"Test AUC mean: {np.mean(auc_test[model_name]):.4f}\n")
        
#         file.write("Precision (Train):\n")
#         file.write(f"  ALL: {np.mean(precision_train[model_name]['ALL']):.4f}, AML: {np.mean(precision_train[model_name]['AML']):.4f}\n")
#         file.write("Precision (Test):\n")
#         file.write(f"  ALL: {np.mean(precision_test[model_name]['ALL']):.4f}, AML: {np.mean(precision_test[model_name]['AML']):.4f}\n")
        
#         file.write("Recall (Train):\n")
#         file.write(f"  ALL: {np.mean(recall_train[model_name]['ALL']):.4f}, AML: {np.mean(recall_train[model_name]['AML']):.4f}\n")
#         file.write("Recall (Test):\n")
#         file.write(f"  ALL: {np.mean(recall_test[model_name]['ALL']):.4f}, AML: {np.mean(recall_test[model_name]['AML']):.4f}\n")
        
#         file.write("F1-Score (Train):\n")
#         file.write(f"  ALL: {np.mean(f1_score_train[model_name]['ALL']):.4f}, AML: {np.mean(f1_score_train[model_name]['AML']):.4f}\n")
#         file.write("F1-Score (Test):\n")
#         file.write(f"  ALL: {np.mean(f1_score_test[model_name]['ALL']):.4f}, AML: {np.mean(f1_score_test[model_name]['AML']):.4f}\n")
#         file.write("\n\n")
        
# print(f"Results have been saved to {output_file}")