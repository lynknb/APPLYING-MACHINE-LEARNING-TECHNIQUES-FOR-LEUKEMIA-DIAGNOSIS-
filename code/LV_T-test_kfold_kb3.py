
import pandas as pd
import numpy as np
import time
import os
from scipy.stats import ttest_ind, f_oneway
from statsmodels.stats.multitest import fdrcorrection
from sklearn.model_selection import KFold


import matplotlib.pyplot as plt
# %matplotlib inline
import seaborn as sns

from collections import Counter

from imblearn.over_sampling import SMOTE

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix


# Lấy đường dẫn đến thư mục hiện tại
data_dir = os.getcwd()

"""### Load data"""

# Loading training dataset
df_train = pd.read_csv(os.path.join(data_dir, 'data_set_ALL_AML_train.csv'))

# Loading testing dataset

df_test = pd.read_csv(os.path.join(data_dir, 'data_set_ALL_AML_independent.csv'))

print("df_train: df_train.shape", df_train.shape)
print("df_test: df_test.shape", df_test.shape)

"""Dữ liệu bao gồm 7129 hàng và 78 cột cho tập huấn luyện và 70 cột cho tập kiểm tra. Mỗi hàng tương ứng với một trong 7129 gen và mỗi cột đại diện cho một bệnh nhân. Do đó, mỗi tế bào chứa mức độ biểu hiện của một gen cụ thể cho một bệnh nhân cụ thể.

# Cleaning data
"""

# Remove call columns in training dataset
columns_to_remove_train = [col for col in df_train if 'call' in col]
train = df_train.drop(columns_to_remove_train, axis=1)

# Remove call columns in testing dataset
columns_to_remove_test = [col for col in df_test if 'call' in col]
test = df_test.drop(columns_to_remove_test, axis=1)

train_columns_titles = ['Gene Description', 'Gene Accession Number', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10',
       '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25',
       '26', '27', '28', '29', '30', '31', '32', '33', '34', '35', '36', '37', '38']

train = train.reindex(columns=train_columns_titles)

test_columns_titles = ['Gene Description', 'Gene Accession Number','39', '40', '41', '42', '43', '44', '45', '46',
       '47', '48', '49', '50', '51' , '52', '53',  '54', '55', '56', '57', '58', '59',
       '60', '61', '62', '63', '64', '65', '66', '67', '68', '69', '70', '71', '72']

test = test.reindex(columns=test_columns_titles)

"""Chuyển vị các cột và hàng sao cho 7129 gen trở thành các đặc điểm và mỗi trường hợp bệnh nhân chiếm một hàng duy nhất."""

# Transpose row and columns in training set
X_train = train.T

# Transpose row and columns in testing set
X_test = test.T

print(X_train.shape)
X_train.head()

"""Chỉ định hàng thứ hai chứa số lượng gen gia nhập làm tên cột. Sau đó, xóa cả hai hàng đầu tiên của Mô tả gen và Số gia nhập gen."""

# Đặt hàng thứ 2 (Gene Accession Number) làm tên cột
X_train.columns = X_train.iloc[1]  # for training set
X_test.columns = X_test.iloc[1]  # for testing set

# Bỏ 2 hàng đầu tiên (Gene Description and Gene Accession Number) và lập lại chỉ mục
X_train = X_train.iloc[2:].reset_index(drop=True)  # for training set
X_test = X_test.iloc[2:].reset_index(drop=True)  # for testing set

print("X_train:", X_train.shape)
print("X_test: ", X_test.shape)

"""Chuyển đổi giá trị dữ liệu thành số."""

# Convert data values to numeric for training set
X_train = X_train.astype(float, 64)
# Convert data values to numeric for testing set
X_test =  X_test.astype(float, 64)

"""**Load labels**"""
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

"""Có nhiều bệnh nhân ALL hơn bệnh nhân AML. Chúng ta cũng hãy chuyển đổi nhãn thành giá trị số với ALL là 0 và AML là 1."""

# Replace values in the 'cancer' column with 0 for 'ALL' and 1 for 'AML'
cancer_mapping = {'ALL': 0, 'AML': 1}
merged_XY['cancer'] = merged_XY['cancer'].map(cancer_mapping)


"""Splitting the data into train and test sets"""

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import fdrcorrection
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report


X = merged_XY.drop(columns=['cancer'])  # Dữ liệu biểu hiện gen
y = merged_XY['cancer']  # Nhãn

# Thiết lập K-Fold
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Mô hình và kết quả
models = {
    'Naive Bayes': GaussianNB(),
    'Logistic Regression': LogisticRegression(),
    'SVM': SVC(probability=True),
    'KNN': KNeighborsClassifier(),
    'Random Forest': RandomForestClassifier(),
    'Decision Tree': DecisionTreeClassifier()
}

# Khởi tạo từ điển để lưu các chỉ số
classification_metrics = {model_name: {'Precision': [], 'Recall': [], 'F1-Score': []} for model_name in models.keys()}

fold_results = []

for fold, (train_index, test_index) in enumerate(kf.split(X)):
    print(f"\nFold {fold + 1}")
    
    # Chia tập huấn luyện và kiểm tra
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # Chuẩn hóa dữ liệu
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # T-Test với điều chỉnh FDR để tìm gen đặc trưng
    X_all = X_train[y_train == 0]  # ALL
    X_aml = X_train[y_train == 1]  # AML


    results1 = []
    for gene in X_train.columns:
        t_value, p_value = ttest_ind(X_all[gene], X_aml[gene], equal_var=False)
        results1.append((gene, t_value, p_value))

    df_results = pd.DataFrame(results1, columns=['Gene Accession Number', 'T_Value', 'P_Value'])
    df_results = df_results.drop(df_results.index[-1])

    rejected, adjusted_p_values = fdrcorrection(df_results['P_Value'])
    significant_genes = df_results[rejected & (adjusted_p_values < 0.05)]
    significant_genes = significant_genes.sort_values(by='P_Value')
    

    # Thêm thứ hạng và chọn top 3 gen
    significant_genes['Rank'] = significant_genes['P_Value'].rank()
    top_genes_names = significant_genes['Gene Accession Number'].head(3).tolist()
    print("Top 3 gen đặc trưng:", top_genes_names)

    # # Số lượng gen có p-value >0,05
    reduced_genes_count = X_train.shape[1] - len(significant_genes) -1
    print(f"Tổng số lượng gen với p-value > 0.05: {reduced_genes_count}")

    # Tính số lượng gen ở ngưỡng adjusted p < 0.05
    print(f"Tổng số gen với p-value < 0.05: {len(significant_genes)}")

    # Chọn dữ liệu với các gen đặc trưng
    X_train_top = X_train_scaled[:, [X.columns.get_loc(g) for g in top_genes_names]]
    X_test_top = X_test_scaled[:, [X.columns.get_loc(g) for g in top_genes_names]]

    # Huấn luyện và đánh giá các mô hình
    for model_name, model in models.items():
        model.fit(X_train_top, y_train)
        y_pred_train = model.predict(X_train_top)
        y_pred = model.predict(X_test_top)
        y_pred_prob_train = model.predict_proba(X_train_top)[:, 1] if hasattr(model, "predict_proba") else None
        y_pred_prob = model.predict_proba(X_test_top)[:, 1] if hasattr(model, "predict_proba") else None

         # Tính toán báo cáo phân loại
        report = classification_report(y_test, y_pred, output_dict=True)

        # Tính toán các chỉ số trên tập train
        acc_train = accuracy_score(y_train, y_pred_train)
        auc_train = roc_auc_score(y_train, y_pred_prob_train) if y_pred_prob_train is not None else None
        cm_train = confusion_matrix(y_train, y_pred_train)
        report_train = classification_report(y_train, y_pred_train)

        # Lưu các chỉ số Precision, Recall, F1-Score cho từng fold
        classification_metrics[model_name]['Precision'].append(report['accuracy'])
        classification_metrics[model_name]['Recall'].append(report['macro avg']['recall'])
        classification_metrics[model_name]['F1-Score'].append(report['macro avg']['f1-score'])

        print(f"{model_name} Train Accuracy: {acc_train:.4f}")
        if auc_train is not None:
            print(f"{model_name} Train AUC: {auc_train:.4f}")
        print(f"{model_name} Train Confusion Matrix:\n{cm_train}")
        print(f"{model_name} Train Classification Report:\n{report_train}")

        # Tính toán các chỉ số trên tập test
        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_prob) if y_pred_prob is not None else None
        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred)

        print(f"{model_name} Test Accuracy: {acc:.4f}")
        if auc is not None:
            print(f"{model_name} Test AUC: {auc:.4f}")
        print(f"{model_name} Test Confusion Matrix:\n{cm}")
        print(f"{model_name} Test Classification Report:\n{report}")

        # Lưu kết quả
        fold_results.append({
            'Fold': fold + 1,
            'Model': model_name,
            'Train Accuracy': acc_train,
            'Train AUC': auc_train,
            'Test Accuracy': acc,
            'Test AUC': auc,
            'Top Genes': top_genes_names,
            # 'Reduced Genes Count': reduced_genes_count
        })

# Chuyển kết quả thành DataFrame
results_df = pd.DataFrame(fold_results)
print("\nTổng hợp kết quả:")
print(results_df.groupby('Model')[['Train Accuracy', 'Train AUC', 'Test Accuracy', 'Test AUC']].mean().sort_values(by='Test Accuracy', ascending=False))

# Tính mean và std cho từng chỉ số của từng model
results_summary = results_df.groupby('Model').agg({
    'Test Accuracy': ['mean', 'std'],
    'Test AUC': ['mean', 'std']
}).reset_index()

# Đổi tên cột cho dễ đọc
results_summary.columns = ['Model', 'Test Accuracy Mean', 'Test Accuracy Std', 'Test AUC Mean', 'Test AUC Std']

print("\nTổng hợp kết quả với Mean và Std:")
print(results_summary)


# Tính mean và std cho từng chỉ số của từng model
results_summary = results_df.groupby('Model').agg({
    'Test Accuracy': ['mean', 'std'],
    'Test AUC': ['mean', 'std']
}).reset_index()

# Đổi tên cột cho dễ đọc
results_summary.columns = ['Model', 'Test Accuracy Mean', 'Test Accuracy Std', 'Test AUC Mean', 'Test AUC Std']

print("\nTổng hợp kết quả với Mean và Std:")
print(results_summary)

# Tính mean và std cho từng chỉ số của từng model
classification_summary = []
for model_name, metrics in classification_metrics.items():
    precision_mean = np.mean(metrics['Precision'])
    precision_std = np.std(metrics['Precision'])
    recall_mean = np.mean(metrics['Recall'])
    recall_std = np.std(metrics['Recall'])
    f1_mean = np.mean(metrics['F1-Score'])
    f1_std = np.std(metrics['F1-Score'])

    classification_summary.append({
        'Model': model_name,
        'Precision Mean': precision_mean,
        'Precision Std': precision_std,
        'Recall Mean': recall_mean,
        'Recall Std': recall_std,
        'F1-Score Mean': f1_mean,
        'F1-Score Std': f1_std,
    })

# Chuyển kết quả thành DataFrame
classification_summary_df = pd.DataFrame(classification_summary)
print("\nTổng hợp kết quả với Mean và Std cho Precision, Recall, F1-Score:")
print(classification_summary_df)
# # Mở tệp để ghi kết quả
# log_file = './results_Kfold/results_Ttest.txt'

# with open(log_file, 'w') as f:
#     for fold, (train_index, test_index) in enumerate(kf.split(X)):
#         f.write(f"\nFold {fold + 1}\n")
#         print(f"\nFold {fold + 1}")

#         # Chia tập huấn luyện và kiểm tra
#         X_train, X_test = X.iloc[train_index], X.iloc[test_index]
#         y_train, y_test = y.iloc[train_index], y.iloc[test_index]

#         # Chuẩn hóa dữ liệu
#         scaler = StandardScaler()
#         X_train_scaled = scaler.fit_transform(X_train)
#         X_test_scaled = scaler.transform(X_test)

#         # T-Test với điều chỉnh FDR để tìm gen đặc trưng
#         X_all = X_train[y_train == 0]  # ALL
#         X_aml = X_train[y_train == 1]  # AML

#         results1 = []
#         for gene in X_train.columns:
#             t_value, p_value = ttest_ind(X_all[gene], X_aml[gene], equal_var=False)
#             results1.append((gene, t_value, p_value))

#         df_results = pd.DataFrame(results1, columns=['Gene Accession Number', 'T_Value', 'P_Value'])
#         rejected, adjusted_p_values = fdrcorrection(df_results['P_Value'])
#         significant_genes = df_results[rejected & (adjusted_p_values < 0.05)]
#         significant_genes = significant_genes.sort_values(by='P_Value')

#         # Thêm thứ hạng và chọn top 3 gen
#         significant_genes['Rank'] = significant_genes['P_Value'].rank()
#         top_genes_names = significant_genes['Gene Accession Number'].head(3).tolist()
#         f.write(f"Top 3 gen đặc trưng: {top_genes_names}\n")
#         print("Top 3 gen đặc trưng:", top_genes_names)

#         # Chọn dữ liệu với các gen đặc trưng
#         X_train_top = X_train_scaled[:, [X.columns.get_loc(g) for g in top_genes_names]]
#         X_test_top = X_test_scaled[:, [X.columns.get_loc(g) for g in top_genes_names]]

#         # Huấn luyện và đánh giá các mô hình
#         for model_name, model in models.items():
#             model.fit(X_train_top, y_train)
#             y_pred_train = model.predict(X_train_top)
#             y_pred = model.predict(X_test_top)
#             y_pred_prob_train = model.predict_proba(X_train_top)[:, 1] if hasattr(model, "predict_proba") else None
#             y_pred_prob = model.predict_proba(X_test_top)[:, 1] if hasattr(model, "predict_proba") else None

#             # Tính toán các chỉ số trên tập train
#             acc_train = accuracy_score(y_train, y_pred_train)
#             auc_train = roc_auc_score(y_train, y_pred_prob_train) if y_pred_prob_train is not None else None
#             cm_train = confusion_matrix(y_train, y_pred_train)
#             report_train = classification_report(y_train, y_pred_train)

#             f.write(f"{model_name} Train Accuracy: {acc_train:.4f}\n")
#             if auc_train is not None:
#                 f.write(f"{model_name} Train AUC: {auc_train:.4f}\n")
#             f.write(f"{model_name} Train Confusion Matrix:\n{cm_train}\n")
#             f.write(f"{model_name} Train Classification Report:\n{report_train}\n")

#             # Tính toán các chỉ số trên tập test
#             acc = accuracy_score(y_test, y_pred)
#             auc = roc_auc_score(y_test, y_pred_prob) if y_pred_prob is not None else None
#             cm = confusion_matrix(y_test, y_pred)
#             report = classification_report(y_test, y_pred)

#             f.write(f"{model_name} Test Accuracy: {acc:.4f}\n")
#             if auc is not None:
#                 f.write(f"{model_name} Test AUC: {auc:.4f}\n")
#             f.write(f"{model_name} Test Confusion Matrix:\n{cm}\n")
#             f.write(f"{model_name} Test Classification Report:\n{report}\n")

#             # Lưu kết quả
#             fold_results.append({
#                 'Fold': fold + 1,
#                 'Model': model_name,
#                 'Train Accuracy': acc_train,
#                 'Train AUC': auc_train,
#                 'Test Accuracy': acc,
#                 'Test AUC': auc,
#                 'Top Genes': top_genes_names,
#             })

#     # Tổng hợp kết quả
#     results_df = pd.DataFrame(fold_results)
#     summary = results_df.groupby('Model')[['Train Accuracy', 'Train AUC', 'Test Accuracy', 'Test AUC']].mean().sort_values(by='Test Accuracy', ascending=False)
#     f.write("\nTổng hợp kết quả:\n")
#     f.write(str(summary))

# # In kết quả ra màn hình
# print("\nTổng hợp kết quả:")
# print(summary)
# print(f"Kết quả đầy đủ đã được lưu vào tệp: {log_file}")
