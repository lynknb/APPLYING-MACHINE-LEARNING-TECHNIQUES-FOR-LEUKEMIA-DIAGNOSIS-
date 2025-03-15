
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

# Khởi tạo KFold và danh sách top k gen
kf = KFold(n_splits=5, shuffle=True, random_state=42)
top_k_gen_list = [3, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300, 400, 500, 600, 700, 800]


# Mô hình và kết quả
models = {
    'Random Forest': RandomForestClassifier()

}

fold_results = []
# Thêm danh sách lưu kết quả từng chỉ số theo fold
metric_results = {
    'accuracy': [],
    'precision': [],
    'recall': [],
    'f1_score': [],
    'auc': []
}
# Mở tệp để ghi kết quả
log_file = 'LV_Topk_kflod_kb4.txt'

with open(log_file, 'w') as f:
    for top_k in top_k_gen_list:
        f.write(f"\n=== Running with top {top_k} gen ===")    
        for fold, (train_index, test_index) in enumerate(kf.split(X)):
            f.write(f"\nFold {fold + 1}\n")
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
            rejected, adjusted_p_values = fdrcorrection(df_results['P_Value'])
            significant_genes = df_results[rejected & (adjusted_p_values < 0.05)]
            significant_genes = significant_genes.sort_values(by='P_Value')

            # Thêm thứ hạng và chọn top 3 gen
            significant_genes['Rank'] = significant_genes['P_Value'].rank()
            top_genes_names = significant_genes['Gene Accession Number'].head(top_k).tolist()
            f.write(f"Top {top_k} gen significant: {top_genes_names}\n")
            print(f"Top {top_k} gen đặc trưng:", top_genes_names)

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

                # Tính toán các chỉ số trên tập train
                acc_train = accuracy_score(y_train, y_pred_train)
                auc_train = roc_auc_score(y_train, y_pred_prob_train) if y_pred_prob_train is not None else None
                cm_train = confusion_matrix(y_train, y_pred_train)
                report_train = classification_report(y_train, y_pred_train)
                precision_train = precision_score(y_train, y_pred_train)
                recall_train = recall_score(y_train, y_pred_train)
                f1_train = f1_score(y_train, y_pred_train)

                f.write(f"{model_name} Train Accuracy: {acc_train:.4f}\n")
                if auc_train is not None:
                    f.write(f"{model_name} Train AUC: {auc_train:.4f}\n")
                f.write(f"{model_name} Train Confusion Matrix:\n{cm_train}\n")
                f.write(f"{model_name} Train Classification Report:\n{report_train}\n")
                f.write(f"{model_name} Train Precision: {precision_train:.4f}\n")
                f.write(f"{model_name} Train Recall: {recall_train:.4f}\n")
                f.write(f"{model_name} Train F1-Score: {f1_train:.4f}\n")

                # Tính toán các chỉ số trên tập test
                acc = accuracy_score(y_test, y_pred)
                auc = roc_auc_score(y_test, y_pred_prob) if y_pred_prob is not None else None
                cm = confusion_matrix(y_test, y_pred)
                report = classification_report(y_test, y_pred)
                precision_test = precision_score(y_test, y_pred)
                recall_test = recall_score(y_test, y_pred)
                f1_test = f1_score(y_test, y_pred)

                # Lưu kết quả từng fold
                metric_results['accuracy'].append(acc)
                metric_results['precision'].append(precision_test)
                metric_results['recall'].append(recall_test)
                metric_results['f1_score'].append(f1_test)
                if auc is not None:
                    metric_results['auc'].append(auc)

                f.write(f"{model_name} Fold {fold + 1} Test Accuracy: {acc:.4f}\n")
                f.write(f"{model_name} Fold {fold + 1} Test Precision: {precision_test:.4f}\n")
                f.write(f"{model_name} Fold {fold + 1} Test Recall: {recall_test:.4f}\n")
                f.write(f"{model_name} Fold {fold + 1} Test F1-Score: {f1_test:.4f}\n")
                if auc is not None:
                    f.write(f"{model_name} Fold {fold + 1} Test AUC: {auc:.4f}\n")
                # Lưu kết quả
                fold_results.append({
                    'Top_k': top_k,
                    'Fold': fold + 1,
                    'Model': model_name,
                    'Train Accuracy': acc_train,
                    'Train AUC': auc_train,
                    'Test Accuracy': acc,
                    'Test AUC': auc,
                    'Train Precision': precision_train,
                    'Train Recall': recall_train,
                    'Train F1-Score': f1_train,
                    'Test Precision': precision_test,
                    'Test Recall': recall_test,
                    'Test F1-Score': f1_test,
                    'Top Genes': top_genes_names,
                })
            

        # Tính độ lệch chuẩn của các chỉ số
        std_acc = np.std(metric_results['accuracy'])
        std_prec = np.std(metric_results['precision'])
        std_rec = np.std(metric_results['recall'])
        std_f1 = np.std(metric_results['f1_score'])
        std_auc = np.std(metric_results['auc']) if metric_results['auc'] else None

        # Ghi độ lệch chuẩn vào tệp
        f.write(f"\n=== Standard Deviation for Top {top_k} Gen ===\n")
        f.write(f"Accuracy Std: {std_acc:.4f}\n")
        f.write(f"Precision Std: {std_prec:.4f}\n")
        f.write(f"Recall Std: {std_rec:.4f}\n")
        f.write(f"F1-Score Std: {std_f1:.4f}\n")
        if std_auc is not None:
            f.write(f"AUC Std: {std_auc:.4f}\n")

        # Xóa kết quả của các fold trước đó để tính toán cho top_k tiếp theo
        for key in metric_results.keys():
            metric_results[key].clear()

# Tổng hợp kết quả
        results_df = pd.DataFrame(fold_results)
        summary = results_df.groupby(['Top_k', 'Model']).agg({
                'Train Accuracy': 'mean',
                'Train AUC': 'mean',
                'Test Accuracy': 'mean',
                'Test AUC': 'mean',
                'Train Precision': 'mean',
                'Train Recall': 'mean',
                'Train F1-Score': 'mean',
                'Test Precision': 'mean',
                'Test Recall': 'mean',
                'Test F1-Score': 'mean'
            }).sort_values(by='Test Accuracy', ascending=False)
# Ghi kết quả vào tệp
with open(log_file, 'a') as f:  # Chế độ 'a' để ghi thêm vào tệp
    f.write("\nResults Summary:\n")
    f.write(summary.to_string())
print(f"Kết quả đầy đủ đã được lưu vào tệp: {log_file}")
