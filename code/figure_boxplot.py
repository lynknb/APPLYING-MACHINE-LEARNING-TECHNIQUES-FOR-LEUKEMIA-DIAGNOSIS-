

"""**Boxplot cho kịch bản 3**"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import matplotlib.pyplot as plt



# Dữ liệu AUC cho từng model ở mỗi fold
data = {
    "Model": [
        "Naive Bayes", "Logistic Regression", "SVM", "KNN", "Random Forest", "Decision Tree",
        "Naive Bayes", "Logistic Regression", "SVM", "KNN", "Random Forest", "Decision Tree",
        "Naive Bayes", "Logistic Regression", "SVM", "KNN", "Random Forest", "Decision Tree",
        "Naive Bayes", "Logistic Regression", "SVM", "KNN", "Random Forest", "Decision Tree",
        "Naive Bayes", "Logistic Regression", "SVM", "KNN", "Random Forest", "Decision Tree"
    ],
    "Fold": [
        1, 1, 1, 1, 1, 1,
        2, 2, 2, 2, 2, 2,
        3, 3, 3, 3, 3, 3,
        4, 4, 4, 4, 4, 4,
        5, 5, 5, 5, 5, 5
    ],
    "AUC": [
        0.9200, 0.9000, 0.9600, 0.9200, 0.9400, 0.8500,  # Fold 1
        1.0000, 0.9815, 1.0000, 0.9444, 1.0000, 0.9444,  # Fold 2
        1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 0.9545,  # Fold 3
        0.8222, 0.8667, 0.8889, 0.8667, 0.8889, 0.7889,  # Fold 4
        1.0000, 1.0000, 1.0000, 0.9896, 1.0000, 1.0000   # Fold 5
    ]
}

# Chuyển dữ liệu thành DataFrame
df = pd.DataFrame(data)

# Định nghĩa màu sắc cho mỗi model
palette = {
    "Naive Bayes": "skyblue",
    "Logistic Regression": "lightgreen",
    "SVM": "lightcoral",
    "KNN": "lightyellow",
    "Random Forest": "lightpink",
    "Decision Tree": "lightgray"
}

# Sắp xếp lại thứ tự các model theo yêu cầu
model_order = ["Naive Bayes", "Logistic Regression", "SVM", "KNN", "Random Forest", "Decision Tree"]

# Vẽ boxplot cho AUC của mỗi model qua các fold với màu sắc khác nhau
plt.figure(figsize=(12, 6))
sns.boxplot(x="Model", y="AUC", data=df, order=model_order, palette=palette, showmeans=True, meanline=True)

# Thêm tiêu đề và nhãn
plt.title("Boxplot giá trị AUC của mỗi mô hình sử dụng 5-fold cross-validation", fontsize=16)
plt.xlabel("Model", fontsize=12)
plt.ylabel("AUC", fontsize=12)

plt.tight_layout()
plt.savefig("Boxplot-kb3.png")
plt.show()





# Đọc dữ liệu từ file Excel
file_path = './Topk_kfold_results.xlsx'  # Đổi đường dẫn nếu cần
fold_results = pd.read_excel(file_path, sheet_name='Fold_Results')

# Đọc dữ liệu từ file Excel
fold_results = pd.read_excel(file_path, sheet_name='Fold_Results')

# Lấy danh sách các mô hình
unique_models = fold_results['Model'].unique()

# Cấu hình để vẽ các biểu đồ
plt.figure(figsize=(15, 5 * len(unique_models)))

# Vẽ boxplot cho từng mô hình
for i, model in enumerate(unique_models, 1):
    plt.subplot(len(unique_models), 1, i)
    model_data = fold_results[fold_results['Model'] == model]

    # Vẽ boxplot
    sns.boxplot(x='Top_k', y='Test AUC', data=model_data)

    # Tính giá trị mean của Test AUC
    mean_value = model_data['Test AUC'].mean()

    # Thêm đường kẻ trung bình
    plt.axhline(mean_value, color='blue', linestyle='-', label=f'Mean AUC: {mean_value:.2f}')

    # Thiết lập tiêu đề và nhãn
    plt.title(f'Boxplot of Test AUC by Top_k for {model}')
    plt.xlabel('Top_k')
    plt.ylabel('AUC')

    # Thêm chú thích (legend)
    plt.legend()

plt.tight_layout()

# Hiển thị biểu đồ
plt.savefig("Boxplot-kb4.png")
plt.show()