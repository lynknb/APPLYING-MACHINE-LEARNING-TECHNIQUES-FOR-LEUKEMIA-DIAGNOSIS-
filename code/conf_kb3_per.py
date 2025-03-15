import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re

# Hàm trích xuất ma trận nhầm lẫn từ file
def extract_confusion_matrices(file_path, model_name):
    matrices = []
    with open(file_path, 'r') as file:
        lines = file.readlines()
        pattern = f"{model_name} Test Confusion Matrix:"
        for i, line in enumerate(lines):
            if pattern in line:
                # Lấy 2 dòng tiếp theo là ma trận nhầm lẫn
                matrix = [list(map(int, re.findall(r'\d+', lines[i + 1]))),
                          list(map(int, re.findall(r'\d+', lines[i + 2])))]
                matrices.append(np.array(matrix))
    return matrices

# Hàm vẽ và lưu ma trận nhầm lẫn từng fold
def plot_and_save_each_fold(matrices, model_name, output_dir):
    for idx, matrix in enumerate(matrices):
        # Chuyển đổi thành phần trăm toàn bộ ma trận
        matrix_percentage = (matrix / matrix.sum().sum()) 
        plt.figure(figsize=(8, 6))
        sns.heatmap(matrix_percentage, annot=True, fmt=".2%", cmap="Oranges", cbar=True,
                    xticklabels=["ALL", "AML"],
                    yticklabels=["ALL", "AML"],
                    annot_kws={"size": 22})
        
        plt.title(f"{model_name} ===== Confusion Matrix Fold {idx + 1}", fontsize=17.5, fontweight='bold')
        plt.xlabel('Predicted labels',fontsize=16)
        plt.ylabel('True labels',fontsize=16)
        plt.xticks(fontsize=22)  # Kích thước chữ nhãn trục X (ALL, AML)
        plt.yticks(fontsize=22) 
        
        # Lưu hình ảnh
        output_path = os.path.join(output_dir, f"{model_name}_Fold{idx + 1}.png")
        plt.tight_layout()
        plt.savefig(output_path, dpi=100)
        plt.close()  # Đóng hình để tiết kiệm bộ nhớ

# Tạo thư mục lưu trữ
output_dir = "confusion_matrices_kb3_percentage"
os.makedirs(output_dir, exist_ok=True)

# File TXT chứa kết quả
file_path = './results_Kfold/results_Ttest.txt'

# Danh sách các mô hình
models = ["Naive Bayes", "Logistic Regression", "SVM", "KNN", "Random Forest", "Decision Tree"]

# Thực hiện lưu từng fold cho từng mô hình
for model in models:
    matrices = extract_confusion_matrices(file_path, model)
    if matrices:
        plot_and_save_each_fold(matrices, model, output_dir)

print(f"Đã lưu tất cả biểu đồ từng fold vào thư mục '{output_dir}'.")
