import os
import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Đường dẫn tương đối đến tệp kết quả
relative_path = './results_Kfold/results_kb1_kfold_final.txt'  # Tệp nằm trong cùng thư mục với file script
file_path = os.path.join(os.path.dirname(__file__), relative_path)

# Đọc file kết quả
with open(file_path, 'r') as file:
    results = file.read()

# Tách dữ liệu theo từng model
models_data = re.split(r"===== Model: ", results)
models_data = [data.strip() for data in models_data if data.strip()]

# Hàm để lấy ma trận nhầm lẫn từ text của từng fold
def extract_confusion_matrices(model_data):
    model_name = model_data.split('\n')[0].strip()
    folds = re.findall(r"--- Fold \d+ ---\n.*?Test Confusion Matrix:\n(.*?)\n\n", model_data, re.S)
    confusion_matrices = []
    for fold in folds:
        matrix = np.array([list(map(int, re.findall(r'\d+', row))) for row in fold.split('\n') if row.strip()])
        confusion_matrices.append(matrix)
    return model_name, confusion_matrices

# Xử lý tất cả các model và ma trận nhầm lẫn
confusion_data = {}
for model_data in models_data:
    model_name, confusion_matrices = extract_confusion_matrices(model_data)
    confusion_data[model_name] = confusion_matrices

# Hàm vẽ và lưu ma trận nhầm lẫn dạng percentage
def save_percentage_confusion_matrix(cm, fold, model_name, output_dir):
    # Tính percentage matrix (chuẩn hóa theo tổng toàn bộ dataset)
    percentage_matrix = (cm / cm.sum().sum())

    # Vẽ heatmap với percentage
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        percentage_matrix, 
        annot=True, 
        fmt='.2%', 
        cmap='Oranges',  # Sử dụng bảng màu YlGnBu
        cbar=True,      # Hiển thị thanh màu
        xticklabels=["ALL", "AML"], 
        yticklabels=["ALL", "AML"],
        annot_kws={"size": 22} # Tùy chỉnh kích thước chữ
    )
    plt.title(f'{model_name} Confusion Matrix (Fold {fold+1})', fontsize=20, fontweight='bold')
    plt.xlabel('Predicted labels',fontsize=18)
    plt.ylabel('True labels',fontsize=18)
    plt.xticks(fontsize=20)  # Kích thước chữ nhãn trục X (ALL, AML)
    plt.yticks(fontsize=20) 
    
    # Đường dẫn lưu hình ảnh
    output_path = os.path.join(output_dir, f"{model_name}_Fold{fold+1}_percentage.png")
    plt.savefig(output_path, dpi=100)
    plt.close()  # Đóng hình để tránh chiếm bộ nhớ
    print(f"Image saved at: {output_path}")

# Tạo thư mục để lưu hình ảnh
output_dir = os.path.join(os.path.dirname(__file__), "confusion_matrices_kb1_percentage")
os.makedirs(output_dir, exist_ok=True)

# Vẽ và lưu từng ma trận nhầm lẫn
for model_name, matrices in confusion_data.items():
    for fold, cm in enumerate(matrices):
        save_percentage_confusion_matrix(cm, fold, model_name, output_dir)

print(f"All percentage confusion matrices are saved in: {output_dir}")
# # In ma trận nhầm lẫn gốc trước khi chuẩn hóa
# print(f"Original Confusion Matrix for {model_name}, Fold {fold+1}:")
# print(cm)

# # In ma trận dạng phần trăm
# percentage_matrix = cm / cm.sum().sum()
# print(f"Percentage Matrix for {model_name}, Fold {fold+1}:")
# print(percentage_matrix)
