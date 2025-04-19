import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# 设置pandas显示选项，保留完整精度
pd.set_option('display.float_format', lambda x: '{:.15f}'.format(x) if isinstance(x, float) else str(x))
np.set_printoptions(precision=15, suppress=True)

# 加载数据，指定header=None表示没有表头
file_path = r"D:\desktop\毕设材料\processed_data.xlsx"
data = pd.read_excel(file_path, header=None)

# 查看数据前几行，了解数据结构
print("First 5 rows:")
print(data.head())

# 提取特征(6个TOA值)和标签
X = data.iloc[:, 3:9].values  # 第4到9列作为特征 (索引从0开始，所以是3到8)
y = data.iloc[:, 2].values    # 第3列作为标签 (索引从0开始，所以是2)

# 确保标签是整数类型
y = y.astype(int)

# 创建k折交叉验证对象，设置随机种子确保可重复性，并进行随机抽样
k_folds = 5
kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)

# 创建一个数组来存储所有数据的无泄漏预测
y_pred_all = np.zeros_like(y)

# 创建一个列表来存储每个折的准确率
fold_accuracies = []

# 使用K折交叉验证生成无泄漏预测
print(f"\n执行{k_folds}折交叉验证...")
for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
    # 获取当前折的训练和测试数据
    X_train_fold, X_test_fold = X[train_idx], X[test_idx]
    y_train_fold, y_test_fold = y[train_idx], y[test_idx]
    
    # 创建并训练随机森林分类器
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_classifier.fit(X_train_fold, y_train_fold)
    
    # 对测试折进行预测
    y_pred_fold = rf_classifier.predict(X_test_fold)
    
    # 将预测结果存储在对应的位置
    y_pred_all[test_idx] = y_pred_fold
    
    # 计算当前折的准确率
    fold_accuracy = accuracy_score(y_test_fold, y_pred_fold)
    fold_accuracies.append(fold_accuracy)
    
    print(f"Fold {fold+1} Accuracy: {fold_accuracy:.4f}")

# 计算整体准确率
overall_accuracy = accuracy_score(y, y_pred_all)
print(f"\nOverall Cross-Validation Accuracy: {overall_accuracy:.4f}")
print(f"Average Fold Accuracy: {np.mean(fold_accuracies):.4f} ± {np.std(fold_accuracies):.4f}")

# 创建混淆矩阵
cm = np.zeros((np.max(y) + 1, np.max(y) + 1), dtype=int)
for true, pred in zip(y, y_pred_all):
    cm[true, pred] += 1

# 可视化混淆矩阵
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix - 6 TOA (Cross-Validation)')
plt.savefig('confusion_matrix_cv_6toa.png')
plt.show()

# 创建一个新的DataFrame，包含原始数据
new_data = data.copy()

# 用无泄漏的预测替换第三列
new_data.iloc[:, 2] = y_pred_all

# 保存到新的Excel文件
output_path = r"D:\desktop\毕设材料\output_classifier.xlsx"
new_data.to_excel(output_path, index=False, header=False)

print(f"\n已将无泄漏预测结果保存到: {output_path}")

# 统计预测标签与真实标签的一致性
correct_predictions = sum(y_pred_all == y)
total_predictions = len(y)
print(f"\n预测正确的样本数: {correct_predictions}/{total_predictions} ({correct_predictions/total_predictions:.4%})")

# 分析每个类别的预测表现
print("\n各类别预测表现:")
for label in np.unique(y):
    mask = (y == label)
    correct = sum(y_pred_all[mask] == y[mask])
    total = sum(mask)
    print(f"标签 {label}: {correct}/{total} ({correct/total:.4%})")