import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# 加载6个TOA的数据
file_path_6toa = r"D:\desktop\毕设材料\processed_data.xlsx"
data_6toa = pd.read_excel(file_path_6toa)

# 加载3个TOA的数据
file_path_3toa = r"D:\desktop\毕设材料\data4trainning.xlsx"
data_3toa = pd.read_excel(file_path_3toa)

# 提取特征和标签 - 6个TOA
X_6toa = data_6toa.iloc[:, 3:9].values  # 第4到9列作为特征
y_6toa = data_6toa.iloc[:, 2].values    # 第3列作为标签

# 提取特征和标签 - 3个TOA
X_3toa = data_3toa.iloc[:, 3:6].values  # 第4到6列作为特征
y_3toa = data_3toa.iloc[:, 2].values    # 第3列作为标签

# 将数据集分为训练集和测试集，测试集大小为20%
X_train_6toa, X_test_6toa, y_train_6toa, y_test_6toa = train_test_split(X_6toa, y_6toa, test_size=0.2, random_state=42)
X_train_3toa, X_test_3toa, y_train_3toa, y_test_3toa = train_test_split(X_3toa, y_3toa, test_size=0.2, random_state=42)

# 创建随机森林分类器
rf_6toa = RandomForestClassifier(n_estimators=100, random_state=42)
rf_3toa = RandomForestClassifier(n_estimators=100, random_state=42)

# 训练分类器
rf_6toa.fit(X_train_6toa, y_train_6toa)
rf_3toa.fit(X_train_3toa, y_train_3toa)

# 在测试集上预测
y_pred_6toa = rf_6toa.predict(X_test_6toa)
y_pred_3toa = rf_3toa.predict(X_test_3toa)

# 计算准确率
accuracy_6toa = accuracy_score(y_test_6toa, y_pred_6toa)
accuracy_3toa = accuracy_score(y_test_3toa, y_pred_3toa)

print(f"6个TOA准确率: {accuracy_6toa:.4f}")
print(f"3个TOA准确率: {accuracy_3toa:.4f}")

# 交叉验证评估
cv_scores_6toa = cross_val_score(rf_6toa, X_6toa, y_6toa, cv=5)
cv_scores_3toa = cross_val_score(rf_3toa, X_3toa, y_3toa, cv=5)

print("\n5折交叉验证结果:")
print(f"6个TOA交叉验证准确率: {cv_scores_6toa.mean():.4f} ± {cv_scores_6toa.std():.4f}")
print(f"3个TOA交叉验证准确率: {cv_scores_3toa.mean():.4f} ± {cv_scores_3toa.std():.4f}")

# 比较两种方法的混淆矩阵
plt.figure(figsize=(16, 7))

plt.subplot(1, 2, 1)
sns.heatmap(confusion_matrix(y_test_6toa, y_pred_6toa), annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix - 6 TOA')

plt.subplot(1, 2, 2)
sns.heatmap(confusion_matrix(y_test_3toa, y_pred_3toa), annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix - 3 TOA')

plt.tight_layout()
plt.savefig('confusion_matrix_comparison.png')
plt.show()

# 比较特征重要性
plt.figure(figsize=(16, 7))

plt.subplot(1, 2, 1)
importance_6toa = rf_6toa.feature_importances_
indices_6toa = np.argsort(importance_6toa)[::-1]
plt.bar(range(X_6toa.shape[1]), importance_6toa[indices_6toa], align='center')
plt.xticks(range(X_6toa.shape[1]), ['TOA'+str(i+1) for i in indices_6toa])
plt.title('Feature Importance - 6 TOA')

plt.subplot(1, 2, 2)
importance_3toa = rf_3toa.feature_importances_
indices_3toa = np.argsort(importance_3toa)[::-1]
plt.bar(range(X_3toa.shape[1]), importance_3toa[indices_3toa], align='center')
plt.xticks(range(X_3toa.shape[1]), ['TOA'+str(i+1) for i in indices_3toa])
plt.title('Feature Importance - 3 TOA')

plt.tight_layout()
plt.savefig('feature_importance_comparison.png')
plt.show()

# 输出详细的分类报告
print("\n6个TOA分类报告:")
print(classification_report(y_test_6toa, y_pred_6toa))

print("\n3个TOA分类报告:")
print(classification_report(y_test_3toa, y_pred_3toa))