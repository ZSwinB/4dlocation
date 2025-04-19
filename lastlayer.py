import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tqdm import tqdm

# 定义数据集类
class TOADataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# 定义全连接神经网络模型
class FullyConnectedModel(nn.Module):
    def __init__(self, input_size):
        super(FullyConnectedModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(64, 32)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(32, 2)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x

def load_and_process_data(file_path):
    """加载数据并进行onehot编码"""
    # 加载Excel文件，不使用表头
    df = pd.read_excel(file_path, header=None)
    
    # 提取x和y坐标（第1列和第2列）
    X_coords = df.iloc[:, 0].values
    Y_coords = df.iloc[:, 1].values
    
    # 提取标签（第3列）转为onehot编码
    try:
        # 较新版本的scikit-learn
        label_encoder = OneHotEncoder(sparse_output=False)
    except TypeError:
        # 旧版本的scikit-learn
        label_encoder = OneHotEncoder(sparse=False)
    
    label_values = df.iloc[:, 2].values.reshape(-1, 1)
    label_onehot = label_encoder.fit_transform(label_values)
    
    # 提取TOA值（第4-9列）
    toa_values = df.iloc[:, 3:9].values
    
    # 提取射线类型（第10-15列）转为onehot编码
    ray_type_values = df.iloc[:, 9:15].values
    
def process_ray_type(ray_type_values):
    """
    将射线类型转换为one-hot编码
    0 -> [1, 0, 0]
    1 -> [0, 1, 0]
    其他 -> [0, 0, 1]
    """
    # 创建空矩阵存储结果
    n_samples = ray_type_values.shape[0]
    n_ray_types = ray_type_values.shape[1]
    result = np.zeros((n_samples, n_ray_types * 3))
    
    # 处理每个射线类型
    for i in range(n_ray_types):
        # 获取当前列
        column = ray_type_values[:, i]
        
        # 为每个样本的当前射线类型创建one-hot编码
        for j in range(n_samples):
            value = column[j]
            start_idx = i * 3
            
            if value == 0:
                result[j, start_idx] = 1
            elif value == 1:
                result[j, start_idx + 1] = 1
            else:  # 2及以上的值
                result[j, start_idx + 2] = 1
    
    return result
    
    # 创建输入特征：TOA值 + 标签onehot + 射线类型onehot
    X_processed = np.hstack((toa_values, label_onehot, ray_onehot_all))
    
    # 创建目标值：x和y坐标
    y_processed = np.column_stack((X_coords, Y_coords))
    
    # 创建raw.xlsx供检查
    # 创建列名
    toa_cols = [f'TOA{i+1}' for i in range(6)]
    label_cols = [f'label_{i+1}' for i in range(label_onehot.shape[1])]
    
    ray_cols = []
    for i in range(6):
        n_classes = ray_onehot_list[i].shape[1]
        for j in range(n_classes):
            ray_cols.append(f'ray{i+1}_type{j+1}')
    
    all_cols = ['x', 'y'] + toa_cols + label_cols + ray_cols
    
    # 合并所有数据
    all_data = np.hstack((y_processed, toa_values, label_onehot, ray_onehot_all))
    raw_df = pd.DataFrame(all_data, columns=all_cols)
    
    # 保存raw.xlsx
    raw_df.to_excel('raw.xlsx', index=False)
    print("已保存raw.xlsx用于检查")
    
    return X_processed, y_processed, raw_df

def train_model(X_train, y_train, input_size, epochs=100):
    """训练模型"""
    # 创建数据加载器
    train_dataset = TOADataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    # 初始化模型
    model = FullyConnectedModel(input_size)
    
    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 训练循环
    progress_bar = tqdm(range(epochs), desc="训练进度")
    for epoch in progress_bar:
        model.train()
        running_loss = 0.0
        
        for inputs, targets in train_loader:
            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        # 更新进度条
        epoch_loss = running_loss/len(train_loader)
        progress_bar.set_postfix(loss=f"{epoch_loss:.4f}")
    
    return model

def test_model(model, X_test, y_test):
    """测试模型并可视化结果"""
    model.eval()
    print("测试模型中...")
    with torch.no_grad():
        X_tensor = torch.tensor(X_test, dtype=torch.float32)
        y_pred = model(X_tensor).numpy()
    
    # 计算损失
    mse = np.mean((y_pred - y_test) ** 2)
    print(f'测试集MSE: {mse:.4f}')
    
    # 计算欧氏距离误差
    distances = np.sqrt(np.sum((y_pred - y_test) ** 2, axis=1))
    mean_distance = np.mean(distances)
    print(f'平均定位误差: {mean_distance:.2f}')
    
    # 可视化结果
    plt.figure(figsize=(10, 8))
    plt.scatter(y_test[:, 0], y_test[:, 1], c='blue', label='原始位置', alpha=0.6)
    plt.scatter(y_pred[:, 0], y_pred[:, 1], c='red', label='预测位置', alpha=0.6)
    
    plt.legend()
    plt.xlabel('X坐标')
    plt.ylabel('Y坐标')
    plt.title('原始位置与预测位置对比')
    plt.grid(True)
    plt.savefig('position_prediction.png')
    plt.close()
    
    return y_pred, distances

def main():
    # 数据文件路径
    data_file = r"D:\desktop\毕设材料\processed_data.xlsx" # 请替换为实际文件路径
    
    # 加载和处理数据
    print("加载和处理数据...")
    try:
        X_processed, y_processed, _ = load_and_process_data(data_file)
    except Exception as e:
        print(f"处理数据时出错: {e}")
        return  # 如果数据处理失败，提前退出函数
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y_processed, test_size=0.2, random_state=42)
    
    print(f"训练集: {X_train.shape[0]} 样本, 测试集: {X_test.shape[0]} 样本")
    print(f"输入特征维度: {X_train.shape[1]}")
    
    # 训练模型
    print("\n开始训练模型...")
    input_size = X_train.shape[1]
    model = train_model(X_train, y_train, input_size, epochs=200)
    
    # 保存模型
    torch.save(model.state_dict(), 'toa_model.pth')
    print("模型已保存为 toa_model.pth")
    
    # 显示模型结构
    print("\n模型结构:")
    print(model)
    
    # 测试模型
    print("\n开始测试模型...")
    y_pred, distances = test_model(model, X_test, y_test)
    
    # 显示一些测试结果样例
    print("\n测试结果样例 (前5个):")
    for i in range(min(5, len(y_test))):
        print(f"样本 {i+1}: 原始坐标 ({y_test[i][0]:.2f}, {y_test[i][1]:.2f}), "
              f"预测坐标 ({y_pred[i][0]:.2f}, {y_pred[i][1]:.2f}), "
              f"误差: {distances[i]:.2f}")
    print("位置预测图已保存为 position_prediction.png")

if __name__ == "__main__":
    main()