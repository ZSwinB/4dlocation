import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch_geometric.nn as geo_nn
from torch_geometric.data import Data, Batch
import torch.nn.functional as F

# 设置随机种子确保结果可复现
np.random.seed(42)
torch.manual_seed(42)

# 1. 数据加载和预处理函数
def load_and_preprocess_data(file_path, test_size=0.2):
    """
    加载Excel数据并进行预处理
    """
    # 加载数据
    df = pd.read_excel(file_path)
    
    # 假设列名按照描述的顺序排列
    df.columns = ['x', 'y', 'label'] + [f'TOA{i}' for i in range(1, 7)] + [f'TOA{i}_ray_type' for i in range(1, 7)]
    
    # 分离特征和目标变量
    X = df.drop(['x', 'y'], axis=1)
    y = df[['x', 'y']]
    
    # 转换标签为one-hot编码（假设标签是从0开始的连续自然数，共25个类别）
    label_encoder = OneHotEncoder(sparse_output=False, categories=[range(25)])
    labels_onehot = label_encoder.fit_transform(X[['label']])
    
    # 转换射线类型为one-hot编码（0, 1, 其他归为一类）
    ray_types = X[[f'TOA{i}_ray_type' for i in range(1, 7)]].values
    ray_types_mapped = np.zeros_like(ray_types)
    ray_types_mapped[ray_types == 0] = 0
    ray_types_mapped[ray_types == 1] = 1
    ray_types_mapped[(ray_types != 0) & (ray_types != 1)] = 2
    
    ray_encoder = OneHotEncoder(sparse_output=False, categories=[range(3)] * 6)
    ray_types_reshaped = ray_types_mapped.reshape(-1, 1)
    ray_types_onehot = ray_encoder.fit_transform(ray_types_reshaped)
    ray_types_onehot = ray_types_onehot.reshape(-1, 6, 3)
    
    # 准备数据
    toa_values = X[[f'TOA{i}' for i in range(1, 7)]].values
    original_labels = X['label'].values
    original_ray_types = X[[f'TOA{i}_ray_type' for i in range(1, 7)]].values
    
    # 将转换后的数据保存到Excel以供检查
    export_encoded_data_to_excel(df, labels_onehot, ray_types_onehot, 'encoded_data_check.xlsx')
    
    # 分割训练集和测试集
    X_train, X_test, y_train, y_test, labels_train, labels_test, ray_types_train, ray_types_test, orig_labels_train, orig_labels_test = train_test_split(
        toa_values, y.values, labels_onehot, ray_types_onehot, original_labels, test_size=test_size, random_state=42
    )
    
    return (X_train, X_test, y_train, y_test, labels_train, labels_test, ray_types_train, ray_types_test, 
            orig_labels_train, orig_labels_test, label_encoder, ray_encoder)

def export_encoded_data_to_excel(df, labels_onehot, ray_types_onehot, output_file):
    """
    将原始数据和转换后的one-hot编码保存到Excel
    """
    # 创建一个新的DataFrame来存储结果
    result_df = pd.DataFrame()
    
    # 添加原始x,y坐标
    result_df['x'] = df['x']
    result_df['y'] = df['y']
    
    # 添加原始标签
    result_df['original_label'] = df['label']
    
    # 添加one-hot编码的标签
    for i in range(labels_onehot.shape[1]):
        result_df[f'label_onehot_{i}'] = labels_onehot[:, i]
    
    # 添加原始TOA和射线类型
    for i in range(1, 7):
        result_df[f'TOA{i}'] = df[f'TOA{i}']
        result_df[f'TOA{i}_original_ray_type'] = df[f'TOA{i}_ray_type']
    
    # 添加one-hot编码的射线类型
    for i in range(6):
        for j in range(3):
            result_df[f'TOA{i+1}_ray_type_onehot_{j}'] = ray_types_onehot[:, i, j]
    
    # 保存到Excel
    result_df.to_excel(output_file, index=False)
    print(f"编码数据已保存到 {output_file}")

# 2. 自定义数据集类
class TOADataset(Dataset):
    def __init__(self, toa_values, labels, ray_types, orig_labels=None, targets=None):
        self.toa_values = torch.FloatTensor(toa_values)
        self.labels = torch.FloatTensor(labels)
        self.ray_types = torch.FloatTensor(ray_types)
        self.orig_labels = orig_labels  # 用于随机森林
        self.targets = torch.FloatTensor(targets) if targets is not None else None
        
    def __len__(self):
        return len(self.toa_values)
    
    def __getitem__(self, idx):
        if self.targets is not None:
            if self.orig_labels is not None:
                return self.toa_values[idx], self.labels[idx], self.ray_types[idx], self.orig_labels[idx], self.targets[idx]
            else:
                return self.toa_values[idx], self.labels[idx], self.ray_types[idx], self.targets[idx]
        else:
            if self.orig_labels is not None:
                return self.toa_values[idx], self.labels[idx], self.ray_types[idx], self.orig_labels[idx]
            else:
                return self.toa_values[idx], self.labels[idx], self.ray_types[idx]

# 3. 随机森林区域分类器
class RegionRandomForest:
    def __init__(self, n_estimators=100, max_depth=None):
        self.model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    
    def train(self, X, y):
        """
        训练随机森林模型
        X: TOA值 [n_samples, 6]
        y: 区域标签索引 [n_samples]
        """
        self.model.fit(X, y)
    
    def predict(self, X):
        """
        预测区域标签
        """
        # 预测类别
        y_pred = self.model.predict(X)
        return y_pred
    
    def predict_proba(self, X):
        """
        预测区域标签概率
        """
        # 预测概率
        y_proba = self.model.predict_proba(X)
        return y_proba

# 4. 图神经网络射线类型分类器
class RayTypeGNN(nn.Module):
    def __init__(self, toa_dim=6, hidden_dim=64, output_dim=3):
        super(RayTypeGNN, self).__init__()
        
        # GNN层
        self.conv1 = geo_nn.GCNConv(toa_dim + 25, hidden_dim)  # 输入: TOA特征 + 区域标签
        self.conv2 = geo_nn.GCNConv(hidden_dim, hidden_dim)
        
        # 预测层
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x, edge_index, batch=None):
        # 第一层GNN卷积
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.3, training=self.training)
        
        # 第二层GNN卷积
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        
        # 预测射线类型
        x = self.fc(x)
        return F.log_softmax(x, dim=1)

# 构建图数据的辅助函数
def build_graph_data(toa_values, region_labels, ray_types=None, batch_size=32):
    """
    为GNN构建图数据
    - toa_values: [batch_size, 6] TOA值
    - region_labels: [batch_size, 25] 区域标签的one-hot编码
    - ray_types: [batch_size, 6, 3] 射线类型的one-hot编码 (训练时提供)
    """
    data_list = []
    
    for i in range(len(toa_values)):
        # 为每个接收机创建节点特征 (TOA + 区域标签)
        x = torch.cat([
            toa_values[i].repeat(6, 1),  # 复制TOA作为每个节点的特征
            region_labels[i].repeat(6, 1)  # 复制区域标签
        ], dim=1)
        
        # 创建完全连接的边
        edge_index = []
        for j in range(6):
            for k in range(6):
                if j != k:  # 排除自环
                    edge_index.append([j, k])
        edge_index = torch.tensor(edge_index, dtype=torch.long).t()
        
        # 如果提供了射线类型，添加作为标签
        if ray_types is not None:
            # 将ray_types的one-hot编码转换为类别索引
            y = torch.argmax(ray_types[i], dim=1)
        else:
            y = None
            
        data = Data(x=x, edge_index=edge_index, y=y)
        data_list.append(data)
    
    # 将图数据批量处理
    batch = Batch.from_data_list(data_list)
    return batch

# 5. 位置预测模型（最终层）
class PositionPredictor(nn.Module):
    def __init__(self, input_size=49):  # 6 (TOA) + 25 (区域标签) + 18 (射线类型)
        super(PositionPredictor, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 2)  # x, y 坐标
        
    def forward(self, toa, region_label, ray_types):
        # 处理射线类型输入
        ray_types_flat = ray_types.reshape(ray_types.shape[0], -1)  # [batch_size, 6*3]
        
        # 合并所有输入
        x = torch.cat([toa, region_label, ray_types_flat], dim=1)
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

# 6. 训练和评估函数
def train_region_classifier(X_train, orig_labels_train, X_val, orig_labels_val):
    """
    训练随机森林区域分类器
    """
    rf_model = RegionRandomForest(n_estimators=100)
    rf_model.train(X_train, orig_labels_train)
    
    # 评估训练集和验证集精度
    train_preds = rf_model.predict(X_train)
    val_preds = rf_model.predict(X_val)
    
    train_accuracy = np.sum(train_preds == orig_labels_train) / len(orig_labels_train)
    val_accuracy = np.sum(val_preds == orig_labels_val) / len(orig_labels_val)
    
    print(f"区域分类器 - 训练集精度: {train_accuracy:.4f}, 验证集精度: {val_accuracy:.4f}")
    
    return rf_model

def train_ray_classifier(gnn_model, region_model, label_encoder, X_train, labels_train, ray_types_train, 
                        X_val, labels_val, ray_types_val, epochs=50, lr=0.001):
    """
    训练GNN射线类型分类器
    """
    optimizer = optim.Adam(gnn_model.parameters(), lr=lr)
    criterion = nn.NLLLoss()
    
    train_losses, val_losses = [], []
    
    for epoch in range(epochs):
        gnn_model.train()
        train_loss = 0.0
        
        # 由于GNN不需要批处理，我们可以一次性处理整个训练集
        # 首先获得区域标签预测
        region_probs_train = label_encoder.transform(region_model.predict(X_train).reshape(-1, 1))
        region_probs_val = label_encoder.transform(region_model.predict(X_val).reshape(-1, 1))
        
        # 将预测结果转换为图数据
        train_graph = build_graph_data(torch.FloatTensor(X_train), 
                                       torch.FloatTensor(region_probs_train), 
                                       torch.FloatTensor(ray_types_train))
        
        optimizer.zero_grad()
        output = gnn_model(train_graph.x, train_graph.edge_index, train_graph.batch)
        loss = criterion(output, train_graph.y)
        loss.backward()
        optimizer.step()
        
        train_loss = loss.item()
        
        # 验证
        gnn_model.eval()
        with torch.no_grad():
            val_graph = build_graph_data(torch.FloatTensor(X_val), 
                                         torch.FloatTensor(region_probs_val), 
                                         torch.FloatTensor(ray_types_val))
            
            output = gnn_model(val_graph.x, val_graph.edge_index, val_graph.batch)
            val_loss = criterion(output, val_graph.y).item()
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
    
    return train_losses, val_losses

def train_position_predictor(model, region_model, ray_model, label_encoder, X_train, labels_train, ray_types_train, y_train,
                           X_val, labels_val, ray_types_val, y_val, epochs=50, lr=0.001):
    """
    训练位置预测模型
    """
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    train_losses, val_losses = [], []
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        # 获取区域标签预测
        region_probs_train = label_encoder.transform(region_model.predict(X_train).reshape(-1, 1))
        region_probs_val = label_encoder.transform(region_model.predict(X_val).reshape(-1, 1))
        
        # 获取射线类型预测
        train_graph = build_graph_data(torch.FloatTensor(X_train), torch.FloatTensor(region_probs_train))
        val_graph = build_graph_data(torch.FloatTensor(X_val), torch.FloatTensor(region_probs_val))
        
        ray_model.eval()
        with torch.no_grad():
            train_ray_output = ray_model(train_graph.x, train_graph.edge_index, train_graph.batch)
            val_ray_output = ray_model(val_graph.x, val_graph.edge_index, val_graph.batch)
        
        # 重塑输出为 [batch_size, 6, 3]
        train_ray_probs = F.softmax(train_ray_output, dim=1).view(-1, 6, 3)
        val_ray_probs = F.softmax(val_ray_output, dim=1).view(-1, 6, 3)
        
        # 训练位置预测器
        for i in range(0, len(X_train), 32):  # 批处理
            batch_end = min(i + 32, len(X_train))
            batch_toa = torch.FloatTensor(X_train[i:batch_end])
            batch_region = torch.FloatTensor(region_probs_train[i:batch_end])
            batch_ray = train_ray_probs[i:batch_end]
            batch_target = torch.FloatTensor(y_train[i:batch_end])
            
            optimizer.zero_grad()
            pred = model(batch_toa, batch_region, batch_ray)
            loss = criterion(pred, batch_target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * (batch_end - i)
        
        train_loss /= len(X_train)
        
        # 验证
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for i in range(0, len(X_val), 32):
                batch_end = min(i + 32, len(X_val))
                batch_toa = torch.FloatTensor(X_val[i:batch_end])
                batch_region = torch.FloatTensor(region_probs_val[i:batch_end])
                batch_ray = val_ray_probs[i:batch_end]
                batch_target = torch.FloatTensor(y_val[i:batch_end])
                
                pred = model(batch_toa, batch_region, batch_ray)
                loss = criterion(pred, batch_target)
                val_loss += loss.item() * (batch_end - i)
            
            val_loss /= len(X_val)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
    
    return train_losses, val_losses

def evaluate_model(region_model, ray_model, position_model, label_encoder, X_test, y_test):
    """
    评估完整模型
    """
    # 获取区域标签预测
    region_probs = label_encoder.transform(region_model.predict(X_test).reshape(-1, 1))
    
    # 获取射线类型预测
    test_graph = build_graph_data(torch.FloatTensor(X_test), torch.FloatTensor(region_probs))
    
    ray_model.eval()
    with torch.no_grad():
        ray_output = ray_model(test_graph.x, test_graph.edge_index, test_graph.batch)
    
    # 重塑输出为 [batch_size, 6, 3]
    ray_probs = F.softmax(ray_output, dim=1).view(-1, 6, 3)
    
    # 预测位置
    position_model.eval()
    predictions = []
    
    with torch.no_grad():
        for i in range(0, len(X_test), 32):
            batch_end = min(i + 32, len(X_test))
            batch_toa = torch.FloatTensor(X_test[i:batch_end])
            batch_region = torch.FloatTensor(region_probs[i:batch_end])
            batch_ray = ray_probs[i:batch_end]
            
            pred = position_model(batch_toa, batch_region, batch_ray)
            predictions.append(pred.numpy())
    
    predictions = np.vstack(predictions)
    
    # 计算MSE和距离误差
    mse = np.mean((predictions - y_test) ** 2)
    distance_error = np.sqrt(np.sum((predictions - y_test) ** 2, axis=1))
    mean_distance_error = np.mean(distance_error)
    
    print(f"测试集MSE: {mse:.4f}")
    print(f"平均距离误差: {mean_distance_error:.4f} 单位")
    
    return predictions, y_test, distance_error

def plot_results(predictions, targets):
    plt.figure(figsize=(10, 8))
    plt.scatter(targets[:, 0], targets[:, 1], c='blue', label='真实位置', alpha=0.5)
    plt.scatter(predictions[:, 0], predictions[:, 1], c='red', label='预测位置', alpha=0.5)
    
    # 连接相应的点以显示误差
    for i in range(len(predictions)):
        plt.plot([targets[i, 0], predictions[i, 0]], [targets[i, 1], predictions[i, 1]], 'k-', alpha=0.2)
    
    plt.legend()
    plt.title('定位结果比较')
    plt.xlabel('X 坐标')
    plt.ylabel('Y 坐标')
    plt.grid(True)
    plt.axis('equal')
    plt.savefig('positioning_results.png')
    plt.show()

# 7. 主函数
def main(file_path):
    # 加载和预处理数据
    data = load_and_preprocess_data(file_path)
    X_train, X_test, y_train, y_test, labels_train, labels_test, ray_types_train, ray_types_test, orig_labels_train, orig_labels_test, label_encoder, ray_encoder = data
    
    # 分割训练集为训练和验证
    val_size = int(0.2 * len(X_train))
    train_idx = np.arange(len(X_train))
    np.random.shuffle(train_idx)
    train_idx, val_idx = train_idx[val_size:], train_idx[:val_size]
    
    X_val = X_train[val_idx]
    y_val = y_train[val_idx]
    labels_val = labels_train[val_idx]
    ray_types_val = ray_types_train[val_idx]
    orig_labels_val = orig_labels_train[val_idx]
    
    X_train = X_train[train_idx]
    y_train = y_train[train_idx]
    labels_train = labels_train[train_idx]
    ray_types_train = ray_types_train[train_idx]
    orig_labels_train = orig_labels_train[train_idx]
    
    # 1. 训练随机森林区域分类器
    print("\n训练随机森林区域分类器...")
    region_model = train_region_classifier(X_train, orig_labels_train, X_val, orig_labels_val)
    
    # 2. 训练GNN射线类型分类器
    print("\n训练GNN射线类型分类器...")
    ray_model = RayTypeGNN()
    train_ray_classifier(ray_model, region_model, label_encoder, X_train, labels_train, ray_types_train, 
                         X_val, labels_val, ray_types_val, epochs=50)
    
    # 3. 训练位置预测器
    print("\n训练位置预测器...")
    position_model = PositionPredictor()
    train_position_predictor(position_model, region_model, ray_model, label_encoder, 
                            X_train, labels_train, ray_types_train, y_train,
                            X_val, labels_val, ray_types_val, y_val, epochs=100)
    
    # 评估模型
    print("\n评估模型...")
    predictions, targets, distance_error = evaluate_model(region_model, ray_model, position_model, 
                                                         label_encoder, X_test, y_test)
    
    # 可视化结果
    plot_results(predictions, targets)
    
    print("模型训练和评估完成！")

if __name__ == "__main__":
    # 替换为你的Excel文件路径
    file_path = r"F:\RT1.4use_thesis _highway\Examples\HSRexamples\resultscity\scene1\result1-10\processed_data.xlsx"
    main(file_path)