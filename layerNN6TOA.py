import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error
import copy
import time

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# 1. Read data
file_path = r"D:\desktop\毕设材料\processed_data.xlsx"
data = pd.read_excel(file_path)

# Extract columns
x = data.iloc[:, 0].values  # x coordinates
y = data.iloc[:, 1].values  # y coordinates
area_labels = data.iloc[:, 2].values  # area labels (1-25)

# Use columns 10-15 (index 9-14) for 6 TOA inputs
toa_1 = data.iloc[:, 9].values   # 10th column
toa_2 = data.iloc[:, 10].values  # 11th column
toa_3 = data.iloc[:, 11].values  # 12th column
toa_4 = data.iloc[:, 12].values  # 13th column
toa_5 = data.iloc[:, 13].values  # 14th column
toa_6 = data.iloc[:, 14].values  # 15th column

# Display basic data information
print(f"Data size: {data.shape[0]} rows x {data.shape[1]} columns")
print(f"x coordinate range: [{np.min(x):.2f}, {np.max(x):.2f}]")
print(f"y coordinate range: [{np.min(y):.2f}, {np.max(y):.2f}]")
print(f"Area label range: [{np.min(area_labels)}, {np.max(area_labels)}]")

# 2. Data preprocessing
# Standardize TOA data - now using 6 TOAs
toa_data = np.column_stack((toa_1, toa_2, toa_3, toa_4, toa_5, toa_6))
scaler = StandardScaler()
toa_scaled = scaler.fit_transform(toa_data)

# One-hot encode area labels
try:
    # For newer scikit-learn versions
    encoder = OneHotEncoder(sparse_output=False)
except TypeError:
    # For older scikit-learn versions
    encoder = OneHotEncoder(sparse=False)
area_onehot = encoder.fit_transform(area_labels.reshape(-1, 1))

# Combine features
X = np.hstack((toa_scaled, area_onehot))
y_target = np.column_stack((x, y))

# Create PyTorch dataset
class TOADataset(Dataset):
    def __init__(self, features, targets):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]

# Create full dataset
full_dataset = TOADataset(X, y_target)

# Split dataset (80% train, 20% test)
train_size = int(0.8 * len(full_dataset))
test_size = len(full_dataset) - train_size
train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

# Create data loaders
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print(f"Training set size: {len(train_dataset)}")
print(f"Test set size: {len(test_dataset)}")

# 3. Define neural network model
class LocationPredictor(nn.Module):
    def __init__(self, input_size):
        super(LocationPredictor, self).__init__()
        # 增加网络深度和宽度
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 128)
        self.fc5 = nn.Linear(128, 64)
        self.fc6 = nn.Linear(64, 32)
        self.fc7 = nn.Linear(32, 2)  # 输出 x,y 坐标
        
        # 添加批归一化层
        self.bn1 = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(64)
        
        # 添加Dropout层防止过拟合
        self.dropout = nn.Dropout(0.3)
        
        # 使用不同的激活函数
        self.relu = nn.ReLU()
        self.leaky_relu = nn.LeakyReLU(0.1)
        self.elu = nn.ELU()
        
    def forward(self, x):
        # 第一层：ReLU激活
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        
        # 第二层：LeakyReLU激活
        x = self.leaky_relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        
        # 第三层：ELU激活
        x = self.elu(self.bn3(self.fc3(x)))
        x = self.dropout(x)
        
        # 第四层：ReLU激活
        x = self.relu(self.bn4(self.fc4(x)))
        
        # 第五层：LeakyReLU激活
        x = self.leaky_relu(self.bn5(self.fc5(x)))
        
        # 第六层：ReLU激活
        x = self.relu(self.fc6(x))
        
        # 输出层：无激活函数（回归问题）
        x = self.fc7(x)
        
        return x

# Create model
input_size = X.shape[1]  # input feature dimension (6 TOAs + one-hot encoded area labels)
model = LocationPredictor(input_size)
print(model)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

# 使用ReduceLROnPlateau替代StepLR，当验证损失停止下降时降低学习率
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, 
    mode='min',          # 监控指标是否应该减小
    factor=0.1,          # 学习率减少的因子
    patience=15,         # 容忍多少个epoch没有改善
    verbose=True,        # 打印学习率变化信息
    min_lr=1e-6          # 最小学习率
)

# 4. 添加早停机制和训练过程监控
def train_model(model, criterion, optimizer, scheduler, train_loader, val_loader, 
                num_epochs=300, patience=25):
    """
    Train the model with early stopping and learning rate scheduling
    
    Args:
        patience: How many epochs to wait for improvement before stopping
    """
    since = time.time()
    
    # 初始化 best_model_wts 和 best_loss
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = float('inf')
    
    # 用于早停的计数器
    no_improve_counter = 0
    
    train_losses = []
    val_losses = []
    learning_rates = []
    
    for epoch in range(num_epochs):
        # 记录当前学习率
        current_lr = optimizer.param_groups[0]['lr']
        learning_rates.append(current_lr)
        
        # 训练阶段
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
        
        epoch_train_loss = running_loss / len(train_loader)
        train_losses.append(epoch_train_loss)
        
        # 验证阶段
        model.eval()
        running_val_loss = 0.0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                running_val_loss += loss.item()
        
        epoch_val_loss = running_val_loss / len(val_loader)
        val_losses.append(epoch_val_loss)
        
        # 根据验证损失调整学习率
        scheduler.step(epoch_val_loss)
        
        # 打印训练信息
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], '
                  f'Train Loss: {epoch_train_loss:.4f}, '
                  f'Val Loss: {epoch_val_loss:.4f}, '
                  f'LR: {current_lr:.7f}')
        
        # 如果得到了更好的验证结果，就保存模型
        if epoch_val_loss < best_loss:
            best_loss = epoch_val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            no_improve_counter = 0  # 重置计数器
        else:
            no_improve_counter += 1  # 增加计数器
        
        # 如果连续多个epochs没有改善，就早停
        if no_improve_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            print(f"Best validation loss: {best_loss:.4f}")
            break
    
    # 计算训练时间
    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best validation loss: {best_loss:.4f}')
    
    # 加载最佳模型权重
    model.load_state_dict(best_model_wts)
    
    return model, train_losses, val_losses, learning_rates

# 训练模型
print("Starting neural network training (6 TOA version)...")
model, train_losses, test_losses, learning_rates = train_model(
    model, criterion, optimizer, scheduler, train_loader, test_loader
)
print("Neural network training completed")

# 5. Plot training process
plt.figure(figsize=(15, 5))

# Plot losses
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Training Loss')
plt.plot(test_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title('Training and Validation Loss (6 TOA Model)')
plt.legend()
plt.grid(True)

# Plot learning rate
plt.subplot(1, 2, 2)
plt.plot(learning_rates)
plt.xlabel('Epoch')
plt.ylabel('Learning Rate')
plt.title('Learning Rate Schedule')
plt.yscale('log')  # 使用对数尺度显示学习率
plt.grid(True)

plt.tight_layout()
plt.savefig('training_process_6toa.png')
plt.show()

# 6. Evaluate model
# Get all test data
all_test_inputs = []
all_test_targets = []
for inputs, targets in test_loader:
    all_test_inputs.append(inputs)
    all_test_targets.append(targets)

test_inputs = torch.cat(all_test_inputs)
test_targets = torch.cat(all_test_targets)

# Make predictions
model.eval()
with torch.no_grad():
    test_outputs = model(test_inputs)

# Convert to NumPy arrays for calculations and plotting
y_test = test_targets.numpy()
y_pred = test_outputs.numpy()

# Calculate error metrics on the full test set
mae = mean_absolute_error(y_test, y_pred, multioutput='raw_values')
rmse = np.sqrt(mean_squared_error(y_test, y_pred, multioutput='raw_values'))
euclidean_error = np.sqrt(np.sum((y_pred - y_test) ** 2, axis=1))
mean_euclidean_error = np.mean(euclidean_error)
max_euclidean_error = np.max(euclidean_error)

print(f'Mean Absolute Error (MAE): x={mae[0]:.4f}, y={mae[1]:.4f}')
print(f'Root Mean Square Error (RMSE): x={rmse[0]:.4f}, y={rmse[1]:.4f}')
print(f'Mean Euclidean Distance Error: {mean_euclidean_error:.4f}')
print(f'Max Euclidean Distance Error: {max_euclidean_error:.4f}')

# 计算唯一预测数
unique_preds = np.unique(y_pred, axis=0)
print(f'Number of unique predictions: {len(unique_preds)} out of {len(y_pred)} test samples')

# 7. Visualize results with only 100 samples
# Select 100 random samples for visualization
if len(y_test) > 100:
    visualization_indices = np.random.choice(len(y_test), 100, replace=False)
else:
    visualization_indices = np.arange(len(y_test))

y_test_viz = y_test[visualization_indices]
y_pred_viz = y_pred[visualization_indices]
euclidean_error_viz = euclidean_error[visualization_indices]

plt.figure(figsize=(12, 5))

# True position vs predicted position
plt.subplot(1, 2, 1)
plt.scatter(y_test_viz[:, 0], y_test_viz[:, 1], c='blue', label='True Position', alpha=0.7, s=25)
plt.scatter(y_pred_viz[:, 0], y_pred_viz[:, 1], c='red', label='Predicted Position', alpha=0.7, s=25)
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.title('Test Set: True vs Predicted Positions (100 Samples)')
plt.grid(True)
plt.legend()
plt.axis('equal')

# Error distribution histogram
plt.subplot(1, 2, 2)
plt.hist(euclidean_error, bins=20)  # Use full test set for statistics
plt.xlabel('Euclidean Distance Error')
plt.ylabel('Frequency')
plt.title('Prediction Error Distribution')
plt.grid(True)

plt.tight_layout()
plt.savefig('prediction_results_6toa.png')
plt.show()

# Error line plot
plt.figure(figsize=(8, 6))
plt.scatter(y_test_viz[:, 0], y_test_viz[:, 1], c='blue', label='True Position', alpha=0.7, s=25)
plt.scatter(y_pred_viz[:, 0], y_pred_viz[:, 1], c='red', label='Predicted Position', alpha=0.7, s=25)

# Draw error lines for each point
for i in range(len(y_test_viz)):
    plt.plot([y_test_viz[i, 0], y_pred_viz[i, 0]], [y_test_viz[i, 1], y_pred_viz[i, 1]], 'gray', alpha=0.3)

plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.title('Prediction Error Lines (100 Samples)')
plt.grid(True)
plt.legend()
plt.axis('equal')
plt.savefig('error_lines_6toa.png')
plt.show()

# Additional: Plot area label spatial distribution
# Use a subset of the original data for this visualization
sample_size = min(1000, len(x))  # Maximum 1000 points for clear visualization
sample_indices = np.random.choice(len(x), sample_size, replace=False)

plt.figure(figsize=(8, 6))
sc = plt.scatter(x[sample_indices], y[sample_indices], c=area_labels[sample_indices], cmap='jet', alpha=0.7, s=25)
plt.colorbar(sc, label='Area Label')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.title('Area Label Spatial Distribution')
plt.grid(True)
plt.axis('equal')
plt.savefig('area_labels_6toa.png')
plt.show()

# 8. Save model
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scaler': scaler,
    'encoder': encoder,
    'input_size': input_size,
    'train_losses': train_losses,
    'val_losses': test_losses
}, 'toa_model_6toa.pth')

print('Model saved as: toa_model_6toa.pth')

# 9. Example function for loading model
def load_model_and_predict(model_path, new_toa_data, new_area_labels):
    # Load saved model and parameters
    checkpoint = torch.load(model_path)
    
    # Rebuild model
    loaded_model = LocationPredictor(checkpoint['input_size'])
    loaded_model.load_state_dict(checkpoint['model_state_dict'])
    loaded_model.eval()
    
    # Extract saved data processors
    saved_scaler = checkpoint['scaler']
    saved_encoder = checkpoint['encoder']
    
    # Process new data
    new_toa_scaled = saved_scaler.transform(new_toa_data)
    new_area_onehot = saved_encoder.transform(new_area_labels.reshape(-1, 1))
    
    # Combine features
    new_X = np.hstack((new_toa_scaled, new_area_onehot))
    new_X_tensor = torch.tensor(new_X, dtype=torch.float32)
    
    # Predict
    with torch.no_grad():
        predictions = loaded_model(new_X_tensor)
    
    return predictions.numpy()

# Example: How to use the loaded model for prediction
# (For reference only, not actually executed)
"""
# Example new data (now using 6 TOA values)
new_toa_data = np.array([[toa1_value, toa2_value, toa3_value, toa4_value, toa5_value, toa6_value]])
new_area_labels = np.array([area_label_value])

# Predict new location
predicted_locations = load_model_and_predict('toa_model_6toa.pth', new_toa_data, new_area_labels)
print(f"Predicted location(x, y): {predicted_locations[0]}")
"""