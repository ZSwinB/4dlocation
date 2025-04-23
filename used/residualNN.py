import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, LeakyReLU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import Activation
from sklearn.preprocessing import OneHotEncoder

import os
from scipy.optimize import least_squares

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Data file path
filtered_data_path = r"D:\desktop\毕设材料\6\classifier_noisy_filtered.xlsx"

# Receiver positions - same as in the filtering code
receivers = np.array([
    [0, 0, 0],      # Receiver 1 position
    [181, 528, 2],  # Receiver 2 position
    [277, 304, 4],  # Receiver 3 position
    [413, 228, 6],  # Receiver 4 position
    [572, 324, 8],  # Receiver 5 position
    [466, 70, 10],  # Receiver 6 position
])

# Speed of light (m/s)
c = 299792458

# Define TOA positioning functions (physical model)
def calculate_toas(emitter_pos, receiver_positions):
    """Calculate theoretical TOA values from emitter to all receivers"""
    emitter_pos = np.array(emitter_pos).flatten()
    distances = []
    for receiver_pos in receiver_positions:
        distance = np.sqrt(np.sum((receiver_pos - emitter_pos)**2))
        distances.append(distance)
    
    toas = np.array(distances) / c
    return toas

def label_to_center(label, region_size=120, base_height=80):
    label -= 1  # 从0开始
    row = label // 5
    col = label % 5
    center_x = (col + 0.5) * region_size
    center_y = (row + 0.5) * region_size
    
    # 添加随机高度误差
    height_error = np.random.uniform(0, 0.5)  # 0到3米的随机误差
    height = base_height + height_error
    
    return np.array([center_x, center_y, height])

def estimate_position(toa_values, receiver_indices, receiver_positions, label=None):
    """Estimate emitter position using three receivers' TOA"""
    selected_receivers = np.array([receiver_positions[i-1] for i in receiver_indices])  # Receiver IDs are 1-6
    selected_toas = np.array(toa_values)
    #label=None
    # 使用标签中心作为初始猜测
    if label is not None and 1 <= label <= 25:
        initial_guess = label_to_center(label)
    else:
        initial_guess = np.mean(selected_receivers, axis=0)
    
    def residuals(pos):
        calculated_toas = calculate_toas(pos, selected_receivers)
        return calculated_toas - selected_toas
    
    result = least_squares(residuals, initial_guess, method='lm')
    
    return result.x

# 1. Load data (no header)
print("Loading filtered data...")
try:
    # Load data without column names
    df = pd.read_excel(filtered_data_path, header=None)
    print(f"Loaded data shape: {df.shape}")
    print("First 5 rows of data:")
    print(df.head())
except Exception as e:
    print(f"Error loading data: {e}")
    raise

# 2. Parse data according to the correct format
print("\nParsing data...")
# Correct column structure:
# 0-1: x, y (true emitter position)
# 2: label
# 3-5: three TOA values
# 6-8: three TOA ray types
# 9-11: three receiver IDs

# Extract required data
true_positions = df.iloc[:, 0:2].values  # x, y true positions
toa_values = df.iloc[:, 3:6].values      # three TOA values

label_raw = df.iloc[:, 2].values.reshape(-1, 1)  # Shape: (N, 1)

encoder = OneHotEncoder(sparse_output=False)
label_onehot = encoder.fit_transform(label_raw)  # Shape: (N, num_classes)
print(f"One-hot encoded labels shape: {label_onehot.shape}")  

ray_types = df.iloc[:, 6:9].values       # three reflection orders
receiver_ids = df.iloc[:, 9:12].values.astype(int)  # three receiver IDs (1-6)

# 3. Calculate estimated positions using the physical model
print("Calculating physical model estimated positions...")
estimated_positions = []

for i in range(len(df)):
    try:
        label_for_row = int(label_raw[i][0])
        # Apply physical model to each row
        est_pos = estimate_position(
            toa_values[i], 
            receiver_ids[i], 
            receivers,
            label=label_for_row
            
        )
        estimated_positions.append(est_pos[:2])  # Only keep x,y coordinates
    except Exception as e:
        print(f"Error processing row {i}: {e}")
        # Use (0,0) as estimated position if calculation fails
        estimated_positions.append(np.array([0, 0]))

estimated_positions = np.array(estimated_positions)

# 4. Calculate residuals
print("Calculating position residuals...")
residuals = true_positions - estimated_positions
print(f"Residual statistics: mean={np.mean(residuals, axis=0)}, std={np.std(residuals, axis=0)}")
print(f"Residual range: min={np.min(residuals, axis=0)}, max={np.max(residuals, axis=0)}")

# Check and filter outliers in residuals
residual_distances = np.sqrt(np.sum(residuals**2, axis=1))
q1, q3 = np.percentile(residual_distances, [25, 75])
iqr = q3 - q1
upper_bound = q3 + 1.5 * iqr

# Identify non-outlier indices
valid_indices = np.where(residual_distances <= upper_bound)[0]
print(f"Filtered {len(df) - len(valid_indices)} outliers from {len(df)} samples")

# Filter data to remove outliers
true_positions_filtered = true_positions[valid_indices]
estimated_positions_filtered = estimated_positions[valid_indices]
toa_values_filtered = toa_values[valid_indices]
ray_types_filtered = ray_types[valid_indices]
label_onehot_filtered = label_onehot[valid_indices]
receiver_ids_filtered = receiver_ids[valid_indices]
residuals_filtered = residuals[valid_indices]
df_filtered = df.iloc[valid_indices]

# 5. Prepare neural network input features
print("\nPreparing neural network input features...")
# Combine TOA values, reflection orders, receiver positions, 
# and physical model estimates as input
features = []
for i in range(len(toa_values_filtered)):
    row_features = []
    # Add 3 TOA values
    row_features.extend(toa_values_filtered[i])
    #row_features.extend(label_onehot_filtered[i]) # Add label
    # Add 3 reflection orders
    #row_features.extend(ray_types_filtered[i])
    # Add 3 receiver positions (x, y, z)
    for j in range(3):
        receiver_idx = receiver_ids_filtered[i, j] - 1  # Convert to 0-5 index
        row_features.extend(receivers[receiver_idx])
    # Add physical model estimate (important additional information)
    row_features.extend(estimated_positions_filtered[i])
    features.append(row_features)

features = np.array(features)
print(f"Feature dimensions: {features.shape}")

# 6. Split into training and test sets
print("\nSplitting into training and test sets...")
X_train, X_test, y_train, y_test = train_test_split(
    features, residuals_filtered, test_size=0.2, random_state=42
)
print(f"Training set size: {X_train.shape[0]}, Test set size: {X_test.shape[0]}")

# Also split the true positions and estimated positions for later evaluation
train_indices, test_indices = train_test_split(
    np.arange(len(features)), test_size=0.2, random_state=42
)
true_positions_test = true_positions_filtered[test_indices]
estimated_positions_test = estimated_positions_filtered[test_indices]

# Standardize data
print("Standardizing features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 7. Build a more complex residual learning neural network
print("\nBuilding improved residual learning neural network...")

model = Sequential([
    # Input layer
    Dense(512, input_shape=(X_train_scaled.shape[1],)),
    LeakyReLU(alpha=0.1),
    BatchNormalization(),
    Dropout(0.3),

 



    Dense(256),
    LeakyReLU(alpha=0.1),
    BatchNormalization(),
    Dropout(0.3),
    
    
    # Hidden layers
    Dense(128),
    LeakyReLU(alpha=0.1),
    BatchNormalization(),
    Dropout(0.3),
    
    Dense(64),
    LeakyReLU(alpha=0.1),
    BatchNormalization(),
    Dropout(0.2),
    
    Dense(32),
    LeakyReLU(alpha=0.1),
    BatchNormalization(),
    
    # Output layer
    Dense(2)  # Output x and y residuals
])

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='mse',
    metrics=['mae']
)
'''
model = Sequential([
    # 第一层：ReLU激活
    Dense(256, input_shape=(X_train_scaled.shape[1],)),
    Activation('relu'),
    BatchNormalization(),
    Dropout(0.3),
    
    # 第二层：LeakyReLU激活
    Dense(256),
    LeakyReLU(alpha=0.1),
    BatchNormalization(),
    Dropout(0.3),
    
    # 第三层：ELU激活
    Dense(128),
    Activation('elu'),
    BatchNormalization(),
    Dropout(0.3),
    
    # 第四层：ReLU激活
    Dense(128),
    Activation('relu'),
    BatchNormalization(),
    
    # 第五层：LeakyReLU激活
    Dense(64),
    LeakyReLU(alpha=0.1),
    BatchNormalization(),
    
    # 第六层：ReLU激活
    Dense(32),
    Activation('relu'),
    
    # 输出层：无激活函数（回归问题）
    Dense(2)
])

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='mse',
    metrics=['mae']
)
'''
print(model.summary())

# 8. Set up callbacks for early stopping and learning rate reduction
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=20,
    verbose=1,
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=10,
    verbose=1,
    min_lr=0.00001
)

callbacks = [early_stopping, reduce_lr]

# 9. Train the model
print("\nTraining the model with early stopping and learning rate reduction...")
history = model.fit(
    X_train_scaled, y_train,
    epochs=500,  # Maximum epochs (early stopping will likely trigger before this)
    batch_size=32,
    validation_split=0.2,
    callbacks=callbacks,
    verbose=2
)

# 10. Evaluate the model
print("\nEvaluating the model...")
train_loss, train_mae = model.evaluate(X_train_scaled, y_train, verbose=0)
test_loss, test_mae = model.evaluate(X_test_scaled, y_test, verbose=0)

print(f"Training set - MSE: {train_loss:.4f}, MAE: {train_mae:.4f}")
print(f"Test set - MSE: {test_loss:.4f}, MAE: {test_mae:.4f}")

# 11. Make predictions on the test set
print("\nMaking predictions on the test set...")
y_pred = model.predict(X_test_scaled)

# Calculate physical model errors (Euclidean distance)
physics_errors = np.sqrt(np.sum((true_positions_test - estimated_positions_test)**2, axis=1))

# Calculate position predictions with residual learning
predicted_positions = estimated_positions_test + y_pred

# Calculate residual learning errors (Euclidean distance)
residual_errors = np.sqrt(np.sum((true_positions_test - predicted_positions)**2, axis=1))

# Calculate error improvement percentage
improvement = (physics_errors - residual_errors) / physics_errors * 100
avg_improvement = np.mean(improvement)

print(f"Physical model average error: {np.mean(physics_errors):.2f} meters")
print(f"Residual learning average error: {np.mean(residual_errors):.2f} meters")
print(f"Average improvement: {avg_improvement:.2f}%")

# 12. Generate residual Excel (label, x residual, y residual)
print("\nGenerating residual Excel...")
# Get labels for test set
labels_test = df_filtered.iloc[test_indices, 2].values

# Create results DataFrame
results_df = pd.DataFrame({
    'Label': labels_test,
    'Residual_X': y_pred[:, 0],
    'Residual_Y': y_pred[:, 1]
})

# Save to Excel file
residual_path = "D:/desktop/毕设材料/residuals_output.xlsx"
results_df.to_excel(residual_path, index=False)
print(f"Residuals saved to: {residual_path}")

# 13. Create visualizations
print("\nGenerating plots...")
plt.figure(figsize=(12, 8))

# Plot 1: Error distribution histogram with clear colors
plt.subplot(2, 2, 1)
# 手动设置 bin：0 到 35，步长为 1，共 35 个柱子
bins = np.arange(0, 72, 2)  # [0, 1, 2, ..., 35]

# 画图
plt.subplot(2, 2, 1)
plt.hist(physics_errors, bins=bins, alpha=0.7, label='Physical Model', color='blue')
plt.hist(residual_errors, bins=bins, alpha=0.7, label='Residual Learning', color='orange')

plt.xlabel('Positioning Error (m)')
plt.ylabel('Frequency')
plt.title('Error Distribution Comparison (Capped at 35m)')
plt.legend()
plt.grid(True, alpha=0.3)

plt.xlabel('Positioning Error (m)')
plt.ylabel('Frequency')
plt.title('Error Distribution Comparison')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 2: Scatter plot of actual vs predicted positions
plt.subplot(2, 2, 2)
# Randomly select 100 samples (or all if less than 100)
sample_size = min(100, len(true_positions_test))
random_indices = np.random.choice(len(true_positions_test), sample_size, replace=False)

# Plot actual positions
plt.scatter(true_positions_test[random_indices, 0], true_positions_test[random_indices, 1], 
           color='blue', label='Actual Position', alpha=0.7)
# Plot final predicted positions only
plt.scatter(predicted_positions[random_indices, 0], predicted_positions[random_indices, 1], 
           color='green', label='Predicted Position', alpha=0.7)

# Draw lines connecting actual and predicted positions
for i in random_indices:
    plt.plot([predicted_positions[i, 0], true_positions_test[i, 0]],
             [predicted_positions[i, 1], true_positions_test[i, 1]],
             'g-', alpha=0.3)

plt.xlabel('X Coordinate (m)')
plt.ylabel('Y Coordinate (m)')
plt.title('Position Prediction Scatter Plot (100 Random Samples)')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 3: Improvement percentage histogram
plt.subplot(2, 2, 3)
plt.hist(improvement, bins=30, color='green')
plt.axvline(x=0, color='red', linestyle='--')  # Line at 0% improvement
plt.xlabel('Error Improvement Percentage (%)')
plt.ylabel('Frequency')
plt.title(f'Residual Learning Improvement (Average: {avg_improvement:.2f}%)')
plt.grid(True, alpha=0.3)

# Plot 4: Training history
plt.subplot(2, 2, 4)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss (MSE)')
plt.title('Training History')
plt.legend()
plt.grid(True, alpha=0.3)
plt.yscale('log')  # Log scale for better visualization

plt.tight_layout()
plt.savefig("D:/desktop/毕设材料/residual_learning_results.png", dpi=300)
plt.show()  # Display the figure on screen
print("Main visualization completed.")

# Detailed individual sample comparison
plt.figure(figsize=(15, 12))

# Select 16 samples from the same random selection
sample_indices = random_indices[:16] if len(random_indices) >= 16 else random_indices

for i, idx in enumerate(sample_indices):
    plt.subplot(4, 4, i+1)
    
    # Plot actual and predicted positions
    plt.scatter(true_positions_test[idx, 0], true_positions_test[idx, 1], 
               color='blue', label='Actual' if i==0 else "", s=100, marker='o')
    plt.scatter(predicted_positions[idx, 0], predicted_positions[idx, 1], 
               color='green', label='Predicted' if i==0 else "", s=80, marker='+')
    
    # Also plot physical model prediction for comparison
    plt.scatter(estimated_positions_test[idx, 0], estimated_positions_test[idx, 1], 
               color='red', label='Physical' if i==0 else "", s=80, marker='x')
    
    # Draw connection lines
    plt.plot([predicted_positions[idx, 0], true_positions_test[idx, 0]],
             [predicted_positions[idx, 1], true_positions_test[idx, 1]],
             'g-', alpha=0.6)
    plt.plot([estimated_positions_test[idx, 0], true_positions_test[idx, 0]],
             [estimated_positions_test[idx, 1], true_positions_test[idx, 1]],
             'r-', alpha=0.6)
    
    # Calculate errors
    phys_err = np.sqrt(np.sum((true_positions_test[idx] - estimated_positions_test[idx])**2))
    resid_err = np.sqrt(np.sum((true_positions_test[idx] - predicted_positions[idx])**2))
    imp = (phys_err - resid_err) / phys_err * 100
    
    plt.title(f'Sample {idx}\nImprovement: {imp:.1f}%', fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Only show legend in the first subplot
    if i == 0:
        plt.legend(loc='upper right', fontsize=8)




# 计算欧氏距离误差
print("\n欧氏距离误差分析:")



# 计算残差学习后的欧氏距离误差
residual_euclidean_errors = np.sqrt(np.sum((true_positions_test - predicted_positions)**2, axis=1))
residual_mean_error = np.mean(residual_euclidean_errors)

residual_under_1 = np.mean(residual_errors < 1) * 100
residual_under_2 = np.mean(residual_errors < 2) * 100
residual_under_3 = np.mean(residual_errors < 3) * 100
residual_under_4 = np.mean(residual_errors < 4) * 100

residual_under_5 = np.mean(residual_errors < 5) * 100
residual_under_10 = np.mean(residual_errors < 10) * 100
residual_under_20 = np.mean(residual_errors < 20) * 100


print("")
print(f"残差学习测试集平均欧氏距离: {residual_mean_error:.2f} 米")

print(f"残差学习欧氏距离 < 1米的样本数: {sum(residual_errors < 1)} / {len(residual_errors)}")
print(f"残差学习欧氏距离 < 2米的样本数: {sum(residual_errors < 2)} / {len(residual_errors)}")
print(f"残差学习欧氏距离 < 3米的样本数: {sum(residual_errors < 3)} / {len(residual_errors)}")
print(f"残差学习欧氏距离 < 4米的样本数: {sum(residual_errors < 4)} / {len(residual_errors)}")
print(f"残差学习欧氏距离 < 8米的样本数: {sum(residual_errors < 8)} / {len(residual_errors)}")
print(f"残差学习欧氏距离 < 16米的样本数: {sum(residual_errors < 16)} / {len(residual_errors)}")


print("\n生成测试集预测散点图...")

# 计算每个点的欧氏距离误差
point_errors = np.sqrt(np.sum((true_positions_test - predicted_positions)**2, axis=1))

# 筛选误差小于等于3米的点
filter_mask = point_errors <= 15.0
filtered_true_x = true_positions_test[filter_mask, 0]
filtered_true_y = true_positions_test[filter_mask, 1]
filtered_pred_x = predicted_positions[filter_mask, 0]
filtered_pred_y = predicted_positions[filter_mask, 1]

print(f"总测试点数: {len(point_errors)}")
print(f"筛选后剩余点数: {sum(filter_mask)} (误差<=15米)")
print(f"筛选掉的点数: {len(point_errors) - sum(filter_mask)} (误差>15米)")

# 创建筛选后的x坐标散点图
plt.figure(figsize=(10, 8))
plt.scatter(filtered_true_x, filtered_pred_x, alpha=0.6)
plt.plot([np.min(filtered_true_x), np.max(filtered_true_x)], 
         [np.min(filtered_true_x), np.max(filtered_true_x)], 
         'r--', label='Perfect Prediction')
plt.xlabel('True X Coordinate (m)')
plt.ylabel('Predicted X Coordinate (m)')
plt.title('X Coordinate: True vs. Predicted (Filtered, Error ≤ 15m)')
plt.grid(True, alpha=0.3)
plt.legend()

# 计算筛选后X坐标的R²值
if len(filtered_true_x) > 1:  # 确保有足够的点计算相关系数
    correlation_matrix = np.corrcoef(filtered_true_x, filtered_pred_x)
    r_squared_x = correlation_matrix[0, 1]**2
    plt.annotate(f'R² = {r_squared_x:.4f} (Filtered)', 
                xy=(0.05, 0.95), xycoords='axes fraction',
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

plt.tight_layout()
plt.savefig("D:/desktop/毕设材料/x_coordinate_prediction_filtered.png", dpi=300)
plt.show()

# 创建筛选后的y坐标散点图
plt.figure(figsize=(10, 8))
plt.scatter(filtered_true_y, filtered_pred_y, alpha=0.6)
plt.plot([np.min(filtered_true_y), np.max(filtered_true_y)], 
         [np.min(filtered_true_y), np.max(filtered_true_y)], 
         'r--', label='Perfect Prediction')
plt.xlabel('True Y Coordinate (m)')
plt.ylabel('Predicted Y Coordinate (m)')
plt.title('Y Coordinate: True vs. Predicted (Filtered, Error ≤ 15m)')
plt.grid(True, alpha=0.3)
plt.legend()

# 计算筛选后Y坐标的R²值
if len(filtered_true_y) > 1:  # 确保有足够的点计算相关系数
    correlation_matrix = np.corrcoef(filtered_true_y, filtered_pred_y)
    r_squared_y = correlation_matrix[0, 1]**2
    plt.annotate(f'R² = {r_squared_y:.4f} (Filtered)', 
                xy=(0.05, 0.95), xycoords='axes fraction',
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

plt.tight_layout()
plt.savefig("D:/desktop/毕设材料/y_coordinate_prediction_filtered.png", dpi=300)
plt.show()

print("筛选后的散点图生成完成。")

# 额外添加一张对比图，同时显示筛选前后的点
plt.figure(figsize=(12, 10))

# 绘制所有点（淡灰色）
plt.scatter(true_positions_test[:, 0], predicted_positions[:, 0], 
           alpha=0.3, color='gray', label='All Points')

# 高亮显示筛选后的点（蓝色）
plt.scatter(filtered_true_x, filtered_pred_x, 
           alpha=0.6, color='blue', label='Filtered Points (Error ≤ 5m)')

# 绘制对角线
plt.plot([np.min(true_positions_test[:, 0]), np.max(true_positions_test[:, 0])], 
         [np.min(true_positions_test[:, 0]), np.max(true_positions_test[:, 0])], 
         'r--', label='Perfect Prediction')

plt.xlabel('True X Coordinate (m)')
plt.ylabel('Predicted X Coordinate (m)')
plt.title('X Coordinate: Comparison of All Points vs. Filtered Points')
plt.grid(True, alpha=0.3)
plt.legend()

# 添加筛选前后的R²值
if len(filtered_true_x) > 1:
    # 原始R²
    orig_corr = np.corrcoef(true_positions_test[:, 0], predicted_positions[:, 0])
    r_squared_orig = orig_corr[0, 1]**2
    
    # 筛选后R²
    filt_corr = np.corrcoef(filtered_true_x, filtered_pred_x)
    r_squared_filt = filt_corr[0, 1]**2
    
    plt.annotate(f'Original R² = {r_squared_orig:.4f}\nFiltered R² = {r_squared_filt:.4f}', 
                xy=(0.05, 0.95), xycoords='axes fraction',
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))


plt.tight_layout()
#plt.savefig("D:/desktop/毕设材料/detailed_samples_comparison.png", dpi=300)
plt.show()  # Display the figure on screen
print("Detailed sample comparison completed.")

print("\nAll processing completed!")


