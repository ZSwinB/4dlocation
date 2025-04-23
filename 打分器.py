import numpy as np
import pandas as pd
from itertools import combinations
from scipy.optimize import least_squares
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import os

# 固定参数
FIXED_HEIGHT = 80  # 发射机高度固定为80米
NUM_RECEIVERS = 6  # 接收机数量
LIGHT_SPEED = 299792458  # 光速 (m/s)
REGION_SIZE = 120  # 区域大小

# 接收机位置
receivers = np.array([
    [0, 0, 0],      # Receiver 1 position
    [181, 528, 2],  # Receiver 2 position
    [277, 304, 4],  # Receiver 3 position
    [413, 228, 6],  # Receiver 4 position
    [572, 324, 8],  # Receiver 5 position
    [466, 70, 10],  # Receiver 6 position
])

# 区域标签到初始猜测位置的映射函数
def label_to_center(label, region_size=120, base_height=80):
    label -= 1  # 从0开始
    row = label // 5
    col = label % 5
    center_x = (col + 0.5) * region_size
    center_y = (row + 0.5) * region_size
    
    # 添加随机高度误差
    height_error = np.random.uniform(0, 3)  # 0到3米的随机误差
    height = base_height + height_error
    
    return center_x, center_y, height

# 预计算区域标签中心点 (1-25区域的几何中心坐标)
area_centers = {}
for label in range(1, 26):
    center_x, center_y, _ = label_to_center(label)
    area_centers[label] = (center_x, center_y)

def tdoa_error_function(pos_xy, receiver_positions, measured_tdoas, ref_receiver_idx=0):
    """计算TDOA误差函数，用于最小二乘优化"""
    # 添加固定高度
    pos = np.array([pos_xy[0], pos_xy[1], FIXED_HEIGHT])
    
    # 计算从发射机到各接收机的距离
    distances = np.linalg.norm(receiver_positions - pos, axis=1)
    
    # 计算TOA (不是TDOA，因为输入数据是TOA)
    calculated_toas = distances / LIGHT_SPEED
    
    # 计算TOA的误差
    toa_errors = calculated_toas - measured_tdoas
    
    # 返回误差
    return toa_errors

def estimate_location(receiver_indices, toa_values, init_guess):
    """使用最小二乘法估计发射机位置"""
    # 正确处理接收机索引
    selected_receivers = np.array([receivers[i] for i in receiver_indices])
    selected_toas = np.array([toa_values[i] for i in receiver_indices])
    
    # 使用最小二乘法优化
    result = least_squares(
        tdoa_error_function, 
        init_guess, 
        args=(selected_receivers, selected_toas, 0)
    )
    
    return result.x

def calculate_physical_consistency(receiver_indices, estimated_pos, toa_values):
    """计算物理一致性得分"""
    selected_receivers = np.array([receivers[i] for i in receiver_indices])
    selected_toas = np.array([toa_values[i] for i in receiver_indices])
    
    # 根据估计位置计算理论TOA
    estimated_pos_3d = np.array([estimated_pos[0], estimated_pos[1], FIXED_HEIGHT])
    distances = np.linalg.norm(selected_receivers - estimated_pos_3d, axis=1)
    theoretical_toas = distances / LIGHT_SPEED
    
    # 计算误差
    errors = np.abs(theoretical_toas - selected_toas)
    consistency_error = np.mean(errors)
    
    return -consistency_error  # 负误差作为得分基础

def calculate_test_score(combination_indices, reflection_orders):
    """计算测试得分
    得分规则:
    - 如果样本中没有直射信号(0)，得0分
    - 如果样本中有直射信号，且选中的组合包含至少一个直射信号，得1分
    """
    # 计算样本中直射信号的总数
    total_direct_signals = sum(1 for order in reflection_orders if order == 0)
    
    # 如果样本中没有直射信号，得0分
    if total_direct_signals == 0:
        return 0, False
    
    # 计算选中组合中直射信号的数量
    direct_in_combination = sum(1 for idx in combination_indices if reflection_orders[idx] == 0)
    
    # 判断是否挑选成功(包含至少一个直射信号)
    success = direct_in_combination > 0
    
    # 如果成功挑选到直射信号，得1分，否则得0分
    score = 1 if success else 0
    
    return score, success

def calculate_rog(positions):
    """计算位置的空间离散程度(Radius of Gyration)"""
    if len(positions) <= 1:
        return 0
    
    positions = np.array(positions)
    centroid = np.mean(positions, axis=0)
    squared_distances = np.sum((positions - centroid) ** 2, axis=1)
    return np.sqrt(np.mean(squared_distances))

def main():
    # 读取Excel数据
    file_path = r"d:\desktop\毕设材料\6\classifier_noisy_train.xlsx"
    if not os.path.exists(file_path):
        print(f"文件不存在: {file_path}")
        # 如果在测试环境，使用模拟数据
        data = create_mock_data()
    else:
        data = pd.read_excel(file_path, header=None)
    
    print(f"数据形状: {data.shape}")
    
    # 列映射
    cols = {
        'x': 0, 'y': 1, 'area': 2,
        'toa1': 3, 'toa2': 4, 'toa3': 5, 'toa4': 6, 'toa5': 7, 'toa6': 8,
        'reflection1': 9, 'reflection2': 10, 'reflection3': 11, 
        'reflection4': 12, 'reflection5': 13, 'reflection6': 14
    }
    
    # 存储所有样本的训练数据
    all_features = []
    all_labels = []
    
    # 处理每个样本
    for idx, row in data.iterrows():
        if idx % 10 == 0:
            print(f"处理样本 {idx + 1}/{len(data)}")
        
        # 提取数据
        tx_x, tx_y = row[cols['x']], row[cols['y']]
        area_label = int(row[cols['area']])
        toa_values = np.array([row[cols[f'toa{i+1}']] for i in range(NUM_RECEIVERS)])
        reflection_orders = np.array([row[cols[f'reflection{i+1}']] for i in range(NUM_RECEIVERS)])
        
        # 根据区域标签获取初始猜测位置
        init_center_x, init_center_y = area_centers[area_label]
        
        # 添加随机噪声到初始猜测 (±0.5米)
        init_guess = np.array([
            init_center_x + np.random.uniform(-0.5, 0.5),
            init_center_y + np.random.uniform(-0.5, 0.5)
        ])
        
        # 枚举所有可能的三接收机组合
        receiver_combinations = list(combinations(range(NUM_RECEIVERS), 3))
        
        # 存储每个组合的信息
        combination_data = []
        
        # 计算每个组合的位置估计
        locations = {}
        for comb_idx, comb in enumerate(receiver_combinations):
            # 估计位置
            estimated_pos = estimate_location(comb, toa_values, init_guess)
            locations[comb] = estimated_pos
            
            # 统计直射信号(阶数为0)的数量
            direct_signal_count = sum(1 for idx in comb if reflection_orders[idx] == 0)
            
            combination_data.append({
                'combination': comb,
                'estimated_pos': estimated_pos,
                'direct_signal_count': direct_signal_count
            })
        
        # 计算最大可能的直射信号数量
        max_direct_signals = sum(1 for order in reflection_orders if order == 0)
        
        # 计算对抗得分前的准备
        # 为每个接收机计算不包含其他接收机的组合位置估计
        opponent_excluded_estimates = {i: {} for i in range(NUM_RECEIVERS)}
        for i in range(NUM_RECEIVERS):
            for j in range(NUM_RECEIVERS):
                if i == j:
                    continue
                
                # 找出包含接收机i但不包含接收机j的所有组合
                valid_combinations = [
                    comb for comb in receiver_combinations 
                    if i in comb and j not in comb
                ]
                
                opponent_excluded_estimates[i][j] = [
                    locations[comb] for comb in valid_combinations
                ]
        
        # 计算RoG对抗得分
        rog_scores = np.zeros(NUM_RECEIVERS)
        for i in range(NUM_RECEIVERS):
            victories = 0
            for j in range(NUM_RECEIVERS):
                if i == j:
                    continue
                
                if not opponent_excluded_estimates[i][j] or not opponent_excluded_estimates[j][i]:
                    continue
                
                rog_i = calculate_rog(opponent_excluded_estimates[i][j])
                rog_j = calculate_rog(opponent_excluded_estimates[j][i])
                
                if rog_i < rog_j:  # 更低的RoG更好
                    victories += (rog_j - rog_i)
            
            rog_scores[i] = victories
        
        # 归一化RoG得分
        if np.sum(rog_scores) > 0:
            rog_scores = rog_scores / np.sum(rog_scores)
        
        # 计算R²对抗得分
        r2_scores = np.zeros(NUM_RECEIVERS)
        for i in range(NUM_RECEIVERS):
            victories = 0
            for j in range(NUM_RECEIVERS):
                if i == j:
                    continue
                
                if not opponent_excluded_estimates[i][j] or not opponent_excluded_estimates[j][i]:
                    continue
                
                # 提取X和Y坐标用于线性拟合
                if len(opponent_excluded_estimates[i][j]) > 1:
                    pos_i = np.array(opponent_excluded_estimates[i][j])
                    x_i, y_i = pos_i[:, 0], pos_i[:, 1]
                    
                    # 简单线性回归拟合
                    if len(np.unique(x_i)) > 1:  # 确保x值不全相同
                        coeffs_i = np.polyfit(x_i, y_i, 1)
                        y_pred_i = np.polyval(coeffs_i, x_i)
                        r2_i = r2_score(y_i, y_pred_i)
                    else:
                        r2_i = 0
                else:
                    r2_i = 0
                
                if len(opponent_excluded_estimates[j][i]) > 1:
                    pos_j = np.array(opponent_excluded_estimates[j][i])
                    x_j, y_j = pos_j[:, 0], pos_j[:, 1]
                    
                    # 简单线性回归拟合
                    if len(np.unique(x_j)) > 1:  # 确保x值不全相同
                        coeffs_j = np.polyfit(x_j, y_j, 1)
                        y_pred_j = np.polyval(coeffs_j, x_j)
                        r2_j = r2_score(y_j, y_pred_j)
                    else:
                        r2_j = 0
                else:
                    r2_j = 0
                
                if r2_i > r2_j:  # 更高的R²更好
                    victories += (r2_i - r2_j)
            
            r2_scores[i] = victories
        
        # 归一化R²得分
        if np.sum(r2_scores) > 0:
            r2_scores = r2_scores / np.sum(r2_scores)
        
        # 为每个组合计算三个得分指标
        for comb_data in combination_data:
            comb = comb_data['combination']
            estimated_pos = comb_data['estimated_pos']
            direct_signal_count = comb_data['direct_signal_count']
            
            # 1. 物理一致性得分
            consistency_score = calculate_physical_consistency(comb, estimated_pos, toa_values)
            
            # 归一化一致性得分 (在该样本的所有组合中)
            min_consistency = min(calculate_physical_consistency(c_data['combination'], 
                                                              c_data['estimated_pos'], 
                                                              toa_values) 
                              for c_data in combination_data)
            max_consistency = max(calculate_physical_consistency(c_data['combination'], 
                                                              c_data['estimated_pos'], 
                                                              toa_values) 
                              for c_data in combination_data)
            
            consistency_range = max_consistency - min_consistency
            if consistency_range > 0:
                normalized_consistency = (consistency_score - min_consistency) / consistency_range
            else:
                normalized_consistency = 1.0  # 如果所有得分相同
            
            # 2. RoG对抗得分 (该组合中接收机的平均RoG得分)
            rog_score = np.mean([rog_scores[i] for i in comb])
            
            # 3. R²对抗得分 (该组合中接收机的平均R²得分)
            r2_score_val = np.mean([r2_scores[i] for i in comb])
            
            # 构建特征向量
            features = [normalized_consistency, rog_score, r2_score_val]
            
            # 构建标签 (直射信号数量 / 最大可能直射信号数量)
            if max_direct_signals > 0:
                label = direct_signal_count / max_direct_signals
            else:
                label = 0  # 如果没有直射信号
            
            all_features.append(features)
            all_labels.append(label)
    
    # 转换为numpy数组
    X = np.array(all_features)
    y = np.array(all_labels)
    
    # 使用最小二乘法拟合权重
    weights, residuals, rank, s = np.linalg.lstsq(X, y, rcond=None)
    
    print("训练完成！评分器权重:")
    print(f"物理一致性得分权重 (w1): {weights[0]:.4f}")
    print(f"RoG对抗得分权重 (w2): {weights[1]:.4f}")
    print(f"R²对抗得分权重 (w3): {weights[2]:.4f}")
    
    # 保存模型权重
    np.save("evaluator_weights.npy", weights)
    
    # 评估模型
    y_pred = X @ weights
    mse = np.mean((y - y_pred) ** 2)
    print(f"训练集均方误差: {mse:.6f}")
    
    # # 绘制散点图比较预测值与真实值
    # plt.figure(figsize=(10, 6))
    # plt.scatter(y, y_pred, alpha=0.5)
    # plt.plot([0, 1], [0, 1], 'r--')
    # plt.xlabel('真实标签')
    # plt.ylabel('预测得分')
    # plt.title('预测得分与真实标签比较')
    # plt.savefig('score_comparison.png')
    
    return weights

def create_mock_data(num_samples=100):
    """创建模拟数据用于测试"""
    mock_data = []
    
    for _ in range(num_samples):
        # 随机发射机位置 (在5x5网格内)
        grid_size = 5
        region_size = REGION_SIZE
        max_coord = grid_size * region_size
        
        tx_x = np.random.uniform(0, max_coord)
        tx_y = np.random.uniform(0, max_coord)
        tx_z = FIXED_HEIGHT  # 固定高度
        
        # 确定区域标签
        grid_x = int(tx_x // region_size)
        grid_y = int(tx_y // region_size)
        grid_x = max(0, min(grid_x, grid_size-1))
        grid_y = max(0, min(grid_y, grid_size-1))
        
        area_label = grid_y * grid_size + grid_x + 1
        
        # 生成TOA数据
        toa_values = []
        reflection_orders = []
        
        for i in range(NUM_RECEIVERS):
            # 计算真实距离
            rx = receivers[i]
            distance = np.sqrt((tx_x - rx[0])**2 + (tx_y - rx[1])**2 + (tx_z - rx[2])**2)
            
            # 生成TOA (添加一些噪声)
            toa = distance / LIGHT_SPEED + np.random.normal(0, 1e-9)
            toa_values.append(toa)
            
            # 随机生成反射阶数 (偏向于直射)
            if np.random.random() < 0.7:
                reflection_orders.append(0)  # 70%概率为直射
            else:
                reflection_orders.append(np.random.randint(1, 4))  # 30%概率为1-3阶反射
        
        # 构建样本行
        row = [tx_x, tx_y, area_label] + toa_values + reflection_orders
        mock_data.append(row)
    
    return pd.DataFrame(mock_data)

def test_model(weights, test_file=None):
    """使用训练好的模型在测试集上进行验证"""
    if test_file and os.path.exists(test_file):
        test_data = pd.read_excel(test_file, header=None)
    else:
        # 使用模拟数据
        test_data = create_mock_data(50)
    
    print(f"测试数据形状: {test_data.shape}")
    
    # 列映射
    cols = {
        'x': 0, 'y': 1, 'area': 2,
        'toa1': 3, 'toa2': 4, 'toa3': 5, 'toa4': 6, 'toa5': 7, 'toa6': 8,
        'reflection1': 9, 'reflection2': 10, 'reflection3': 11, 
        'reflection4': 12, 'reflection5': 13, 'reflection6': 14
    }
    
    # 性能评估指标
    position_errors = []
    proportion_direct_signals = []
     # 新增统计指标
    test_scores = []
    max_possible_scores = []
    success_count = 0
    total_samples_with_direct = 0
    # 处理每个测试样本
    for idx, row in test_data.iterrows():
        if idx % 10 == 0:
            print(f"测试样本 {idx + 1}/{len(test_data)}")
        
        # 提取数据
        tx_x, tx_y = row[cols['x']], row[cols['y']]
        area_label = int(row[cols['area']])
        toa_values = np.array([row[cols[f'toa{i+1}']] for i in range(NUM_RECEIVERS)])
        reflection_orders = np.array([row[cols[f'reflection{i+1}']] for i in range(NUM_RECEIVERS)])
        has_direct_signal = any(order == 0 for order in reflection_orders)
        max_score = 1 if has_direct_signal else 0
        max_possible_scores.append(max_score)
        
        if has_direct_signal:
            total_samples_with_direct += 1
        # 根据区域标签获取初始猜测位置
        init_center_x, init_center_y = area_centers[area_label]
        init_guess = np.array([
            init_center_x + np.random.uniform(-0.5, 0.5),
            init_center_y + np.random.uniform(-0.5, 0.5)
        ])
        
        # 枚举所有可能的三接收机组合
        receiver_combinations = list(combinations(range(NUM_RECEIVERS), 3))
        
        # 存储每个组合的评分
        combination_scores = []
        
        # 计算每个组合的位置估计和评分
        locations = {}
        for comb in receiver_combinations:
            # 估计位置
            estimated_pos = estimate_location(comb, toa_values, init_guess)
            locations[comb] = estimated_pos
        
        # 计算对抗得分前的准备
        opponent_excluded_estimates = {i: {} for i in range(NUM_RECEIVERS)}
        for i in range(NUM_RECEIVERS):
            for j in range(NUM_RECEIVERS):
                if i == j:
                    continue
                
                valid_combinations = [
                    comb for comb in receiver_combinations 
                    if i in comb and j not in comb
                ]
                
                opponent_excluded_estimates[i][j] = [
                    locations[comb] for comb in valid_combinations
                ]
        
        # 计算RoG对抗得分
        rog_scores = np.zeros(NUM_RECEIVERS)
        for i in range(NUM_RECEIVERS):
            victories = 0
            for j in range(NUM_RECEIVERS):
                if i == j:
                    continue
                
                if not opponent_excluded_estimates[i][j] or not opponent_excluded_estimates[j][i]:
                    continue
                
                rog_i = calculate_rog(opponent_excluded_estimates[i][j])
                rog_j = calculate_rog(opponent_excluded_estimates[j][i])
                
                if rog_i < rog_j:
                    victories += (rog_j - rog_i)
            
            rog_scores[i] = victories
        
        # 归一化RoG得分
        if np.sum(rog_scores) > 0:
            rog_scores = rog_scores / np.sum(rog_scores)
        
        # 计算R²对抗得分
        r2_scores = np.zeros(NUM_RECEIVERS)
        for i in range(NUM_RECEIVERS):
            victories = 0
            for j in range(NUM_RECEIVERS):
                if i == j:
                    continue
                
                if not opponent_excluded_estimates[i][j] or not opponent_excluded_estimates[j][i]:
                    continue
                
                # 提取X和Y坐标用于线性拟合
                if len(opponent_excluded_estimates[i][j]) > 1:
                    pos_i = np.array(opponent_excluded_estimates[i][j])
                    x_i, y_i = pos_i[:, 0], pos_i[:, 1]
                    
                    if len(np.unique(x_i)) > 1:
                        coeffs_i = np.polyfit(x_i, y_i, 1)
                        y_pred_i = np.polyval(coeffs_i, x_i)
                        r2_i = r2_score(y_i, y_pred_i)
                    else:
                        r2_i = 0
                else:
                    r2_i = 0
                
                if len(opponent_excluded_estimates[j][i]) > 1:
                    pos_j = np.array(opponent_excluded_estimates[j][i])
                    x_j, y_j = pos_j[:, 0], pos_j[:, 1]
                    
                    if len(np.unique(x_j)) > 1:
                        coeffs_j = np.polyfit(x_j, y_j, 1)
                        y_pred_j = np.polyval(coeffs_j, x_j)
                        r2_j = r2_score(y_j, y_pred_j)
                    else:
                        r2_j = 0
                else:
                    r2_j = 0
                
                if r2_i > r2_j:
                    victories += (r2_i - r2_j)
            
            r2_scores[i] = victories
        
        # 归一化R²得分
        if np.sum(r2_scores) > 0:
            r2_scores = r2_scores / np.sum(r2_scores)
        
        # 计算每个组合的最终得分
        all_consistency_scores = []
        for comb in receiver_combinations:
            estimated_pos = locations[comb]
            
            # 1. 物理一致性得分
            consistency_score = calculate_physical_consistency(comb, estimated_pos, toa_values)
            all_consistency_scores.append(consistency_score)
        
        # 归一化一致性得分
        min_consistency = min(all_consistency_scores)
        max_consistency = max(all_consistency_scores)
        consistency_range = max_consistency - min_consistency
        
        for comb_idx, comb in enumerate(receiver_combinations):
            estimated_pos = locations[comb]
            consistency_score = all_consistency_scores[comb_idx]
            
            if consistency_range > 0:
                normalized_consistency = (consistency_score - min_consistency) / consistency_range
            else:
                normalized_consistency = 1.0
            
            # 2. RoG对抗得分
            rog_score = np.mean([rog_scores[i] for i in comb])
            
            # 3. R²对抗得分
            r2_score_val = np.mean([r2_scores[i] for i in comb])
            
            # 特征向量
            features = np.array([normalized_consistency, rog_score, r2_score_val])
            
            # 使用训练好的权重计算最终得分
            final_score = np.dot(features, weights)
            
            # 统计直射信号数量 (仅用于评估)
            direct_signal_count = sum(1 for idx in comb if reflection_orders[idx] == 0)
            
            combination_scores.append({
                'combination': comb,
                'estimated_pos': estimated_pos,
                'final_score': final_score,
                'direct_signal_count': direct_signal_count
            })
        
        # 选择得分最高的组合
        best_combination = max(combination_scores, key=lambda x: x['final_score'])
                # 选择最佳组合后，添加这段代码计算测试得分
        test_score, is_success = calculate_test_score(best_combination['combination'], reflection_orders)
        test_scores.append(test_score)
        if is_success:
            success_count += 1
        
        # 计算定位误差
        true_pos = np.array([tx_x, tx_y])
        est_pos = best_combination['estimated_pos']
        position_error = np.linalg.norm(true_pos - est_pos)
        position_errors.append(position_error)
        
        # 计算直射信号比例
        max_direct_signals = sum(1 for order in reflection_orders if order == 0)
        if max_direct_signals > 0:
            proportion = best_combination['direct_signal_count'] / max_direct_signals
        else:
            proportion = 0
        proportion_direct_signals.append(proportion)
    
    # 输出评估结果
    avg_position_error = np.mean(position_errors)
    median_position_error = np.median(position_errors)
    avg_direct_proportion = np.mean(proportion_direct_signals)
    
    # print(f"平均定位误差: {avg_position_error:.2f} 米")
    # print(f"中位数定位误差: {median_position_error:.2f} 米")
    # print(f"平均直射信号比例: {avg_direct_proportion:.4f}")
    
    # # 绘制定位误差分布图
    # plt.figure(figsize=(10, 6))
    # plt.hist(position_errors, bins=20)
    # plt.xlabel('定位误差 (米)')
    # plt.ylabel('频率')
    # plt.title('定位误差分布')
    # plt.axvline(avg_position_error, color='r', linestyle='--', label=f'平均值: {avg_position_error:.2f}米')
    # plt.axvline(median_position_error, color='g', linestyle='--', label=f'中位数: {median_position_error:.2f}米')
    # plt.legend()
    # plt.savefig('position_error_distribution.png')
    total_score = sum(test_scores)
    max_possible_total = sum(max_possible_scores)
    
    print(f"测试得分: {total_score}/{max_possible_total} ({total_score/max_possible_total*100:.2f}%)")
    print(f"挑选成功率: {success_count}/{total_samples_with_direct} ({success_count/total_samples_with_direct*100:.2f}% 有直射信号的样本)")
    
    return avg_position_error, avg_direct_proportion, total_score, max_possible_total

if __name__ == "__main__":
    print("开始训练评分器...")
    weights = main()
    
    print("\n开始测试评分器...")
    test_file_path = r"D:\desktop\毕设材料\6\classifier_noisy_test.xlsx"  # 添加实际测试文件路径
    avg_error, avg_proportion, total_score, max_possible_score = test_model(weights, test_file_path)