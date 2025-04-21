import numpy as np
import pandas as pd
from itertools import combinations
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from tqdm import tqdm
import random
# Speed of light (m/s)
c = 299792458

# Excel file path
excel_path = r"D:\desktop\毕设材料\output_classifier.xlsx"
df = pd.read_excel(excel_path, header=None, engine='openpyxl')

#random.seed(41)
#np.random.seed(41)  # 如果你也使用NumPy
sample_indices = random.sample(range(len(df)), 1)

df = df.iloc[sample_indices].reset_index(drop=True)
print(sample_indices)
print(df)
# Receiver positions (x, y, z) - in meters
receivers = np.array([
    [0, 0, 0],      # Receiver 1 position
    [181, 528, 2],  # Receiver 2 position
    [277, 304, 4],  # Receiver 3 position
    [413, 228, 6],  # Receiver 4 position
    [572, 324, 8],  # Receiver 5 position
    [466, 70, 10],  # Receiver 6 position
])

# Reading Excel data
try:
    #df = pd.read_excel(excel_path, header=None, engine='openpyxl')
    print("Original data shape:", df.shape)
    
    # Manually specify column names
    column_names = ['x', 'y', 'label', 
                   'TOA1', 'TOA2', 'TOA3', 'TOA4', 'TOA5', 'TOA6',
                   'TOA1_ray_type', 'TOA2_ray_type', 'TOA3_ray_type', 'TOA4_ray_type', 'TOA5_ray_type', 'TOA6_ray_type']
    
    # Ensure column names match DataFrame columns
    if len(column_names) > df.shape[1]:
        column_names = column_names[:df.shape[1]]
    elif len(column_names) < df.shape[1]:
        for i in range(len(column_names), df.shape[1]):
            column_names.append(f'unknown_col{i+1}')
    
    df.columns = column_names
    
except Exception as e:
    print(f"Error reading Excel file: {e}")
    import traceback
    print(traceback.format_exc())
    raise

# Ensure all required columns exist and convert data types
numeric_cols = ['x', 'y', 'TOA1', 'TOA2', 'TOA3', 'TOA4', 'TOA5', 'TOA6']
for col in numeric_cols:
    if col in df.columns and not pd.api.types.is_float_dtype(df[col]):
        df[col] = pd.to_numeric(df[col], errors='coerce', downcast=None)

reflection_cols = ['label', 'TOA1_ray_type', 'TOA2_ray_type', 'TOA3_ray_type', 'TOA4_ray_type', 'TOA5_ray_type', 'TOA6_ray_type']
for col in reflection_cols:
    if col in df.columns and not pd.api.types.is_integer_dtype(df[col]):
        df[col] = pd.to_numeric(df[col], errors='coerce', downcast='integer')

# Physical consistency function: calculate TOAs from emitter to all receivers
def calculate_toas(emitter_pos, receiver_positions):
    """
    Calculate theoretical TOA values from emitter to all receivers
    
    Parameters:
    emitter_pos: Emitter position [x, y, z]
    receiver_positions: Array of receiver positions
    
    Returns:
    Array of TOA values
    """
    # Ensure emitter_pos is a 1D array
    emitter_pos = np.array(emitter_pos).flatten()
    
    # Calculate distance from emitter to each receiver
    distances = []
    for receiver_pos in receiver_positions:
        # Calculate Euclidean distance
        distance = np.sqrt(np.sum((receiver_pos - emitter_pos)**2))
        distances.append(distance)
    
    # Convert to TOA
    toas = np.array(distances) / c
    return toas

def label_to_center(label, region_size=120, base_height=80):
    """Convert label to center position"""
    label -= 1  # 从0开始
    row = label // 5
    col = label % 5
    center_x = (col + 0.5) * region_size
    center_y = (row + 0.5) * region_size
    
    # 添加随机高度误差
    height_error = np.random.uniform(0, 3)  # 0到3米的随机误差
    height = base_height + height_error
    
    return np.array([center_x, center_y, height])

# Estimate emitter position based on three receivers' TOA
def estimate_position(toa_values, receiver_indices, receiver_positions, label=None):
    """
    Estimate emitter position using three receivers' TOA
    
    Parameters:
    toa_values: All receivers' TOA values
    receiver_indices: Indices of three receivers to use
    receiver_positions: All receiver positions
    label: Optional label for initial guess
    
    Returns:
    Estimated emitter position
    """
    # 选中三台接收机
    selected_receivers = np.array([receiver_positions[i] for i in receiver_indices])
    selected_toas = np.array([toa_values[i] for i in receiver_indices])
    
    # 初始猜测
    if label is not None and 1 <= label <= 25:
        center = label_to_center(label)
        fixed_z = center[2]
        x0, y0 = center[0], center[1]
    else:
        avg = np.mean(selected_receivers, axis=0)
        x0, y0 = avg[0], avg[1]
        fixed_z = avg[2]

    # 残差函数
    def residuals(pos_xy):
        pos = np.array([pos_xy[0], pos_xy[1], fixed_z])
        calculated_toas = calculate_toas(pos, selected_receivers)
        return calculated_toas - selected_toas

    # 最小二乘拟合
    try:
        result = least_squares(residuals, [x0, y0], method='lm')
        # 返回优化后的结果，不是初始猜测
        return np.array([result.x[0], result.x[1], fixed_z])
    except Exception as e:
        print(f"❌ least_squares failed: {e}")
        return np.array([x0, y0, fixed_z])  # 返回初始猜测作为后备

# Calculate physical consistency score
def calculate_consistency_score(estimated_position, toa_values, receiver_positions):
    """
    Calculate physical consistency score for estimated position
    
    Parameters:
    estimated_position: Estimated emitter position
    toa_values: All receivers' TOA values
    receiver_positions: All receiver positions
    
    Returns:
    Consistency score (negative mean error)
    """
    try:
        expected_toas = calculate_toas(estimated_position, receiver_positions)
        toa_errors = np.abs(expected_toas - toa_values)
        consistency_score = -np.mean(toa_errors)
        return consistency_score
    except Exception as e:
        print(f"Error calculating consistency score: {e}")
        return -np.inf  # Return negative infinity as worst score

# Calculate radius of gyration
def radius_of_gyration(positions):
    """Calculate radius of gyration for a set of points"""
    if len(positions) < 2:
        return float('inf')  # 如果点不足，返回无穷大
    
    centroid = np.mean(positions, axis=0)
    squared_distances = np.sum((positions - centroid) ** 2, axis=1)
    return np.sqrt(np.mean(squared_distances))

# Calculate R² value for linear fit
def calculate_r2(positions):
    """Calculate R² value for linear fit of points"""
    if len(positions) < 2:
        return -float('inf')  # 如果点不足，返回负无穷大
    
    X = positions[:, 0].reshape(-1, 1)  # x坐标作为特征
    y = positions[:, 1]                 # y坐标作为目标变量
    
    try:
        model = LinearRegression()
        model.fit(X, y)
        y_pred = model.predict(X)
        return r2_score(y, y_pred)
    except Exception as e:
        print(f"❌ R² calculation failed: {e}")
        return -float('inf')

# Find unreliable receivers using geometric analysis0.7
def identify_blacklist_receivers(toa_values, receiver_positions, label=None, rog_threshold=1000, r2_threshold=1):
    """
    Identify receivers that have significantly worse geometric properties
    
    Parameters:
    toa_values: All receivers' TOA values
    receiver_positions: All receiver positions
    label: Optional label for initial guess
    rog_threshold: Threshold factor for radius of gyration outlier detection
    r2_threshold: Threshold factor for R² outlier detection
    
    Returns:
    List of blacklisted receiver indices
    """
    # All possible three-receiver combinations
    all_combinations = list(combinations(range(6), 3))
    
    # Calculate estimated positions for all combinations
    all_positions = []
    for combo in all_combinations:
        pos = estimate_position(toa_values, combo, receiver_positions, label=label)
        all_positions.append(pos)
    
    # Calculate geometric metrics for each receiver
    receiver_metrics = {}
    
    for r in range(6):
        # Get combinations containing this receiver
        r_indices = [i for i, combo in enumerate(all_combinations) if r in combo]
        r_positions = np.array([all_positions[i] for i in r_indices])
        
        # Calculate radius of gyration
        rog = radius_of_gyration(r_positions)
        
        # Calculate R² value
        r2 = calculate_r2(r_positions)
        
        receiver_metrics[r] = {'rog': rog, 'r2': r2}
    
    # Calculate median values
    rog_values = [m['rog'] for m in receiver_metrics.values()]
    r2_values = [m['r2'] for m in receiver_metrics.values()]
    
    median_rog = np.median(rog_values)
    median_r2 = np.median(r2_values)
    
    # Identify outliers
    blacklist = []
    
    for r, metrics in receiver_metrics.items():
        # Check if radius of gyration is significantly worse than median
        if metrics['rog'] > median_rog * rog_threshold:
            blacklist.append(r)
            continue
        
        # Check if R² is significantly worse than median
        # For R², lower values are worse, so we compare with median * (1 - threshold)
        if metrics['r2'] < median_r2 * (1 - r2_threshold):
            blacklist.append(r)
    
    return blacklist, receiver_metrics

def predict_lowest_reflection_receivers_hybrid(row, receiver_positions, debug=True):
    """
    Predict which three receivers have lowest reflection order using hybrid method
    """
    try:
        # Extract TOA values
        toa_values = np.array([
            float(row['TOA1']), float(row['TOA2']), float(row['TOA3']), 
            float(row['TOA4']), float(row['TOA5']), float(row['TOA6'])
        ])
    except (ValueError, TypeError) as e:
        print(f"Error parsing TOAs: {e}")
        return (0, 1, 2)
    
    # Check data validity
    if np.any(np.isnan(toa_values)) or np.any(np.isinf(toa_values)):
        print(f"Invalid TOA values: {toa_values}")
        return (0, 1, 2)
    
    try:
        label = int(row['label'])
    except (ValueError, TypeError):
        label = None
    
    # Step 1: Identify blacklist receivers using geometric analysis
    blacklist, receiver_metrics = identify_blacklist_receivers(toa_values, receiver_positions, label=label)
    
    # Step 2: Calculate consistency scores for all combinations (只计算一次)
    all_combinations = list(combinations(range(6), 3))
    combinations_with_scores = []
    all_positions = []  # 保存所有组合的位置估计
    
    for combo in all_combinations:
        try:
            # Estimate emitter position
            est_pos = estimate_position(toa_values, combo, receiver_positions, label=label)
            all_positions.append(est_pos)  # 保存位置
            
            # Calculate consistency score
            score = calculate_consistency_score(est_pos, toa_values, receiver_positions)
            
            combinations_with_scores.append((combo, score, est_pos))
        except Exception as e:
            if debug:
                print(f"Error calculating combination {combo}: {e}")
            all_positions.append(None)  # 保存None表示计算失败
            continue
    
    # Sort combinations by consistency score (descending)
    combinations_with_scores.sort(key=lambda x: x[1], reverse=True)
    
    # 新增: 计算胜负关系
    if debug:
        # 计算每对接收机的RoG
        pair_rogs = {}
        for i, j in combinations(range(6), 2):
            # 找出包含接收机i和j的所有组合的索引
            pair_indices = [k for k, combo in enumerate(all_combinations) 
                        if i in combo and j in combo and all_positions[k] is not None]

            if len(pair_indices) > 0:
                pair_positions = np.array([all_positions[k] for k in pair_indices])
                # 计算这些位置的RoG
                rog = radius_of_gyration(pair_positions)
                pair_rogs[(min(i, j), max(i, j))] = rog

        # 打印各对接收机的RoG
        print("\n===== 接收机对的方差半径 =====")
        print(f"{'接收机对':^15}{'RoG值':^15}")
        for (i, j), rog in sorted(pair_rogs.items(), key=lambda x: x[1]):
            print(f"({i+1},{j+1}){'':<9}{rog:^15.2f}")

        # 计算胜负关系（改进：按比例得分）
        win_records = {i: [] for i in range(6)}  # 每个接收机战胜的其他接收机
        win_scores = {i: 0.0 for i in range(6)}  # 每个接收机总得分

        for i in range(6):
            for j in range(6):
                if i == j:
                    continue

                score = 0.0  # i相对于j的累计得分

                for k in range(6):
                    if k == i or k == j:
                        continue

                    key_ik = (min(i, k), max(i, k))
                    key_jk = (min(j, k), max(j, k))

                    if key_ik in pair_rogs and key_jk in pair_rogs:
                        rog_i = pair_rogs[key_ik]
                        rog_j = pair_rogs[key_jk]
                        max_rog = max(rog_i, rog_j)

                        # 归一化后得分（更小得分更高）
                        norm_i = 1 - (rog_i / max_rog)
                        norm_j = 1 - (rog_j / max_rog)
                        score += norm_i - norm_j

                if score > 0:
                    win_records[i].append(j)
                    win_scores[i] += score

        # 打印胜负榜
        print("\n===== 接收机胜负榜 =====")
        for i in range(6):
            defeated = [j+1 for j in win_records[i]]
            defeated_str = ", ".join(map(str, defeated)) if defeated else "无"
            print(f"接收机{i+1}战胜的接收机: {defeated_str}")

        # 打印荣誉榜（按总得分排名）
        ranked_receivers = sorted(range(6), key=lambda i: -win_scores[i])
        print("\n===== 接收机荣誉榜 =====")
        print(f"{'排名':^6}{'接收机':^10}{'胜利得分':^12}")
        for rank, r in enumerate(ranked_receivers):
            print(f"{rank+1:^6}{r+1:^10}{win_scores[r]:^12.2f}")

    
    if debug:
        # 打印物理一致性得分和排名
        print("\n===== 物理一致性得分排名 =====")
        print(f"{'排名':^6}{'组合':^15}{'物理一致性得分':^20}{'包含黑名单':^12}")
        
        for rank, (combo, score, pos) in enumerate(combinations_with_scores):  # 打印全部20个
            has_blacklisted = "是" if any(r in blacklist for r in combo) else "否"
            combo_str = f"({combo[0]+1},{combo[1]+1},{combo[2]+1})"
            print(f"{rank+1:^6}{combo_str:^15}{score:^20.3e}{has_blacklisted:^12}")
    
    # Step 3: Select first combination not containing blacklisted receivers
    for combo, score, pos in combinations_with_scores:
        if not any(r in blacklist for r in combo):
            if debug:
                print(f"\n选择的组合: ({combo[0]+1},{combo[1]+1},{combo[2]+1})")
                print(f"一致性得分: {score:.3e}")
                print(f"估计位置: {pos}")
            return combo
    
    # If all combinations contain blacklisted receivers, return the best overall
    if combinations_with_scores:
        best_combo = combinations_with_scores[0][0]
        if debug:
            print(f"\n所有组合都包含黑名单接收机。")
            print(f"使用整体最佳组合: ({best_combo[0]+1},{best_combo[1]+1},{best_combo[2]+1})")
        return best_combo
    
    # Fallback if no combinations could be calculated
    print("警告: 无法计算任何组合，使用默认组合 (0,1,2)")
    return (0, 1, 2)

# Main analysis process
def main_analysis():
    print("Starting analysis with hybrid method...")
    
    # Check data validity
    invalid_rows = df.isnull().any(axis=1).sum()
    if invalid_rows > 0:
        print(f"WARNING: {invalid_rows} rows contain NaN values")
    
    # Prepare for results
    predictions = []
    errors_count = 0
    processed_count = 0
    
    # 得分评估变量
    total_max_score = 0        # 总共最多能得的分数
    total_actual_score = 0     # 总共实际得到的分数
    original_avg_ray = 0       # 原始平均反射阶数
    selected_avg_ray = 0       # 选择后平均反射阶数
    
    # 场景统计
    scenarios = {
        'siege': {
            'count': 0,            # 样本数量
            'max_score': 0,        # 最多能得的分数
            'actual_score': 0,     # 实际得到的分数
            'orig_ray': 0,         # 原始平均反射阶数
            'sel_ray': 0           # 选择后平均反射阶数
        },
        'battle': {
            'count': 0, 
            'max_score': 0, 
            'actual_score': 0, 
            'orig_ray': 0, 
            'sel_ray': 0
        },
        'counter_siege': {
            'count': 0, 
            'max_score': 0, 
            'actual_score': 0, 
            'orig_ray': 0, 
            'sel_ray': 0
        }
    }
    
    # For debugging: analyze a small subset
    debug_mode = False
    if debug_mode:
        sample_indices = [0, 100, 200, 300, 400]
        sample_rows = [df.iloc[i] for i in sample_indices if i < len(df)]
        debug_sample = sample_rows[0] if sample_rows else None
        if debug_sample is not None:
            print("\n===== Testing hybrid method on sample row =====")
            result = predict_lowest_reflection_receivers_hybrid(debug_sample, receivers, debug=True)
            print(f"Hybrid method predicted: {result}")
            
            # Also test the original method for comparison
            true_reflection_orders = np.array([
                int(debug_sample['TOA1_ray_type']), int(debug_sample['TOA2_ray_type']), int(debug_sample['TOA3_ray_type']),
                int(debug_sample['TOA4_ray_type']), int(debug_sample['TOA5_ray_type']), int(debug_sample['TOA6_ray_type'])
            ])
            print(f"True reflection orders: {true_reflection_orders}")
            print(f"True best combination would be: {np.argsort(true_reflection_orders)[:3]}")
    
    # Process each row
    for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing data with hybrid method"):
        try:
            # Check if current row has NaN values
            if row.isnull().any():
                missing_cols = row.index[row.isnull()].tolist()
                print(f"Row {index} has missing values: {missing_cols}")
                errors_count += 1
                continue
            
            # Get true ray types (reflection orders)
            true_reflection_orders = np.array([
                int(row['TOA1_ray_type']), int(row['TOA2_ray_type']), int(row['TOA3_ray_type']),
                int(row['TOA4_ray_type']), int(row['TOA5_ray_type']), int(row['TOA6_ray_type'])
            ])
            
            # Predict lowest reflection order receivers using hybrid method
            predicted_indices = predict_lowest_reflection_receivers_hybrid(row, receivers)
            predictions.append(predicted_indices)
            
            # 计算直射信号相关统计
            direct_rays = [i for i, rt in enumerate(true_reflection_orders) if rt == 0]
            selected_direct_rays = [r for r in predicted_indices if true_reflection_orders[r] == 0]
            
            # 场景分类和得分计算
            direct_count = len(direct_rays)
            selected_direct_count = len(selected_direct_rays)
            
            if direct_count >= 4:
                scenario = 'siege'  # 围剿
                max_score = 3       # 最多得3分
            elif direct_count == 3:
                scenario = 'battle'  # 苦战
                max_score = 3       # 最多得3分
            else:
                scenario = 'counter_siege'  # 反围剿
                max_score = direct_count  # 有多少个0就最多得多少分
            
            actual_score = selected_direct_count  # 选中几个直射信号就得几分
            
            # 更新全局统计
            total_max_score += max_score
            total_actual_score += actual_score
            original_avg_ray += np.mean(true_reflection_orders)
            selected_avg_ray += np.mean([true_reflection_orders[r] for r in predicted_indices])
            
            # 更新场景统计
            scenarios[scenario]['count'] += 1
            scenarios[scenario]['max_score'] += max_score
            scenarios[scenario]['actual_score'] += actual_score
            scenarios[scenario]['orig_ray'] += np.mean(true_reflection_orders)
            scenarios[scenario]['sel_ray'] += np.mean([true_reflection_orders[r] for r in predicted_indices])
            
            # Track successfully processed rows
            processed_count += 1
            
            # Print progress periodically
            if debug_mode and (index + 1) % 100 == 0:
                print(f"Progress: Processed {index + 1} rows")
                
        except Exception as e:
            print(f"Error processing row {index}: {e}")
            import traceback
            print(traceback.format_exc())
            errors_count += 1
            continue
    
    # Calculate statistics
    total_rows = len(df)
    valid_rows = processed_count
    
    # 计算得分统计
    score_percentage = (total_actual_score / total_max_score * 100) if total_max_score > 0 else 0
    original_avg_ray /= valid_rows if valid_rows > 0 else 1
    selected_avg_ray /= valid_rows if valid_rows > 0 else 1
    ray_improvement = original_avg_ray - selected_avg_ray
    
    # 计算场景统计
    for scenario, stats in scenarios.items():
        if stats['count'] > 0:
           # stats['score_percentage'] = (stats['actual_score'] / stats['max_score'] * 100) if stats['max_score'] > 0 else 0
            stats['orig_ray'] /= stats['count']
            stats['sel_ray'] /= stats['count']
            stats['ray_improvement'] = stats['orig_ray'] - stats['sel_ray']
    
    # Print results
    print(f"\nAnalysis complete! Total {total_rows} rows, successfully processed {valid_rows}, errors {errors_count}")
    
    # Print score evaluation results
    print("\n===== 得分评估结果 =====")
    print(f"总行数: {valid_rows}")
    print(f"总得分: {total_actual_score}/{total_max_score} ({score_percentage:.2f}%)")
    print(f"原始平均反射阶数: {original_avg_ray:.2f}")
    print(f"选择后平均反射阶数: {selected_avg_ray:.2f}")
    print(f"阶数改善: {ray_improvement:.2f}")
    
    print("\n===== 场景分析 =====")
    # 检查各场景样本数总和
    total_scenario_count = sum(stats['count'] for stats in scenarios.values())
    if total_scenario_count != valid_rows:
        print(f"警告: 场景样本数总和 ({total_scenario_count}) 与总行数 ({valid_rows}) 不一致!")
    
    print("\n1. 围剿 (直射信号 >= 4)")
    print(f"  样本数: {scenarios['siege']['count']}")
    print(f"  得分: {scenarios['siege']['actual_score']}/{scenarios['siege']['max_score']} ")
    print(f"  原始平均反射阶数: {scenarios['siege']['orig_ray']:.2f}")
    print(f"  选择后平均反射阶数: {scenarios['siege']['sel_ray']:.2f}")
  #  print(f"  阶数改善: {scenarios['siege']['ray_improvement']:.2f}")
    
    print("\n2. 苦战 (直射信号 = 3)")
    print(f"  样本数: {scenarios['battle']['count']}")
    print(f"  得分: {scenarios['battle']['actual_score']}/{scenarios['battle']['max_score']} ")
    print(f"  原始平均反射阶数: {scenarios['battle']['orig_ray']:.2f}")
    print(f"  选择后平均反射阶数: {scenarios['battle']['sel_ray']:.2f}")
   # print(f"  阶数改善: {scenarios['battle']['ray_improvement']:.2f}")
    
    print("\n3. 反围剿 (直射信号 <= 2)")
    print(f"  样本数: {scenarios['counter_siege']['count']}")
    print(f"  得分: {scenarios['counter_siege']['actual_score']}/{scenarios['counter_siege']['max_score']} ")
    print(f"  原始平均反射阶数: {scenarios['counter_siege']['orig_ray']:.2f}")
    print(f"  选择后平均反射阶数: {scenarios['counter_siege']['sel_ray']:.2f}")
   # print(f"  阶数改善: {scenarios['counter_siege']['ray_improvement']:.2f}")

    # === 保存筛选后的结果 ===
    filtered_rows = []

    for idx, row in df.iterrows():
        if idx >= len(predictions):
            continue
        selected_indices = predictions[idx]

        # 提取数据
        data_row = [
            row['x'], row['y'], row['label'],
            # 三行TOA
            row[f'TOA{selected_indices[0]+1}'], row[f'TOA{selected_indices[1]+1}'], row[f'TOA{selected_indices[2]+1}'],
            # 三行反射阶数
            row[f'TOA{selected_indices[0]+1}_ray_type'], row[f'TOA{selected_indices[1]+1}_ray_type'], row[f'TOA{selected_indices[2]+1}_ray_type'],
            # 三行接收机编号 (从1开始编号)
            selected_indices[0]+1, selected_indices[1]+1, selected_indices[2]+1
        ]

        filtered_rows.append(data_row)

    # 转为 DataFrame
    filtered_df = pd.DataFrame(filtered_rows)

    # 保存为Excel（无标题）
    output_path = r"D:\desktop\毕设材料\hybrid_method_results.xlsx"
    filtered_df.to_excel(output_path, index=False, header=False)
    print(f"\n结果已保存至：{output_path}")
    
    # 返回增强的结果集
    return {
        'score_percentage': score_percentage,
        'ray_improvement': ray_improvement,
        'scenarios': scenarios
    }

if __name__ == "__main__":
    results = main_analysis()
    '''
    # 打印核心结果摘要
    print("\n===== 结果摘要 =====")
    print(f"得分百分比: {results['score_percentage']:.2f}%")
    print(f"阶数改善: {results['ray_improvement']:.2f}")
    print(f"'围剿'场景得分百分比: {results['scenarios']['siege']['score_percentage']:.2f}%")
    print(f"'苦战'场景得分百分比: {results['scenarios']['battle']['score_percentage']:.2f}%")
    print(f"'反围剿'场景得分百分比: {results['scenarios']['counter_siege']['score_percentage']:.2f}%")
    '''