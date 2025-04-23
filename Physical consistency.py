import numpy as np
import pandas as pd
from itertools import combinations
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
import random
#这个可以进行纯的物理一致性分析
# Speed of light (m/s)
c = 299792458

# Excel file path

excel_path = r"D:\desktop\毕设材料\500M\fingerprint6_500MHz.xlsx"
df = pd.read_excel(excel_path, header=None, engine='openpyxl')

# random.seed(42)
# sample_indices = random.sample(range(len(df)), 20)
# df = df.iloc[sample_indices].reset_index(drop=True)

# Receiver positions (x, y, z) - in meters
receivers = np.array([
    [0, 0, 0],      # Receiver 1 position
    [181, 528, 2],  # Receiver 2 position
    [277, 304, 4],  # Receiver 3 position
    [413, 228, 6],  # Receiver 4 position
    [572, 324, 8],  # Receiver 5 position
    [466, 70, 10],  # Receiver 6 position
])

# Reading Excel data - using precise method
try:
    # Try using openpyxl engine to maintain precision
   # df = pd.read_excel(excel_path, header=None, engine='openpyxl')
    print("Original data shape:", df.shape)
    
    # View the first few rows of original data to check precision
    print("First few rows of original data:")
    pd.set_option('display.float_format', '{:.10e}'.format)  # Show more decimal places
    print(df.head(3))
    
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
    
    # Display processed data and check precision
    print("\nFirst 5 rows of processed data (maintaining original precision):")
    print(df[['TOA1', 'TOA2', 'TOA3', 'TOA4', 'TOA5', 'TOA6']].head())
    
    # Verify each column has enough variability (ensure values aren't all the same)
    for col in ['TOA1', 'TOA2', 'TOA3', 'TOA4', 'TOA5', 'TOA6']:
        unique_values = df[col].nunique()
        print(f"Column {col} has {unique_values} different values")
        if unique_values < 10:
            print("WARNING: This column has low variability, possible reading issue!")
    
except Exception as e:
    print(f"Error reading Excel file: {e}")
    import traceback
    print(traceback.format_exc())
    raise

# Alternative reading method: if above method still has issues, try using xlrd directly
try:
    import xlrd
    print("\nTrying to read Excel file directly with xlrd...")
    
    workbook = xlrd.open_workbook(excel_path)
    sheet = workbook.sheet_by_index(0)
    
    # Check first few rows of data
    print("First few rows read with xlrd:")
    for i in range(min(3, sheet.nrows)):
        row_values = sheet.row_values(i)
        print(f"Row {i+1}: {row_values}")
    
    # If normal reading failed, rebuild DataFrame using xlrd
    if 'TOA1' in df.columns and df['TOA1'].nunique() < 10:
        print("Due to pandas reading inaccuracy, rebuilding DataFrame using xlrd...")
        
        data = []
        for i in range(sheet.nrows):
            row = sheet.row_values(i)
            data.append(row)
        
        df_xlrd = pd.DataFrame(data)
        
        # Name columns for new DataFrame
        if len(column_names) > df_xlrd.shape[1]:
            column_names = column_names[:df_xlrd.shape[1]]
        elif len(column_names) < df_xlrd.shape[1]:
            column_names.extend([f'unknown_col{i+1}' for i in range(len(column_names), df_xlrd.shape[1])])
        
        df_xlrd.columns = column_names
        
        # Check precision of xlrd-read data
        print("\nFirst 5 rows of data read with xlrd:")
        print(df_xlrd[['TOA1', 'TOA2', 'TOA3', 'TOA4', 'TOA5', 'TOA6']].head())
        
        # Check variability
        for col in ['TOA1', 'TOA2', 'TOA3', 'TOA4', 'TOA5', 'TOA6']:
            unique_values = df_xlrd[col].nunique()
            print(f"Column {col} read with xlrd has {unique_values} different values")
        
        # Replace original DataFrame with xlrd-read DataFrame
        df = df_xlrd
    
except ImportError:
    print("xlrd not installed, skipping this attempt...")
except Exception as e:
    print(f"Error using xlrd: {e}")
    import traceback
    print(traceback.format_exc())

# Data type conversion - maintain original precision
if 'TOA1' in df.columns:
    # Check if already float type
    numeric_cols = ['x', 'y', 'TOA1', 'TOA2', 'TOA3', 'TOA4', 'TOA5', 'TOA6']
    for col in numeric_cols:
        if col in df.columns:
            # Check column data type
            if not pd.api.types.is_float_dtype(df[col]):
                print(f"Converting column {col} data type, maintaining original precision")
                # For string columns, ensure scientific notation format is correct
                if pd.api.types.is_string_dtype(df[col]):
                    # Ensure consistent scientific notation format without changing values
                    df[col] = df[col].astype(str).str.replace('E-', 'e-', regex=False)
                    df[col] = df[col].astype(str).str.replace('e-0', 'e-', regex=False)
                # Convert to float64 to maintain precision
                df[col] = pd.to_numeric(df[col], errors='coerce', downcast=None)
    
    reflection_cols = ['label', 'TOA1_ray_type', 'TOA2_ray_type', 'TOA3_ray_type', 'TOA4_ray_type', 'TOA5_ray_type', 'TOA6_ray_type']
    for col in reflection_cols:
        if col in df.columns and not pd.api.types.is_integer_dtype(df[col]):
            print(f"Converting column {col} to integer type")
            df[col] = pd.to_numeric(df[col], errors='coerce', downcast='integer')

# Check TOA column variability again to ensure data is correct
for col in ['TOA1', 'TOA2', 'TOA3', 'TOA4', 'TOA5', 'TOA6']:
    if col in df.columns:
        print(f"\nFirst 20 values of column {col}:")
        print(df[col].head(20).tolist())
        
        # Calculate column statistics
        col_std = df[col].std()
        col_min = df[col].min()
        col_max = df[col].max()
        print(f"{col} statistics: min={col_min:.10e}, max={col_max:.10e}, std_dev={col_std:.10e}")

# Ensure all required columns exist
required_cols = ['x', 'y', 'label', 
                'TOA1', 'TOA2', 'TOA3', 'TOA4', 'TOA5', 'TOA6',
                'TOA1_ray_type', 'TOA2_ray_type', 'TOA3_ray_type', 'TOA4_ray_type', 'TOA5_ray_type', 'TOA6_ray_type']

for col in required_cols:
    if col not in df.columns:
        print(f"WARNING: Missing column '{col}', creating default values")
        df[col] = np.nan

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
    
    Returns:
    Estimated emitter position
    """
    # 选中三台接收机
    selected_receivers = np.array([receiver_positions[i] for i in receiver_indices])
    selected_toas = np.array([toa_values[i] for i in receiver_indices])
    label=None
    # 初始猜测
    if label is not None and 1 <= label <= 25:
        initial_guess = label_to_center(label)
    else:
        initial_guess = np.mean(selected_receivers, axis=0)

    # 残差函数
    def residuals(pos):
        calculated_toas = calculate_toas(pos, selected_receivers)
        return calculated_toas - selected_toas

    # 最小二乘拟合
    result = least_squares(residuals, initial_guess, method='lm',verbose=0)

    return result.x
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

# Main function: predict lowest reflection order receivers using physical consistency
def predict_lowest_reflection_receivers(row, receiver_positions):
    """
    Predict which three receivers have lowest reflection order
    
    Parameters:
    row: Data row containing TOA values
    receiver_positions: Receiver positions
    
    Returns:
    Indices of predicted three lowest reflection order receivers
    """
    # Extract TOA values and ensure they are float
    try:
        toa_values = np.array([
            float(row['TOA1']), float(row['TOA2']), float(row['TOA3']), 
            float(row['TOA4']), float(row['TOA5']), float(row['TOA6'])
        ])
    except (ValueError, TypeError) as e:
        print(f"Error processing row: {row}")
        print(f"Error: {e}")
        # Return default combination if conversion fails
        return (0, 1, 2)
    
    # Check data validity
    if np.any(np.isnan(toa_values)) or np.any(np.isinf(toa_values)):
        print(f"Row contains invalid values: {row}")
        return (0, 1, 2)
    
    # All possible three-receiver combinations
    all_combinations = list(combinations(range(6), 3))
    
    best_score = -np.inf
    best_combination = None
    
    # Evaluate each combination
    for combination in all_combinations:
        try:
            # Estimate emitter position
            estimated_position = estimate_position(toa_values, combination, receiver_positions, label=int(row['label']))

            
            # Ensure estimated position is valid
            if np.any(np.isnan(estimated_position)) or np.any(np.isinf(estimated_position)):
                continue
            
            # Calculate consistency score
            score = calculate_consistency_score(estimated_position, toa_values, receiver_positions)
            
            # Update if better score
            if score > best_score:
                best_score = score
                best_combination = combination
        except Exception as e:
            # Handle calculation errors
            print(f"Error calculating combination {combination}: {e}")
            continue
    
    # If no best combination found, return default
    if best_combination is None:
        print("WARNING: Could not find best combination, using default (0,1,2)")
        return (0, 1, 2)
        
    return best_combination

# Main analysis process
# 在main_analysis函数中添加得分评估部分

def main_analysis():
    print("Starting analysis...")
    
    # Check data validity
    invalid_rows = df.isnull().any(axis=1).sum()
    if invalid_rows > 0:
        print(f"WARNING: {invalid_rows} rows contain NaN values")
    
    # Prepare for results
    predictions = []
    correct_count = 0
    off_by_one_count = 0
    off_by_two_count = 0
    total_reflection_diff = 0
    errors_count = 0
    processed_count = 0
    
    # For tracking reflection order sums
    all_true_reflection_sums = []
    all_predicted_reflection_sums = []
    
    # 得分评估新增变量
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
    
    # Process each row
    for index, row in df.iterrows():
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
            
            # Predict lowest reflection order receivers
            predicted_indices = predict_lowest_reflection_receivers(row, receivers)
            predictions.append(predicted_indices)
            
            # Get true lowest three reflection order receivers
            true_indices = np.argsort(true_reflection_orders)[:3]
            
            # Calculate reflection order sums
            predicted_sum = np.sum(true_reflection_orders[list(predicted_indices)])
            true_sum = np.sum(true_reflection_orders[list(true_indices)])
            
            # Track reflection sums for statistics
            all_true_reflection_sums.append(true_sum)
            all_predicted_reflection_sums.append(predicted_sum)
            
            # 新增: 计算直射信号相关统计
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
            
            # Calculate reflection order sum difference
            diff = abs(predicted_sum - true_sum)
            total_reflection_diff += diff
            
            # Check if prediction is correct (considering same order cases)
            correct = True
            
            # Handle cases with same reflection orders
            unique_orders = np.unique(true_reflection_orders)
            if len(unique_orders) < 3:
                # Find receivers with lowest order
                lowest_order = unique_orders[0]
                lowest_indices = np.where(true_reflection_orders == lowest_order)[0]
                
                # Check if predicted receivers are all within lowest order range
                if len(lowest_indices) >= 3:
                    correct = all(idx in lowest_indices for idx in predicted_indices)
                else:
                    # If fewer than 3 lowest order receivers, check if included all lowest ones
                    if len(unique_orders) > 1:
                        second_lowest = unique_orders[1]
                        second_lowest_indices = np.where(true_reflection_orders == second_lowest)[0]
                        
                        # Check if selected all lowest order receivers plus needed from second lowest
                        remaining_needed = 3 - len(lowest_indices)
                        correct = (all(idx in np.concatenate([lowest_indices, second_lowest_indices]) for idx in predicted_indices) and
                                  len(set(predicted_indices) & set(lowest_indices)) == len(lowest_indices))
                    else:
                        # If only one order type, any three receivers are correct
                        correct = True
            else:
                # Simple case: check if predicted three indices match true three lowest indices
                correct = set(predicted_indices) == set(true_indices)
            
            # Count exact matches and close matches
            if correct:
                correct_count += 1
            elif diff <= 1:
                off_by_one_count += 1
            elif diff <= 2:
                off_by_two_count += 1
            
            # Track successfully processed rows
            processed_count += 1
            
            # Print progress every 100 rows
            if (index + 1) % 100 == 0 or index == 0:
                print(f"Progress: Processed {index + 1} rows")
                # If first row, print detailed results for debugging
                if index == 0:
                    print(f"  Predicted receivers: {predicted_indices}")
                    print(f"  True lowest order receivers: {true_indices}")
                    print(f"  Prediction correct: {correct}")
                    print(f"  Reflection order difference: {diff}")
                
        except Exception as e:
            print(f"Error processing row {index}: {e}")
            import traceback
            print(traceback.format_exc())
            errors_count += 1
            continue
    
    # Calculate statistics
    total_rows = len(df)
    valid_rows = processed_count
    
    # Accuracy metrics
    exact_match_accuracy = correct_count / valid_rows if valid_rows > 0 else 0
    off_by_one_or_exact = (correct_count + off_by_one_count) / valid_rows if valid_rows > 0 else 0
    off_by_two_or_less = (correct_count + off_by_one_count + off_by_two_count) / valid_rows if valid_rows > 0 else 0
    
    # Average reflection order sums
    avg_true_reflection_sum = np.mean(all_true_reflection_sums) if all_true_reflection_sums else 0
    avg_predicted_reflection_sum = np.mean(all_predicted_reflection_sums) if all_predicted_reflection_sums else 0
    
    # 新增: 计算最终得分统计
    score_percentage = (total_actual_score / total_max_score * 100) if total_max_score > 0 else 0
    original_avg_ray /= valid_rows
    selected_avg_ray /= valid_rows
    ray_improvement = original_avg_ray - selected_avg_ray
    
    # 计算场景统计
    for scenario, stats in scenarios.items():
        if stats['count'] > 0:
            stats['score_percentage'] = (stats['actual_score'] / stats['max_score'] * 100) if stats['max_score'] > 0 else 0
            stats['orig_ray'] /= stats['count']
            stats['sel_ray'] /= stats['count']
            stats['ray_improvement'] = stats['orig_ray'] - stats['sel_ray']
    
    # Print results
    print(f"\nAnalysis complete! Total {total_rows} rows, successfully processed {valid_rows}, errors {errors_count}")
    
    print("\nAccuracy Metrics:")
    print(f"Exact match: {exact_match_accuracy:.4f} ({correct_count}/{valid_rows}) = {exact_match_accuracy*100:.2f}%")
    print(f"Exact match or off by one: {off_by_one_or_exact:.4f} ({correct_count+off_by_one_count}/{valid_rows}) = {off_by_one_or_exact*100:.2f}%")
    print(f"Off by two or less: {off_by_two_or_less:.4f} ({correct_count+off_by_one_count+off_by_two_count}/{valid_rows}) = {off_by_two_or_less*100:.2f}%")
    
    print("\nReflection Order Sum Statistics:")
    print(f"Average true reflection order sum: {avg_true_reflection_sum:.4f}")
    print(f"Average predicted reflection order sum: {avg_predicted_reflection_sum:.4f}")
    
    # 新增: 打印得分评估结果
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
    print(f"  得分: {scenarios['siege']['actual_score']}/{scenarios['siege']['max_score']} ({scenarios['siege']['score_percentage']:.2f}%)")
    print(f"  原始平均反射阶数: {scenarios['siege']['orig_ray']:.2f}")
    print(f"  选择后平均反射阶数: {scenarios['siege']['sel_ray']:.2f}")
    print(f"  阶数改善: {scenarios['siege']['ray_improvement']:.2f}")
    
    print("\n2. 苦战 (直射信号 = 3)")
    print(f"  样本数: {scenarios['battle']['count']}")
    print(f"  得分: {scenarios['battle']['actual_score']}/{scenarios['battle']['max_score']} ({scenarios['battle']['score_percentage']:.2f}%)")
    print(f"  原始平均反射阶数: {scenarios['battle']['orig_ray']:.2f}")
    print(f"  选择后平均反射阶数: {scenarios['battle']['sel_ray']:.2f}")
    print(f"  阶数改善: {scenarios['battle']['ray_improvement']:.2f}")
    
    print("\n3. 反围剿 (直射信号 <= 2)")
    print(f"  样本数: {scenarios['counter_siege']['count']}")
    print(f"  得分: {scenarios['counter_siege']['actual_score']}/{scenarios['counter_siege']['max_score']} ({scenarios['counter_siege']['score_percentage']:.2f}%)")
    print(f"  原始平均反射阶数: {scenarios['counter_siege']['orig_ray']:.2f}")
    print(f"  选择后平均反射阶数: {scenarios['counter_siege']['sel_ray']:.2f}")
    print(f"  阶数改善: {scenarios['counter_siege']['ray_improvement']:.2f}")
    
    # Additional analysis: plot accuracy metrics
    plt.figure(figsize=(12, 7))
    metrics = ['Exact Match', 'Off by ≤1', 'Off by ≤2', 'Off by >2']
    values = [
        exact_match_accuracy, 
        off_by_one_count/valid_rows, 
        off_by_two_count/valid_rows, 
        1-off_by_two_or_less
    ]
    colors = ['green', 'yellow', 'orange', 'red']
    
    plt.bar(metrics, values, color=colors)
    plt.title('Physical Consistency Method Prediction Accuracy')
    plt.ylabel('Proportion')
    plt.ylim(0, 1)
    
    for i, v in enumerate(values):
        plt.text(i, v+0.01, f'{v:.2%}', ha='center')
    
    plt.tight_layout()
    plt.savefig('prediction_accuracy_detailed.png')
    
    # Additional analysis: receiver selection accuracy
    actual_indices = []
    predicted_indices_flat = []
    
    for index, row in df.iterrows():
        try:
            if row.isnull().any():
                continue
                
            true_reflection_orders = np.array([
                int(row['TOA1_ray_type']), int(row['TOA2_ray_type']), int(row['TOA3_ray_type']),
                int(row['TOA4_ray_type']), int(row['TOA5_ray_type']), int(row['TOA6_ray_type'])
            ])
            true_indices = np.argsort(true_reflection_orders)[:3]
            
            if index < len(predictions):
                for idx in true_indices:
                    actual_indices.append(idx)
                for idx in predictions[index]:
                    predicted_indices_flat.append(idx)
        except:
            continue
    

    # === 补充：保存筛选后的结果（纯数据，3个TOA + 3个反射阶数） ===
    filtered_rows = []

    for idx, row in df.iterrows():
        if idx >= len(predictions):
            continue
        selected_indices = predictions[idx]

        # 提取原始数据的前三列（x, y, label）
        data_row = [row['x'], row['y'], row['label']]
        
        # 提取三个 TOA 值
        for sel_idx in selected_indices:
            toa_col = f'TOA{sel_idx+1}'
            data_row.append(row[toa_col])
        
        # 提取三个 ray_type 值
        for sel_idx in selected_indices:
            ray_type_col = f'TOA{sel_idx+1}_ray_type'
            data_row.append(row[ray_type_col])
        
        # 添加选中的TOA编号（从1开始）
        for sel_idx in selected_indices:
            data_row.append(sel_idx + 1)  # 加1是因为TOA编号从1开始

        filtered_rows.append(data_row)

    # 转为 DataFrame，不加列名
    filtered_df = pd.DataFrame(filtered_rows)

    # 保存为纯数据 Excel（无标题）
    output_path = r"D:\desktop\毕设材料\500M\fingerprint6_500MHz_filtered.xlsx"
    filtered_df.to_excel(output_path, index=False, header=False)
    print(f"\n纯数据结果已保存至：{output_path}")

    
    # Calculate receiver selection accuracy
    if len(actual_indices) > 0 and len(predicted_indices_flat) > 0:
        receiver_accuracy = []
        for i in range(6):
            actual_count = actual_indices.count(i)
            predicted_count = predicted_indices_flat.count(i)
            if actual_count > 0:
                receiver_accuracy.append((i, predicted_count / actual_count))
            else:
                receiver_accuracy.append((i, 0))
        
        print("\nReceiver selection accuracy:")
        for i, acc in receiver_accuracy:
            print(f"Receiver {i+1}: {acc:.4f}")
    
    # 返回增强的结果集
    return {
        'exact_match': exact_match_accuracy,
        'off_by_one': off_by_one_or_exact,
        'off_by_two': off_by_two_or_less,
        'avg_true_sum': avg_true_reflection_sum,
        'score_percentage': score_percentage,
        'ray_improvement': ray_improvement,
        'scenarios': scenarios
    }


if __name__ == "__main__":
    results = main_analysis()
    
    # 打印核心结果摘要
    print("\n===== 结果摘要 =====")
    print(f"得分百分比: {results['score_percentage']:.2f}%")
    print(f"阶数改善: {results['ray_improvement']:.2f}")
    print(f"'围剿'场景得分百分比: {results['scenarios']['siege']['score_percentage']:.2f}%")
    print(f"'苦战'场景得分百分比: {results['scenarios']['battle']['score_percentage']:.2f}%")
    print(f"'反围剿'场景得分百分比: {results['scenarios']['counter_siege']['score_percentage']:.2f}%")