import numpy as np
import pandas as pd
from itertools import combinations
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

# Speed of light (m/s)
c = 299792458

# Excel file path
excel_path = r"D:\desktop\毕设材料\7\outputclassifier7_noisy.xlsx"

# Receiver positions (x, y, z) - in meters
receivers = np.array([
    [0, 0, 0],      # Receiver 1 position
    [181, 528, 2],  # Receiver 2 position
    [277, 304, 4],  # Receiver 3 position
    [413, 228, 6],  # Receiver 4 position
    [572, 324, 8],  # Receiver 5 position
    [466, 70, 10],  # Receiver 6 position
    [497, 454, 3] # Receiver 7 position
])

# 读取Excel数据并进行处理（与原始代码相同）
try:
    df = pd.read_excel(excel_path, header=None, engine='openpyxl')
    print("Original data shape:", df.shape)
    
    # View the first few rows of original data to check precision
    print("First few rows of original data:")
    pd.set_option('display.float_format', '{:.10e}'.format)
    print(df.head(3))
    
    # Manually specify column names
    column_names = ['x', 'y', 'label', 
                   'TOA1', 'TOA2', 'TOA3', 'TOA4', 'TOA5', 'TOA6', 'TOA7',
                   'TOA1_ray_type', 'TOA2_ray_type', 'TOA3_ray_type', 'TOA4_ray_type', 'TOA5_ray_type', 'TOA6_ray_type','TOA7_ray_type',]
    
    # Ensure column names match DataFrame columns
    if len(column_names) > df.shape[1]:
        column_names = column_names[:df.shape[1]]
    elif len(column_names) < df.shape[1]:
        for i in range(len(column_names), df.shape[1]):
            column_names.append(f'unknown_col{i+1}')
    
    df.columns = column_names
    
    # 数据类型转换和验证（与原始代码相同）
    # ...

except Exception as e:
    print(f"Error reading Excel file: {e}")
    import traceback
    print(traceback.format_exc())
    raise

# 物理一致性相关函数（与原始代码相同）
def calculate_toas(emitter_pos, receiver_positions):
    """计算从发射机到所有接收机的理论TOA值"""
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
    height_error = np.random.uniform(0, 3)  # 0到3米的随机误差
    height = base_height + height_error
    
    return np.array([center_x, center_y, height])

from scipy.optimize import least_squares

def estimate_position(toa_values, receiver_indices, receiver_positions, label=None, region_size=120, debug=False):
    selected_receivers = np.array([receiver_positions[i] for i in receiver_indices])
    selected_toas = np.array([toa_values[i] for i in receiver_indices])

    if label is not None and 1 <= label <= 25:
        center = label_to_center(label)
        fixed_z = center[2]
        x0, y0 = center[0], center[1]



        label_idx = label - 1
        row = label_idx // 5
        col = label_idx % 5
        min_x = col * region_size
        max_x = (col + 1) * region_size
        min_y = row * region_size
        max_y = (row + 1) * region_size

        bounds = ([min_x, min_y], [max_x, max_y])
    else:
        avg = np.mean(selected_receivers, axis=0)
        x0, y0 = avg[0], avg[1]
        fixed_z = avg[2]
        bounds = (-np.inf, -np.inf), (np.inf, np.inf)








    trajectory = []

    def residuals(pos_xy):
        pos = np.array([pos_xy[0], pos_xy[1], fixed_z])
        predicted_toas = calculate_toas(pos, selected_receivers)
        residual = predicted_toas - selected_toas
        if debug:
            error = np.mean(np.abs(residual))
            trajectory.append((pos_xy[0], pos_xy[1], error))
        return residual

    result = least_squares(residuals, [x0, y0], method='lm')

    if debug and trajectory:
        print("\n[Optimization Trajectory]")
        for i, (x, y, err) in enumerate(trajectory):
            print(f"  Step {i+1}: x={x:.2f}, y={y:.2f}, Mean Residual={err:.3e}")

    if debug:
       # print(f"Bounds: x ∈ [{bounds[0][0]:.2f}, {bounds[1][0]:.2f}], "
       #     f"y ∈ [{bounds[0][1]:.2f}, {bounds[1][1]:.2f}]")
        print(f"Initial guess: x={x0:.2f}, y={y0:.2f}, z={fixed_z:.2f}")

    try:
        result = least_squares(residuals, [x0, y0])
    except Exception as e:
        if debug:
            print(f"❌ least_squares failed: {e}")
        return np.array([x0, y0, fixed_z])  # fallback
    

    return np.array([result.x[0], result.x[1], fixed_z])



def calculate_consistency_score(estimated_position, toa_values, receiver_positions, receiver_indices):
    """
    根据三个接收机计算物理一致性得分（误差越小越一致）。
    """
    try:
        selected_receivers = [receiver_positions[i] for i in receiver_indices]
        selected_toas = np.array([toa_values[i] for i in receiver_indices])
        expected_toas = calculate_toas(estimated_position, selected_receivers)
        toa_errors = np.abs(expected_toas - selected_toas)
        return -np.mean(toa_errors)
    except Exception as e:
        print(f"Error calculating consistency score: {e}")
        return -np.inf


def predict_lowest_reflection_receivers(row, receiver_positions):
    """预测三个最低反射阶接收机，固定高度+标签区域限制"""
    try:
        toa_values = np.array([
            float(row['TOA1']), float(row['TOA2']), float(row['TOA3']), 
            float(row['TOA4']), float(row['TOA5']), float(row['TOA6']),float(row['TOA7'])
        ])
    except (ValueError, TypeError) as e:
        print(f"Error parsing TOAs: {e}")
        return (0, 1, 2)

    if np.any(np.isnan(toa_values)) or np.any(np.isinf(toa_values)):
        print(f"Invalid TOA values: {toa_values}")
        return (0, 1, 2)

    try:
        label = int(row['label'])
    except (ValueError, TypeError):
        label = None

    best_score = -np.inf
    best_combination = None

    for combination in combinations(range(7), 3):
        try:
            est_pos = estimate_position(toa_values, combination, receiver_positions, label=label, debug=False)


            if np.any(np.isnan(est_pos)) or np.any(np.isinf(est_pos)):
                continue

            # ✅ 修复：加上 receiver_indices 参数
            score = calculate_consistency_score(est_pos, toa_values, receiver_positions, combination)

            if score > best_score:
                best_score = score
                best_combination = combination
        except Exception as e:
            print(f"Error calculating combination {combination}: {e}")
            continue

    if best_combination is None:
        print("WARNING: Could not find best combination, using default (0,1,2)")
        return (0, 1, 2)

    return best_combination



# 主要分析过程
def main_analysis_expanded():
    print("Starting expanded analysis...")
    
    # 准备结果
    predictions = []
    all_positions_estimations = []  # 存储所有估计位置
    
    # 处理每行数据
    for index, row in df.iterrows():
        try:
            if row.isnull().any():
                missing_cols = row.index[row.isnull()].tolist()
                print(f"Row {index} has missing values: {missing_cols}")
                continue
            
            # 预测最低反射阶接收机
            predicted_indices = predict_lowest_reflection_receivers(row, receivers)
            predictions.append(predicted_indices)
            
            # 提取TOA值
            toa_values = np.array([
                float(row['TOA1']), float(row['TOA2']), float(row['TOA3']), 
                float(row['TOA4']), float(row['TOA5']), float(row['TOA6']),float(row['TOA7'])
            ])
            
            # 使用预测的接收机组合估计位置
            try:
                label = int(row['label'])
                estimated_position = estimate_position(toa_values, predicted_indices, receivers, label=label)
                all_positions_estimations.append(estimated_position)
               
            except Exception as e:
                print(f"Error estimating position for row {index}: {e}")
                all_positions_estimations.append(np.array([np.nan, np.nan, np.nan]))
            
            # 每100行打印进度
            if (index + 1) % 100 == 0 or index == 0:
                print(f"Progress: Processed {index + 1} rows")
                
                
        except Exception as e:
            print(f"Error processing row {index}: {e}")
            import traceback
            print(traceback.format_exc())
            continue
    
    # 创建包含接收机编号的结果DataFrame
    filtered_rows = []

    for idx, row in df.iterrows():
        if idx >= len(predictions):
            continue
        
        selected_indices = predictions[idx]
        
        # 创建数据行
        data_row = []
        
        # 1-3. 添加三个选定的TOA值
        for sel_idx in selected_indices:
            toa_col = f'TOA{sel_idx+1}'
            data_row.append(row[toa_col])
        
        # 4-6. 添加三个选定的ray_type值
        for sel_idx in selected_indices:
            ray_type_col = f'TOA{sel_idx+1}_ray_type'
            data_row.append(row[ray_type_col])
        
        # 7-9. 添加三个选定的接收机编号（从1开始编号）
        for sel_idx in selected_indices:
            data_row.append(sel_idx + 1)  # 添加接收机编号（1-7）
        
        # 10-11. 添加真实发射机位置（用于训练残差学习模型）
        data_row.append(row['x'])
        data_row.append(row['y'])
        
        # # 12-14. 添加估计的发射机位置（物理模型预测）
        # if idx < len(all_positions_estimations):
        #     data_row.append(all_positions_estimations[idx][0])  # 估计的x
        #     data_row.append(all_positions_estimations[idx][1])  # 估计的y
        #     data_row.append(all_positions_estimations[idx][2])  # 估计的z
        # else:
        #     data_row.extend([np.nan, np.nan, np.nan])
            
        filtered_rows.append(data_row)

    # 创建DataFrame
    columns = [
        'TOA_1', 'TOA_2', 'TOA_3',                  # 三个选定的TOA值
        'Ray_Type_1', 'Ray_Type_2', 'Ray_Type_3',   # 三个选定的反射阶数
        'Receiver_ID_1', 'Receiver_ID_2', 'Receiver_ID_3',  # 三个选定的接收机编号
        'True_X', 'True_Y',                          # 真实发射机位置
        'Estimated_X', 'Estimated_Y', 'Estimated_Z'  # 物理模型估计的位置
    ]
    
    filtered_df = pd.DataFrame(filtered_rows, columns=columns)


    
    # 保存纯数据结果（无列名，兼容原有格式）
    output_path = r"D:\desktop\毕设材料\7\coordinate7.xlsx"
    filtered_df.to_excel(output_path, index=False, header=False)
    print(f"\n纯数据结果已保存至：{output_path}")
    
    return filtered_df


def debug_single_row_combinations_with_viz(row_index, df, receiver_positions, range_limit=20, 
                                 highlight_receiver=None, highlight_pair=None):
    """
    Debug a single row with all combinations and visualize estimated positions
    
    Parameters:
    row_index: The index of the data row to analyze
    df: DataFrame containing TOA data
    receiver_positions: Array of receiver positions
    range_limit: The range limit around the true transmitter position (in meters)
    highlight_receiver: Single receiver number to highlight (1-6, None for no highlighting)
    highlight_pair: Tuple of two receiver numbers to highlight (e.g., (1,3), None for no highlighting)
    """
    print(f"\n===== Debugging row {row_index} =====\n")

    # Validating highlight inputs
    if highlight_receiver is not None and highlight_pair is not None:
        print("Warning: Both highlight_receiver and highlight_pair are specified. Using highlight_pair.")
        highlight_receiver = None
    
    # Convert highlight_pair to zero-based indices if provided
    highlight_pair_idx = None
    if highlight_pair is not None:
        if len(highlight_pair) != 2:
            print("Error: highlight_pair must contain exactly two receiver numbers")
            highlight_pair = None
        else:
            highlight_pair_idx = (highlight_pair[0] - 1, highlight_pair[1] - 1)
            print(f"Highlighting combinations with Receivers {highlight_pair[0]} AND {highlight_pair[1]}")

    row = df.iloc[row_index]
    
    # Extract TOA values and label
    toa_values = np.array([
        float(row['TOA1']), float(row['TOA2']), float(row['TOA3']), 
        float(row['TOA4']), float(row['TOA5']), float(row['TOA6']), float(row['TOA7'])
    ])
    label = int(row['label'])
    true_x, true_y = row['x'], row['y']

    print(f"TOA values: {toa_values}")
    print(f"Label: {label}")
    print(f"True position: ({true_x:.2f}, {true_y:.2f})")
    if highlight_receiver:
        print(f"Highlighting combinations with Receiver {highlight_receiver}")

    all_scores = []
    all_positions = []
    combo_labels = []
    all_combos = list(combinations(range(7), 3))
    valid_indices = []  # Track points within valid range
    
    # Modified estimate_position function to capture final optimization result
    def estimate_position_with_return(toa_values, receiver_indices, receiver_positions, label=None, region_size=120):
        """Modified version of estimate_position function to return correct optimization results"""
        selected_receivers = np.array([receiver_positions[i] for i in receiver_indices])
        selected_toas = np.array([toa_values[i] for i in receiver_indices])

        if label is not None and 1 <= label <= 25:
            center = label_to_center(label)
            fixed_z = center[2]
            x0, y0 = center[0], center[1]
        else:
            avg = np.mean(selected_receivers, axis=0)
            x0, y0 = avg[0], avg[1]
            fixed_z = avg[2]

        def residuals(pos_xy):
            pos = np.array([pos_xy[0], pos_xy[1], fixed_z])
            predicted_toas = calculate_toas(pos, selected_receivers)
            return predicted_toas - selected_toas

        # Use least squares optimization
        try:
            result = least_squares(residuals, [x0, y0], method='lm')
            # Return optimized result, not initial guess
            return np.array([result.x[0], result.x[1], fixed_z])
        except Exception as e:
            print(f"❌ least_squares failed: {e}")
            return np.array([x0, y0, fixed_z])  # Return initial guess as fallback

    # Evaluate all combinations
    for i, combination in enumerate(all_combos):
        try:
            # Use modified function to estimate transmitter position
            est_pos = estimate_position_with_return(toa_values, combination, receiver_positions, label=label)
            
            # Print original optimization trajectory (for debugging)
            print(f"\n[Combo {combination}]")
            original_est = estimate_position(toa_values, combination, receiver_positions, label=label, debug=False)
            print(f"Original estimate_position returned: x={original_est[0]:.2f}, y={original_est[1]:.2f}, z={original_est[2]:.2f}")
            print(f"Modified return: x={est_pos[0]:.2f}, y={est_pos[1]:.2f}, z={est_pos[2]:.2f}")

            # Calculate consistency score
            score = calculate_consistency_score(est_pos, toa_values, receiver_positions, combination)

            # Check if estimated position is within range limit
            distance = np.sqrt((est_pos[0] - true_x)**2 + (est_pos[1] - true_y)**2)
            
            # Save all results for printing
            all_scores.append(score)
            all_positions.append(est_pos)
            combo_labels.append(f"R{combination[0]+1}-R{combination[1]+1}-R{combination[2]+1}")
            
            # If within valid range, record index
            if distance <= range_limit:
                valid_indices.append(i)

            # Print information for this combination
            in_range_str = "✓" if distance <= range_limit else "✗"
            print(f"Estimated Position: x={est_pos[0]:.2f}, y={est_pos[1]:.2f}, z={est_pos[2]:.2f} {in_range_str}")
            print(f"Consistency Score: {score:.3e}")
            print(f"Distance to true position: {distance:.2f}m")

        except Exception as e:
            print(f"\n[Combo {combination}] - Error: {e}")
            continue

    # Find the best combination
    if all_scores:
        best_idx = np.argmax(all_scores)
        best_combo = all_combos[best_idx]
        best_pos = all_positions[best_idx]
        best_distance = np.sqrt((best_pos[0] - true_x)**2 + (best_pos[1] - true_y)**2)
        print(f"\n✔️ Best Combination: {best_combo}")
        print(f"✔️ Best Estimated Position: x={best_pos[0]:.2f}, y={best_pos[1]:.2f}, z={best_pos[2]:.2f}")
        print(f"✔️ Best Consistency Score: {all_scores[best_idx]:.3e}")
        print(f"✔️ Distance to true position: {best_distance:.2f}m")
    else:
        print("No valid combinations found!")
        return

    # Create visualization
    plt.figure(figsize=(10, 10))
    
    # Set plot bounds to specified range around true position
    plt.xlim(true_x - range_limit - 5, true_x + range_limit + 5)
    plt.ylim(true_y - range_limit - 5, true_y + range_limit + 5)
    
    # Plot receiver positions (only those in view)
    receivers_in_view = []
    for i, rec_pos in enumerate(receiver_positions):
        # Check if receiver is in view
        if (true_x - range_limit - 5 <= rec_pos[0] <= true_x + range_limit + 5 and
            true_y - range_limit - 5 <= rec_pos[1] <= true_y + range_limit + 5):
            # Use different color for highlighted receiver(s)
            if (highlight_receiver and i+1 == highlight_receiver) or \
               (highlight_pair and (i+1 == highlight_pair[0] or i+1 == highlight_pair[1])):
                plt.scatter(rec_pos[0], rec_pos[1], color='red', marker='s', s=120, edgecolor='black')
                plt.text(rec_pos[0]+2, rec_pos[1]+2, f"R{i+1}", fontsize=10, color='red')
            else:
                plt.scatter(rec_pos[0], rec_pos[1], color='black', marker='s', s=100)
                plt.text(rec_pos[0]+2, rec_pos[1]+2, f"R{i+1}", fontsize=10)
            receivers_in_view.append(i+1)
    
    # Add note about receivers out of view
    if len(receivers_in_view) < 7:
        missing_receivers = [i+1 for i in range(7) if i+1 not in receivers_in_view]
        plt.figtext(0.02, 0.02, f"Receivers {', '.join(map(str, missing_receivers))} out of view", 
                   fontsize=9, bbox=dict(facecolor='white', alpha=0.8))
    
    # Plot true transmitter position
    plt.scatter(true_x, true_y, color='red', marker='*', s=300, label='True Position')
    plt.text(true_x+2, true_y+2, "True Transmitter", fontsize=10, color='red')
    
    # Create colormap normalizer
    import matplotlib.colors as mcolors
    if valid_indices:
        valid_scores = [all_scores[i] for i in valid_indices]
        norm = mcolors.Normalize(vmin=min(valid_scores), vmax=max(valid_scores))
    else:
        norm = mcolors.Normalize(vmin=min(all_scores), vmax=max(all_scores))
    
    # Find indices of combinations containing the highlighted receiver(s)
    highlight_indices = []
    
    if highlight_receiver is not None:
        # Single receiver highlighting
        highlight_receiver_idx = highlight_receiver - 1  # Convert to 0-based index
        for i, combo in enumerate(all_combos):
            if highlight_receiver_idx in combo and i in valid_indices:
                highlight_indices.append(i)
        highlight_label = f'With Receiver {highlight_receiver}'
        
    elif highlight_pair is not None:
        # Dual receiver highlighting
        for i, combo in enumerate(all_combos):
            if highlight_pair_idx[0] in combo and highlight_pair_idx[1] in combo and i in valid_indices:
                highlight_indices.append(i)
        highlight_label = f'With Receivers {highlight_pair[0]} AND {highlight_pair[1]}'
    
    # Only plot estimated positions within range
    if valid_indices:
        # First plot non-highlighted points
        non_highlight_indices = [i for i in valid_indices if i not in highlight_indices]
        if non_highlight_indices:
            non_highlight_positions = np.array([all_positions[i] for i in non_highlight_indices])
            non_highlight_scores = [all_scores[i] for i in non_highlight_indices]
            non_highlight_labels = [combo_labels[i] for i in non_highlight_indices]
            
            scatter1 = plt.scatter(non_highlight_positions[:, 0], non_highlight_positions[:, 1], 
                      c=non_highlight_scores, cmap='viridis', norm=norm,
                      alpha=0.3, s=60, label='Other Combinations')
            
            # Add labels to each point (same font size)
            for i, txt in enumerate(non_highlight_labels):
                plt.annotate(txt, (non_highlight_positions[i, 0], non_highlight_positions[i, 1]), 
                             xytext=(5, 5), textcoords='offset points', fontsize=9, alpha=0.7)
        
        # Then plot highlighted points
        if highlight_indices:
            highlight_positions = np.array([all_positions[i] for i in highlight_indices])
            highlight_scores = [all_scores[i] for i in highlight_indices]
            highlight_labels = [combo_labels[i] for i in highlight_indices]
            
            # Calculate radius of gyration for highlighted combinations
            if len(highlight_positions) > 1:
                centroid = np.mean(highlight_positions, axis=0)
                squared_distances = np.sum((highlight_positions - centroid) ** 2, axis=1)
                rog = np.sqrt(np.mean(squared_distances))
                
                # Print RoG information based on what's being highlighted
                if highlight_receiver is not None:
                    print(f"\nRadius of Gyration for combinations with Receiver {highlight_receiver}: {rog:.2f}m")
                else:
                    print(f"\nRadius of Gyration for combinations with Receivers {highlight_pair}: {rog:.2f}m")
                
                # Add circle showing radius of gyration
                radius_circle = plt.Circle((centroid[0], centroid[1]), rog, color='purple', 
                                          fill=False, linestyle='-.', alpha=0.7)
                plt.gca().add_patch(radius_circle)
                plt.text(centroid[0], centroid[1] + rog + 5, 
                         f"RoG: {rog:.2f}m", color='purple', ha='center')
                
                # Linear fit for the highlighted points (NEW)
                if len(highlight_positions) >= 2:  # Need at least 2 points for linear fit
                    from sklearn.linear_model import LinearRegression
                    from sklearn.metrics import r2_score
                    
                    X = highlight_positions[:, 0].reshape(-1, 1)
                    y = highlight_positions[:, 1]
                    
                    # Fit linear model
                    model = LinearRegression()
                    model.fit(X, y)
                    
                    # Calculate R² score
                    y_pred = model.predict(X)
                    r2 = r2_score(y, y_pred)
                    
                    # Print R² information
                    if highlight_receiver is not None:
                        print(f"Linear fit R² for combinations with Receiver {highlight_receiver}: {r2:.4f}")
                    else:
                        print(f"Linear fit R² for combinations with Receivers {highlight_pair}: {r2:.4f}")
                    
                    # Plot the fitted line
                    x_range = np.array([min(X), max(X)])
                    y_range = model.predict(x_range.reshape(-1, 1))
                    plt.plot(x_range, y_range, color='purple', linestyle='-', linewidth=2,
                             label=f'Linear Fit (R²: {r2:.4f})')
                    
                    # Add R² text to the plot
                    plt.figtext(0.02, 0.93, f"Linear Fit R² value: {r2:.4f}", 
                               fontsize=10, bbox=dict(facecolor='lightpink', alpha=0.8))
            
            scatter2 = plt.scatter(highlight_positions[:, 0], highlight_positions[:, 1], 
                      c=highlight_scores, cmap='plasma', norm=norm,
                      alpha=1.0, s=100, edgecolor='black', linewidth=1, 
                      label=highlight_label)
            
            # Add labels to highlighted points (same font size as others)
            for i, txt in enumerate(highlight_labels):
                plt.annotate(txt, (highlight_positions[i, 0], highlight_positions[i, 1]), 
                             xytext=(5, 5), textcoords='offset points', fontsize=9)
                             
            # Calculate and display average distance for these combinations
            if len(highlight_positions) > 0:
                mean_distance = np.mean([np.sqrt((pos[0] - true_x)**2 + (pos[1] - true_y)**2) for pos in highlight_positions])
                
                # Display text based on what's being highlighted
                if highlight_receiver is not None:
                    plt.figtext(0.02, 0.96, f"Avg distance with R{highlight_receiver}: {mean_distance:.2f}m", 
                              fontsize=10, bbox=dict(facecolor='lightyellow', alpha=0.8))
                else:
                    plt.figtext(0.02, 0.96, f"Avg distance with R{highlight_pair[0]}&R{highlight_pair[1]}: {mean_distance:.2f}m", 
                              fontsize=10, bbox=dict(facecolor='lightyellow', alpha=0.8))
    else:
        print("Warning: No estimated positions within specified range!")
    
    # Add best position (if in range)
    best_distance = np.sqrt((best_pos[0] - true_x)**2 + (best_pos[1] - true_y)**2)
    if best_distance <= range_limit:
        plt.scatter(best_pos[0], best_pos[1], color='orange', marker='X', s=150, 
                    edgecolor='black', linewidth=1.5, label='Best Estimate')
        plt.text(best_pos[0]+2, best_pos[1]+2, "Best Estimate", fontsize=10, color='orange')
    else:
        plt.figtext(0.5, 0.01, f"Best estimate ({best_pos[0]:.2f}, {best_pos[1]:.2f}) out of range - Distance: {best_distance:.2f}m", 
                   fontsize=10, ha='center', bbox=dict(facecolor='yellow', alpha=0.5))
    
    # Add colorbar
    if valid_indices:
        if highlight_indices:
            plt.colorbar(scatter2, label='Consistency Score')
        else:
            plt.colorbar(scatter1, label='Consistency Score')
    
    # Add legend, labels and title
    plt.xlabel('X Position (m)')
    plt.ylabel('Y Position (m)')
    title = f'Transmitter Position Estimates (Row {row_index}, Range: {range_limit}m)'
    if highlight_receiver:
        title += f', Highlighting R{highlight_receiver}'
    elif highlight_pair:
        title += f', Highlighting R{highlight_pair[0]}+R{highlight_pair[1]}'
    plt.title(title)
    plt.grid(True, alpha=0.3)
    
    # Keep aspect ratio equal
    plt.axis('equal')
    
    # Add range indicator circle
    circle = plt.Circle((true_x, true_y), range_limit, color='blue', fill=False, 
                        linestyle='--', alpha=0.5)
    plt.gca().add_patch(circle)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    if highlight_receiver:
        output_path = f'transmitter_positions_row_{row_index}_range_{range_limit}m_R{highlight_receiver}.png'
    elif highlight_pair:
        output_path = f'transmitter_positions_row_{row_index}_range_{range_limit}m_R{highlight_pair[0]}R{highlight_pair[1]}.png'
    else:
        output_path = f'transmitter_positions_row_{row_index}_range_{range_limit}m.png'
   # plt.savefig(output_path, dpi=300)
    #print(f"\nVisualization saved to: {output_path}")
    
    # Show plot
    #plt.show()
    
    # Calculate and return radius of gyration for each receiver
    print("\n=== Analysis of all receivers ===")
    receiver_rogs = {}
    receiver_r2s = {}  # NEW: Store R² values for each receiver
    
    for r in range(7):
        # Find all valid combinations containing this receiver
        r_indices = [i for i, combo in enumerate(all_combos) if r in combo and i in valid_indices]
        
        if len(r_indices) > 1:
            # Get positions for this receiver's combinations
            r_positions = np.array([all_positions[i] for i in r_indices])
            
            # Calculate Radius of Gyration
            r_centroid = np.mean(r_positions, axis=0)
            r_squared_distances = np.sum((r_positions - r_centroid) ** 2, axis=1)
            r_rog = np.sqrt(np.mean(r_squared_distances))
            receiver_rogs[r+1] = r_rog
            
            # Calculate R² for linear fit (NEW)
            if len(r_positions) >= 2:  # Need at least 2 points for linear fit
                from sklearn.linear_model import LinearRegression
                from sklearn.metrics import r2_score
                
                X = r_positions[:, 0].reshape(-1, 1)
                y = r_positions[:, 1]
                
                # Fit linear model
                model = LinearRegression()
                model.fit(X, y)
                
                # Calculate R² score
                y_pred = model.predict(X)
                r2 = r2_score(y, y_pred)
                receiver_r2s[r+1] = r2
                
                print(f"Receiver {r+1}: RoG = {r_rog:.2f}m, R² = {r2:.4f}, from {len(r_indices)} combinations")
            else:
                print(f"Receiver {r+1}: RoG = {r_rog:.2f}m, insufficient points for R², from {len(r_indices)} combinations")
        else:
            print(f"Receiver {r+1}: insufficient combinations within range")
    
    # Sort receivers by R² (higher is better, more linear)
    if receiver_r2s:
        sorted_r2s = sorted(receiver_r2s.items(), key=lambda x: x[1], reverse=True)
        print("\nReceivers sorted by R² (higher is better, more collinear):")
        for r, r2 in sorted_r2s:
            print(f"Receiver {r}: R² = {r2:.4f}")
    
    # Sort receivers by radius of gyration (smaller is better)
    if receiver_rogs:
        sorted_rogs = sorted(receiver_rogs.items(), key=lambda x: x[1])
        print("\nReceivers sorted by Radius of Gyration (smaller is better):")
        for r, rog in sorted_rogs:
            print(f"Receiver {r}: RoG = {rog:.2f}m")
    
    # Calculate radius of gyration and R² for each receiver pair (if we have enough data)
    print("\n=== Analysis of receiver pairs ===")
    pair_rogs = {}
    pair_r2s = {}  # NEW: Store R² values for each pair
    
    for r1, r2 in combinations(range(7), 2):
        # Find all valid combinations containing both receivers
        pair_indices = [i for i, combo in enumerate(all_combos) 
                       if r1 in combo and r2 in combo and i in valid_indices]
        
        if len(pair_indices) > 0:  # Need at least 1 point
            pair_positions = np.array([all_positions[i] for i in pair_indices])
            
            # Calculate Radius of Gyration if we have multiple points
            if len(pair_indices) > 1:
                pair_centroid = np.mean(pair_positions, axis=0)
                pair_squared_distances = np.sum((pair_positions - pair_centroid) ** 2, axis=1)
                pair_rog = np.sqrt(np.mean(pair_squared_distances))
                pair_rogs[(r1+1, r2+1)] = pair_rog
                
                # Calculate R² for linear fit if we have enough points
                if len(pair_positions) >= 2:
                    from sklearn.linear_model import LinearRegression
                    from sklearn.metrics import r2_score
                    
                    X = pair_positions[:, 0].reshape(-1, 1)
                    y = pair_positions[:, 1]
                    
                    # Fit linear model
                    model = LinearRegression()
                    model.fit(X, y)
                    
                    # Calculate R² score
                    y_pred = model.predict(X)
                    r2_value = r2_score(y, y_pred)
                    pair_r2s[(r1+1, r2+1)] = r2_value
                    
                    print(f"Receivers {r1+1},{r2+1}: RoG = {pair_rog:.2f}m, R² = {r2:.4f}, from {len(pair_indices)} combinations")
                else:
                    print(f"Receivers {r1+1},{r2+1}: RoG = {pair_rog:.2f}m, insufficient points for R², from {len(pair_indices)} combinations")
            else:
                # With only one point, use distance to true position as a metric
                pair_rog = np.sqrt((pair_positions[0][0] - true_x)**2 + (pair_positions[0][1] - true_y)**2)
                pair_rogs[(r1+1, r2+1)] = pair_rog
                print(f"Receivers {r1+1},{r2+1}: Only one combination - distance to true position = {pair_rog:.2f}m")
    
    # Sort receiver pairs by R² (higher is better)
    if pair_r2s:
        sorted_pair_r2s = sorted(pair_r2s.items(), key=lambda x: x[1], reverse=True)
        print("\nReceiver pairs sorted by R² (higher is better, more collinear):")
        for pair, r2 in sorted_pair_r2s:
            print(f"Receivers {pair[0]},{pair[1]}: R² = {r2:.4f}")
    
    # Sort receiver pairs by RoG (smaller is better)
    if pair_rogs:
        sorted_pairs = sorted(pair_rogs.items(), key=lambda x: x[1])
        print("\nReceiver pairs sorted by Radius of Gyration (smaller is better):")
        for pair, rog in sorted_pairs:
            print(f"Receivers {pair[0]},{pair[1]}: RoG = {rog:.2f}m")
    
    # Final summary of best receivers
    print("\n=== Summary recommendations ===")
    
    # Recommended best three receivers by R² (highest R²)
    if len(receiver_r2s) >= 3:
        best_r2_receivers = [r for r, _ in sorted(receiver_r2s.items(), key=lambda x: x[1], reverse=True)[:3]]
        print(f"Recommended receivers by highest R² (most collinear): {best_r2_receivers}")
    
    # Recommended best three receivers by RoG (smallest RoG)
    if len(receiver_rogs) >= 3:
        best_rog_receivers = [r for r, _ in sorted(receiver_rogs.items(), key=lambda x: x[1])[:3]]
        print(f"Recommended receivers by smallest RoG (most clustered): {best_rog_receivers}")
    
    return {
        'all_positions': all_positions,
        'all_scores': all_scores,
        'best_pos': best_pos,
        'best_combo': best_combo,
        'valid_indices': valid_indices,
        'receiver_rogs': receiver_rogs,
        'receiver_r2s': receiver_r2s,
        'pair_rogs': pair_rogs,
        'pair_r2s': pair_r2s
    }

# Usage examples:
# results = debug_single_row_combinations_with_viz(9999, df, receivers, range_limit=20)
# Highlight combinations with receiver 1:
# results = debug_single_row_combinations_with_viz(9999, df, receivers, range_limit=20, highlight_receiver=1)
# Highlight combinations with both receivers 1 AND 3:
# results = debug_single_row_combinations_with_viz(9999, df, receivers, range_limit=20, highlight_pair=(1,3))





if __name__ == "__main__":
    result_df = main_analysis_expanded()
    #results = debug_single_row_combinations_with_viz(9999, df, receivers, range_limit=1000, highlight_receiver=7) #用于调试单行数据

    print("\n处理完成！结果数据前5行：")
    #print(result_df.head())