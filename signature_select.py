import numpy as np
import pandas as pd
from itertools import combinations
from scipy.optimize import least_squares
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import os
import matplotlib.pyplot as plt
from tqdm import tqdm

# 光速常量 (m/s)
c = 299792458

# 接收机位置 (x, y, z) - 单位米
receivers = np.array([
    [0, 0, 0],      # 接收机 1
    [181, 528, 2],  # 接收机 2
    [277, 304, 4],  # 接收机 3
    [413, 228, 6],  # 接收机 4
    [572, 324, 8],  # 接收机 5
    [466, 70, 10],  # 接收机 6
])

# 从标签计算中心点坐标
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

# 计算TOA
def calculate_toas(emitter_pos, receiver_positions):
    """计算从发射机到所有接收机的理论TOA值"""
    emitter_pos = np.array(emitter_pos).flatten()
    distances = []
    for receiver_pos in receiver_positions:
        distance = np.sqrt(np.sum((receiver_pos - emitter_pos)**2))
        distances.append(distance)
    
    toas = np.array(distances) / c
    return toas

# 估计发射机位置
def estimate_position(toa_values, receiver_indices, receiver_positions, label=None):
    """使用最小二乘法估计发射机位置"""
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

    # 使用最小二乘法优化
    try:
        result = least_squares(residuals, [x0, y0], method='lm')
        # 返回优化后的结果，不是初始猜测
        return np.array([result.x[0], result.x[1], fixed_z])
    except Exception as e:
        print(f"❌ least_squares failed: {e}")
        return np.array([x0, y0, fixed_z])  # 返回初始猜测作为后备

# 计算方差半径
def radius_of_gyration(positions):
    """计算点集的方差半径"""
    if len(positions) < 2:
        return float('inf')  # 如果点不足，返回无穷大
    
    centroid = np.mean(positions, axis=0)
    squared_distances = np.sum((positions - centroid) ** 2, axis=1)
    return np.sqrt(np.mean(squared_distances))

# 计算R²值
def calculate_r2(positions):
    """计算点集的线性拟合度R²值"""
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

# 选择最佳接收机组合
def select_best_receivers(row_data, debug=False):
    """实现基于排名的接收机选择算法"""
    # 提取数据
    label = int(row_data['label'])
    true_x, true_y = row_data['x'], row_data['y']
    
    toa_values = np.array([
        float(row_data['TOA1']), float(row_data['TOA2']), float(row_data['TOA3']), 
        float(row_data['TOA4']), float(row_data['TOA5']), float(row_data['TOA6'])
    ])
    
    ray_types = np.array([
        int(row_data['TOA1_ray_type']), int(row_data['TOA2_ray_type']), 
        int(row_data['TOA3_ray_type']), int(row_data['TOA4_ray_type']), 
        int(row_data['TOA5_ray_type']), int(row_data['TOA6_ray_type'])
    ])
    
    # 计算所有三元组组合的位置估计
    all_combos = list(combinations(range(6), 3))
    all_positions = []
    
    for combo in all_combos:
        pos = estimate_position(toa_values, combo, receivers, label=label)
        all_positions.append(pos)
    
    # 根据接收机分组计算方差半径和R²值
    receiver_metrics = {}
    
    for r in range(6):
        # 包含该接收机的组合索引
        combo_indices = [i for i, combo in enumerate(all_combos) if r in combo]
        
        # 获取这些组合的位置估计
        positions = np.array([all_positions[i] for i in combo_indices])
        
        # 计算方差半径
        rog = radius_of_gyration(positions)
        
        # 计算R²值
        r2 = calculate_r2(positions)
        
        # 保存指标
        receiver_metrics[r] = {
            'rog': rog,
            'r2': r2,
            'toa': toa_values[r],
            'ray_type': ray_types[r]
        }
    
    # 对方差半径进行排序（从小到大）
    rog_ranking = sorted(range(6), key=lambda r: receiver_metrics[r]['rog'])
    
    # 对R²值进行排序（从大到小）
    r2_ranking = sorted(range(6), key=lambda r: -receiver_metrics[r]['r2'])
    
    # 找出TOA最小的接收机
    min_toa_receiver = np.argmin(toa_values)
    
    # 计算每个接收机的总分
    scores = {}
    
    for r in range(6):
        # 初始分数
        score = 0
        
        # 方差半径排名分数 (第一名得6分，最后一名得1分)
        rog_score = 6 - rog_ranking.index(r)
        score += rog_score
        
        # R²排名分数 (第一名得6分，最后一名得1分)
        r2_score = 6 - r2_ranking.index(r)
        score += r2_score
        
        # TOA最小的接收机奖励（当前设为0）
        toa_bonus = 1 if r == min_toa_receiver else 0
        score += toa_bonus
        
        scores[r] = score
    
    # 按分数排序（从高到低）
    ranked_receivers = sorted(range(6), key=lambda r: (-scores[r], toa_values[r]))
    
    # 选择前三名
    selected_receivers = ranked_receivers[:3]
    
    if debug:
        print(f"\n===== 行数据分析 =====")
        print(f"真实位置: ({true_x:.2f}, {true_y:.2f})")
        print(f"TOA值: {toa_values}")
        print(f"射线类型: {ray_types}")
        
        print("\n方差半径排名 (从小到大):")
        for rank, r in enumerate(rog_ranking):
            print(f"第{rank+1}名: 接收机{r+1}, 方差半径 = {receiver_metrics[r]['rog']:.2f}m")
        
        print("\nR²排名 (从大到小):")
        for rank, r in enumerate(r2_ranking):
            print(f"第{rank+1}名: 接收机{r+1}, R² = {receiver_metrics[r]['r2']:.4f}")
        
        print("\nTOA值 (从小到大):")
        sorted_toa = sorted(range(6), key=lambda r: toa_values[r])
        for rank, r in enumerate(sorted_toa):
            print(f"第{rank+1}名: 接收机{r+1}, TOA = {toa_values[r]:.8f}")
        
        print("\n最终分数:")
        for r in range(6):
            print(f"接收机{r+1}: {scores[r]}分 (RoG排名: {rog_ranking.index(r)+1}, R²排名: {r2_ranking.index(r)+1}, TOA: {toa_values[r]:.8f})")
        
        print("\n选择的接收机:", [r+1 for r in selected_receivers])
        print(f"选择的接收机射线类型: {[ray_types[r] for r in selected_receivers]}")
        
        # 计算原始和选择后的平均反射阶数
        orig_avg_ray = np.mean(ray_types)
        selected_avg_ray = np.mean([ray_types[r] for r in selected_receivers])
        improvement = orig_avg_ray - selected_avg_ray
        
        print(f"原始平均反射阶数: {orig_avg_ray:.2f}")
        print(f"选择后平均反射阶数: {selected_avg_ray:.2f}")
        print(f"阶数改善: {improvement:.2f}")
        
        # 统计直射信号选择率
        direct_rays = [i for i, rt in enumerate(ray_types) if rt == 0]
        selected_direct_rays = [r for r in selected_receivers if ray_types[r] == 0]
        
        if len(direct_rays) > 0:
            direct_ray_percentage = len(selected_direct_rays) / len(direct_rays) * 100
            print(f"直射信号选择率: {direct_ray_percentage:.2f}% ({len(selected_direct_rays)}/{len(direct_rays)})")
        else:
            print("没有直射信号")
    
    # 返回选择的接收机编号、TOA值和射线类型
    return {
        'selected_receivers': selected_receivers,
        'selected_toas': [toa_values[r] for r in selected_receivers],
        'selected_ray_types': [ray_types[r] for r in selected_receivers],
        'all_ray_types': ray_types
    }

# 处理整个数据集
# 修改process_dataset函数中的评估部分

# 修改process_dataset函数中的评估部分，加入得分机制

def process_dataset(input_file, output_file):
    """处理整个数据集并生成结果文件，使用基于得分的评估方法"""
    # 加载数据
    df = pd.read_excel(input_file, header=None)
    
    # 添加列名以便后续处理
    column_names = ['x', 'y', 'label', 
                   'TOA1', 'TOA2', 'TOA3', 'TOA4', 'TOA5', 'TOA6',
                   'TOA1_ray_type', 'TOA2_ray_type', 'TOA3_ray_type', 
                   'TOA4_ray_type', 'TOA5_ray_type', 'TOA6_ray_type']
    
    # 确保列名与DataFrame列数匹配
    if len(column_names) > df.shape[1]:
        column_names = column_names[:df.shape[1]]
    elif len(column_names) < df.shape[1]:
        for i in range(len(column_names), df.shape[1]):
            column_names.append(f'unknown_col{i+1}')
    
    df.columns = column_names
    
    # 初始化结果统计
    total_rows = len(df)
    
    # 得分统计
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
    
    # 准备结果数据
    result_data = []
    
    # 处理每一行
    for index, row in tqdm(df.iterrows(), total=len(df), desc="处理数据"):
        # 跳过含有缺失值的行
        if row.isnull().any():
            continue
        
        # 选择最佳接收机
        result = select_best_receivers(row)
        
        # 准备输出行数据
        output_row = [
            row['x'], row['y'], row['label'],
            # 三行TOA
            result['selected_toas'][0], result['selected_toas'][1], result['selected_toas'][2],
            # 三行反射阶数
            result['selected_ray_types'][0], result['selected_ray_types'][1], result['selected_ray_types'][2],
            # 三行接收机编号 (从1开始编号)
            result['selected_receivers'][0] + 1, result['selected_receivers'][1] + 1, result['selected_receivers'][2] + 1
        ]
        
        result_data.append(output_row)
        
        # 直射信号统计
        ray_types = result['all_ray_types']
        direct_rays = [i for i, rt in enumerate(ray_types) if rt == 0]
        selected_direct_rays = [r for r in result['selected_receivers'] if ray_types[r] == 0]
        
        # 计算得分
        direct_count = len(direct_rays)
        selected_direct_count = len(selected_direct_rays)
        
        # 场景分类和得分计算
        if direct_count >= 4:
            scenario = 'siege'  # 围剿
            max_score = 3      # 最多得3分
        elif direct_count == 3:
            scenario = 'battle'  # 苦战
            max_score = 3      # 最多得3分
        else:
            scenario = 'counter_siege'  # 反围剿
            max_score = direct_count  # 有多少个0就最多得多少分
        
        actual_score = selected_direct_count  # 选中几个直射信号就得几分
        
        # 更新全局统计
        total_max_score += max_score
        total_actual_score += actual_score
        original_avg_ray += np.mean(ray_types)
        selected_avg_ray += np.mean(result['selected_ray_types'])
        
        # 更新场景统计
        scenarios[scenario]['count'] += 1
        scenarios[scenario]['max_score'] += max_score
        scenarios[scenario]['actual_score'] += actual_score
        scenarios[scenario]['orig_ray'] += np.mean(ray_types)
        scenarios[scenario]['sel_ray'] += np.mean(result['selected_ray_types'])
    
    # 保存结果到Excel
    result_df = pd.DataFrame(result_data)
    result_df.to_excel(output_file, index=False, header=False)
    
    # 计算最终统计
    score_percentage = (total_actual_score / total_max_score * 100) if total_max_score > 0 else 0
    original_avg_ray /= total_rows
    selected_avg_ray /= total_rows
    ray_improvement = original_avg_ray - selected_avg_ray
    
    # 计算场景统计
    for scenario, stats in scenarios.items():
        if stats['count'] > 0:
            stats['score_percentage'] = (stats['actual_score'] / stats['max_score'] * 100) if stats['max_score'] > 0 else 0
            stats['orig_ray'] /= stats['count']
            stats['sel_ray'] /= stats['count']
            stats['ray_improvement'] = stats['orig_ray'] - stats['sel_ray']
    
    # 打印统计结果
    print("\n===== 得分评估结果 =====")
    print(f"总行数: {total_rows}")
    print(f"总得分: {total_actual_score}/{total_max_score} ({score_percentage:.2f}%)")
    print(f"原始平均反射阶数: {original_avg_ray:.2f}")
    print(f"选择后平均反射阶数: {selected_avg_ray:.2f}")
    print(f"阶数改善: {ray_improvement:.2f}")
    
    print("\n===== 场景分析 =====")
    # 检查各场景样本数总和
    total_scenario_count = sum(stats['count'] for stats in scenarios.values())
    if total_scenario_count != total_rows:
        print(f"警告: 场景样本数总和 ({total_scenario_count}) 与总行数 ({total_rows}) 不一致!")
    
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
    
    # 额外检查确保没有重复计算
    print("\n===== 场景分布确认 =====")
    print(f"围剿: {scenarios['siege']['count']} 行")
    print(f"苦战: {scenarios['battle']['count']} 行")
    print(f"反围剿: {scenarios['counter_siege']['count']} 行")
    print(f"总计: {total_scenario_count} 行 (总数据: {total_rows} 行)")
    
    print(f"\n结果已保存至 {output_file}")
    
    return {
        'score_percentage': score_percentage,
        'ray_improvement': ray_improvement,
        'scenarios': scenarios
    }

# 主函数
if __name__ == "__main__":
    input_file = r"D:\desktop\毕设材料\output_classifier.xlsx"
    output_file = r"D:\desktop\毕设材料\signature_filter.xlsx"
    
    # 处理整个数据集
    stats = process_dataset(input_file, output_file)
    
    # 如果需要测试单个样本
    #test_single_sample(input_file, 390)  # 测试第391行