import pandas as pd
import random

# 设置随机种子以确保结果可重现（如果需要）
random.seed(42)

# === 路径设置 ===
original_path = r"D:\desktop\毕设材料\6\classifier_noisy.xlsx"  # 原始数据路径
output_path = r"D:\desktop\毕设材料\6\random_noisy.xlsx"    # 输出路径

# === 读取文件 ===
df_original = pd.read_excel(original_path, header=None)  # 不读取列标签

# 存储筛选后的结果
result_rows = []

# 遍历每一行
for idx, row in df_original.iterrows():
    # 提取前三列（索引0到2）
    selected_row = row[:3].tolist()
    
    # 选取第10到第15列（索引9到14）
    raytypes_10_to_15 = row[9:15].tolist()
    
    # 随机选择三个值及其对应的索引
    # 创建索引列表
    indices = list(range(len(raytypes_10_to_15)))
    
    # 随机打乱索引
    random.shuffle(indices)
    
    # 取前三个随机索引
    random_indices = indices[:3]
    
    # 根据随机索引获取对应的值
    random_values = [raytypes_10_to_15[i] for i in random_indices]
     # 根据随机选择的索引，反选出对应的第4到第9列的数据
    selected_raytypes_4_to_9 = [row[3 + i] for i in random_indices]  
        # 将反选的数据添加到结果行
    selected_row.extend(selected_raytypes_4_to_9)
    # 将随机选择的三个值添加到结果行
    selected_row.extend(random_values)
    #索引加入到结果行
    selected_row.extend([i + 1 for i in random_indices])

    
    # 将反选的数据添加到结果行
    selected_row.extend(selected_raytypes_4_to_9)
    
    # 将筛选出的数据行添加到结果列表
    result_rows.append(selected_row)


# === 将结果保存为新的Excel文件 ===
result_df = pd.DataFrame(result_rows)
result_df.to_excel(output_path, index=False, header=False)

print(f"数据已保存至：{output_path}")