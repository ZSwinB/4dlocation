import pandas as pd

# === 路径设置 ===
original_path = r"D:\desktop\毕设材料\processed_data.xlsx"  # 原始数据路径
output_path = r"D:\desktop\毕设材料\worst.xlsx"    # 输出路径

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
    
    # 找到最小的三个值及其对应的索引位置（考虑重复值）
    # 创建(值, 原始索引)的元组列表
    indexed_values = [(val, i) for i, val in enumerate(raytypes_10_to_15)]
    
    # 按值排序
    indexed_values.sort(key=lambda x: x[0], reverse=True)
    
    # 取前三个最小值及其索引
    min_values_with_indices = indexed_values[:3]
    
    # 分离值和索引
    min_values_10_to_15 = [val for val, _ in min_values_with_indices]
    min_indices_10_to_15 = [idx for _, idx in min_values_with_indices]
    
    # 将最小的三个值添加到第4到第6列（正确位置）
    selected_row.extend(min_values_10_to_15)
    
    # 根据最小值的索引，反选出对应的第4到第9列的数据
    selected_raytypes_4_to_9 = [row[3 + i] for i in min_indices_10_to_15]
    
    # 将反选的数据添加到第7到第9列（正确位置）
    selected_row.extend(selected_raytypes_4_to_9)
    
    # 将筛选出的数据行添加到结果列表
    result_rows.append(selected_row)
    
for i in range(len(result_rows)):
    # 提取第4-6列 (索引3-5)
    cols_4_to_6 = result_rows[i][3:6]
    
    # 提取第7-9列 (索引6-8)
    cols_7_to_9 = result_rows[i][6:9]
    
    # 替换第4-6列为原来的第7-9列
    result_rows[i][3:6] = cols_7_to_9
    
    # 替换第7-9列为原来的第4-6列
    result_rows[i][6:9] = cols_4_to_6
# === 将结果保存为新的Excel文件 ===
result_df = pd.DataFrame(result_rows)
result_df.to_excel(output_path, index=False, header=False)

print(f"数据已保存至：{output_path}")