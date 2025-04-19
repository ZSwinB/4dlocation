import pandas as pd

# === 路径设置 ===
original_path = r"D:\desktop\毕设材料\processed_data.xlsx"        # 原始数据路径
filtered_path = r"D:\desktop\毕设材料\filtered_result.xlsx"           # 筛选后数据路径
output_path = r"D:\desktop\毕设材料\raytype_comparison.xlsx"           # 输出路径

# === 读取文件 ===
df_original = pd.read_excel(original_path, header=None)
df_filtered = pd.read_excel(filtered_path, header=None)

# 打印读取的前五行数据
print("原始数据 (前五行):")
print(df_original.head())
print("\n筛选后数据 (前五行):")
print(df_filtered.head())

result_rows = []

for idx in range(min(len(df_original), len(df_filtered))):
    # 原始数据中第 10~14 列为反射阶数（索引 9~13）
    original_raytypes = df_original.iloc[idx, 9:15].tolist()
    avg_all = sum(original_raytypes) / 6
    min3_avg = sum(sorted(original_raytypes)[:3]) / 3

    # 处理后的三个反射阶数：在 filtered_result 中第 6~8 列（索引 5~7）
    processed_raytypes = df_filtered.iloc[idx, 3:6].tolist()
    processed_avg = sum(processed_raytypes) / 3

    result_rows.append([avg_all, min3_avg, processed_avg])

    # 打印每一行的计算结果，前五行
    if idx < 5:
        print(f"\n第 {idx+1} 行计算：")
        print(f"原始数据反射阶数: {original_raytypes}")
        print(f"原始数据5个反射阶数的平均值: {avg_all}")
        print(f"原始数据3个最小反射阶数的平均值: {min3_avg}")
        print(f"处理后的三个反射阶数: {processed_raytypes}")
        print(f"处理后三个反射阶数的平均值: {processed_avg}")

# === 写入新 Excel：纯数据（不带标题）===
result_df = pd.DataFrame(result_rows)
result_df.to_excel(output_path, index=False, header=False)

print(f"\n反射阶数三列平均值比较已保存至：{output_path}")
