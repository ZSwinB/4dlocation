import pandas as pd

# 路径
original_path = r"D:\desktop\毕设材料\processed_data.xlsx"
filtered_path = r"D:\desktop\毕设材料\filtered_result.xlsx"
output_path = r"D:\desktop\毕设材料\data4trainning.xlsx"

# 读取两个文件（无表头）
df_original = pd.read_excel(original_path, header=None)
df_filtered = pd.read_excel(filtered_path, header=None)

# 取原始数据前三列
prefix = df_original.iloc[:, :3]

# 拼接：把 prefix 放在 filtered 前面
merged_df = pd.concat([prefix, df_filtered], axis=1)

# 保存为纯数据（无索引，无列名）
merged_df.to_excel(output_path, index=False, header=False)

print(f"已成功拼接，文件保存至：{output_path}")
