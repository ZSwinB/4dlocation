import pandas as pd
import numpy as np

# 文件路径
file1_path = "D:/desktop/毕设材料/f_with_classifier.xlsx"
file2_path = "D:/desktop/毕设材料/filtered_result.xlsx"

# 加载Excel文件，不使用表头
df1 = pd.read_excel(file1_path, header=None)
df2 = pd.read_excel(file2_path, header=None)

print(f"第一个文件形状: {df1.shape}")
print(f"第二个文件形状: {df2.shape}")

# 检查两个文件是否有足够的列
if df1.shape[1] < 6 or df2.shape[1] < 6:
    print("警告: 至少一个文件列数少于6列")
else:
    # 针对第一个文件的统计
    print("\n文件1: f_add_classifier.xlsx")
    # 计算第4-6列(索引3-5)的平均值
    col_sums1 = df1.iloc[:, 3:6].sum(axis=1)
    avg_sum1 = col_sums1.mean()
    print(f"4-6列和的平均值: {avg_sum1:.8e}")
    
    # 计算第4-6列中包含的0的个数
    zeros_count1 = (df1.iloc[:, 3:6] == 0).sum().sum()
    print(f"4-6列中包含的0的个数: {zeros_count1}")
    
    # 每列零值统计
    for i in range(3, 6):
        col_zeros = (df1.iloc[:, i] == 0).sum()
        print(f"  列 {i+1} 中0的个数: {col_zeros}")
    
    # 针对第二个文件的统计
    print("\n文件2: filtered_result.xlsx")
    # 计算第4-6列(索引3-5)的平均值
    col_sums2 = df2.iloc[:, 3:6].sum(axis=1)
    avg_sum2 = col_sums2.mean()
    print(f"4-6列和的平均值: {avg_sum2:.8e}")
    
    # 计算第4-6列中包含的0的个数
    zeros_count2 = (df2.iloc[:, 3:6] == 0).sum().sum()
    print(f"4-6列中包含的0的个数: {zeros_count2}")
    
    # 每列零值统计
    for i in range(3, 6):
        col_zeros = (df2.iloc[:, i] == 0).sum()
        print(f"  列 {i+1} 中0的个数: {col_zeros}")
    
    # 比较结果
    print("\n比较结果:")
    diff_avg = avg_sum1 - avg_sum2
    diff_zeros = zeros_count1 - zeros_count2
    print(f"4-6列和的平均值差异: {diff_avg:.8e}")
    print(f"4-6列中0的个数差异: {diff_zeros}")
    
    # 检查是否存在非常小的值(可能是接近0但不完全是0的值)
    print("\n检查极小值:")
    small_threshold = 1e-10
    
    small_values1 = ((df1.iloc[:, 3:6].abs() > 0) & (df1.iloc[:, 3:6].abs() < small_threshold)).sum().sum()
    print(f"文件1中4-6列中绝对值小于{small_threshold}的非零值个数: {small_values1}")
    
    small_values2 = ((df2.iloc[:, 3:6].abs() > 0) & (df2.iloc[:, 3:6].abs() < small_threshold)).sum().sum()
    print(f"文件2中4-6列中绝对值小于{small_threshold}的非零值个数: {small_values2}")
    
    # 检查每行和是否为0
    zero_sum_rows1 = (col_sums1 == 0).sum()
    zero_sum_rows2 = (col_sums2 == 0).sum()
    print(f"\n文件1中4-6列和为0的行数: {zero_sum_rows1}")
    print(f"文件2中4-6列和为0的行数: {zero_sum_rows2}")
    
    # 计算基本统计量
    print("\n基本统计量:")
    print("文件1 4-6列统计:")
    print(df1.iloc[:, 3:6].describe())
    print("\n文件2 4-6列统计:")
    print(df2.iloc[:, 3:6].describe())