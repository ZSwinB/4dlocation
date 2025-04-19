import pandas as pd

# 文件路径
source_file = r"D:\desktop\毕设材料\idealdata.xlsx"
replacement_file = r"D:\desktop\毕设材料\output_classifier.xlsx"
output_file = r"D:\desktop\毕设材料\idealdata_label.xlsx"

# 读取两个Excel文件
# 假设这些文件没有表头，所以设置header=None
source_data = pd.read_excel(source_file, header=None)
replacement_data = pd.read_excel(replacement_file, header=None)

# 检查行数是否匹配
if len(source_data) != len(replacement_data):
    print(f"警告：两个文件的行数不匹配！源文件有{len(source_data)}行，替换文件有{len(replacement_data)}行。")
    print("将继续处理，但可能会导致数据不一致。")

# 打印前5行数据进行查看
print("源文件前5行:")
print(source_data.head())
print("\n替换文件前5行:")
print(replacement_data.head())

# 替换第三列（索引为2，因为索引从0开始）
source_data.iloc[:, 2] = replacement_data.iloc[:, 2]

# 打印替换后的前5行数据
print("\n替换后的数据前5行:")
print(source_data.head())

# 保存更新后的数据到新文件
source_data.to_excel(output_file, index=False, header=False)

print(f"\n已完成替换并保存到新文件: {output_file}")