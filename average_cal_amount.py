import csv

csv_filename = "12_0.8_test.csv"

column_name = 'calculation_amount'  # 要计算平均值的列名
values = []

with open(csv_filename, 'r', newline='', encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        # 获取指定列的值，并转换为浮点数
        value = row[column_name]
        if value != '':
            value = value.replace('%', '')
            values.append(float(value))

# 计算平均值，不转化为百分号
if values:
    average_calculation_amount = sum(values) / len(values)
    print(f'{column_name} 的平均值是：{average_calculation_amount:.3f}%')
