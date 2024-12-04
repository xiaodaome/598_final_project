from transformers import ViTImageProcessor, ViTForImageClassification
from transformers.models.vit.modeling_vit import ViTSelfAttention
from PIL import Image
import requests
import os
import scipy.io
import csv
import pandas as pd
import numpy as np

IMAG_NUM = 200  # 设置验证图片数量
correct_num = 0
error_records = []  # 用于记录错误信息
correct = []

# 定义CSV文件名 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
csv_filename = "16_0.8_test.csv"

# 打开CSV文件
csvfile = open(csv_filename, 'w', newline='', encoding='utf-8')
writer = csv.writer(csvfile)

# 写入表头
header = ['image_file'] + [f'alpha_{i}' for i in range(24)] + ['calculation_amount']
writer.writerow(header)
# 关闭CSV文件
csvfile.close()

# 设置验证集文件夹路径
VALIDATE_FOLDER     = "../ILSVRC2012_img_val"
ground_truth_file   = "../ILSVRC2012_devkit_t12/data/ILSVRC2012_validation_ground_truth.txt"
meta_file           = "../ILSVRC2012_devkit_t12/data/meta.mat"

# 获取文件夹中的所有图片
image_files = [f for f in os.listdir(VALIDATE_FOLDER) if f.endswith(".JPEG")]
image_files = sorted(image_files)[0:IMAG_NUM]  # 根据参数选择指定数量的图片

feature_extractor = ViTImageProcessor.from_pretrained('google/vit-large-patch16-384')
model = ViTForImageClassification.from_pretrained('google/vit-large-patch16-384')

# 获取模型中所有的ViTSelfAttention实例
def get_attention_modules(model):
    attention_modules = []
    for module in model.modules():
        if isinstance(module, ViTSelfAttention):
            attention_modules.append(module)
    return attention_modules

attention_modules = get_attention_modules(model)

meta_data = scipy.io.loadmat(meta_file)
# 假设 synsets 是 numpy 数组，且每个元素包含 ['ILSVRC2012_ID', 'WNID', 'words']

synsets = meta_data['synsets']
meta_dict = {
    int(entry['ILSVRC2012_ID'].item()): entry['words'][0]  # 索引 -> 文字描述
    for entry in synsets
}

# 读取文件内容
with open(ground_truth_file, "r") as f:
    ground_truth_labels = [int(line.strip()) for line in f]

# 验证图片
for i, image_file in enumerate(image_files, 1):
    image_path = os.path.join(VALIDATE_FOLDER, image_file)
    try:
        # 加载图片
        image = Image.open(image_path).convert("RGB")
        # 预处理图片
        inputs = feature_extractor(images=image, return_tensors="pt")
        # 推理
        outputs = model(**inputs)
        # 获取预测结果
        logits = outputs.logits
        predicted_class_idx = logits.argmax(-1).item()
        prediction = model.config.id2label[predicted_class_idx]


        # 获取真实标签
        image_label = ground_truth_labels[i - 1]
        description = meta_dict.get(image_label, "Unknown")

        # 比较预测结果与真实标签
        if model.config.id2label[predicted_class_idx] == description[0]:
            correct.append(1)
            correct_num += 1
        else:
            correct.append(0)
            error_records.append({
                "image_file": image_file,
                "predicted": model.config.id2label[predicted_class_idx],
                "ground_truth": description
            })

        print('accuracy rate = ', correct_num, ' / ', i )
        print()


    except Exception as e:
        error_records.append({
            "image_file": image_file,
            "error": str(e)
        })

# 写入正确与否
# 读取现有的 CSV 文件
with open(csv_filename, 'r', newline='', encoding='utf-8') as csvfile:
    reader = csv.reader(csvfile)
    rows = list(reader)

# 在表头中添加 'correct' 列
header = rows[0] + ['correct']
updated_rows = [header]

# 对于每一行，添加对应的 `correct` 值
for i, row in enumerate(rows[1:]):  # 跳过表头
    row_with_correct = row + [correct[i]]
    updated_rows.append(row_with_correct)

# 将更新后的内容写回 CSV 文件
with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(updated_rows)


# 计算准确率
accuracy = correct_num / IMAG_NUM
print('accuracy = ', accuracy)

column_name = 'calculation_amount'  # 要计算平均值的列名
values = []
with open(csv_filename, 'r', newline='', encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        # 获取指定列的值，并转换为浮点数
        value = row[column_name]
        # 移除百分号
        value = value.replace('%', '')
        if value != '':
            values.append(float(value))

# 计算平均值，转化为百分号
if values:
    average_calculation_amount = 0.01 * sum(values) / len(values)


# 读取现有的 CSV 文件
with open(csv_filename, 'r', newline='', encoding='utf-8') as csvfile:
    reader = csv.reader(csvfile)
    rows = list(reader)

# 在表头中添加 'correct_num' 和 'accuracy' 列
header = rows[0] + ['correct_num', 'IMAG_NUM', 'accuracy', 'Average_calculation_amount']
updated_rows = [header]

# 在表头下面添加一行，写入 correct_num 和 accuracy
# 其他列可以填充空字符串 ''
first_data_row = ['Summary:'] + [correct_num, IMAG_NUM, f'{accuracy:.2%}', f'{average_calculation_amount:.3%}']
updated_rows.append(first_data_row)

# 将原来的数据行添加回去
updated_rows.extend(rows[1:])

# 将更新后的内容写回 CSV 文件
with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(updated_rows)


# 打印错误记录
print("\nErrors and mismatches:")
for record in error_records:
    if "error" in record:
        print(f"Image: {record['image_file']}, Error: {record['error']}")
    else:
        print(f"Image: {record['image_file']}, Predicted: {record['predicted']}, Ground Truth: {record['ground_truth']}")

# print(model.config)