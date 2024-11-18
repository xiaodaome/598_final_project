from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image
import requests
import os
import scipy.io
import numpy as np

IMAG_NUM = 10  # 设置验证图片数量
correct_num = 0
error_records = []  # 用于记录错误信息

# 设置验证集文件夹路径
VALIDATE_FOLDER     = "../ILSVRC2012_img_val"
ground_truth_file   = "../ILSVRC2012_devkit_t12/data/ILSVRC2012_validation_ground_truth.txt"
meta_file           = "../ILSVRC2012_devkit_t12/data/meta.mat"

# 获取文件夹中的所有图片
image_files = [f for f in os.listdir(VALIDATE_FOLDER) if f.endswith(".JPEG")]
image_files = sorted(image_files)[0:IMAG_NUM]  # 根据参数选择指定数量的图片

feature_extractor = ViTImageProcessor.from_pretrained('google/vit-large-patch16-384')
model = ViTForImageClassification.from_pretrained('google/vit-large-patch16-384')

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
            correct_num += 1
        else:
            error_records.append({
                "image_file": image_file,
                "predicted": model.config.id2label[predicted_class_idx],
                "ground_truth": description
            })

        print('accuracy rate = ', correct_num, ' / ', i )

    except Exception as e:
        error_records.append({
            "image_file": image_file,
            "error": str(e)
        })

# 计算准确率
accuracy = correct_num / IMAG_NUM
print('accuracy = ', accuracy)

# 打印错误记录
print("\nErrors and mismatches:")
for record in error_records:
    if "error" in record:
        print(f"Image: {record['image_file']}, Error: {record['error']}")
    else:
        print(f"Image: {record['image_file']}, Predicted: {record['predicted']}, Ground Truth: {record['ground_truth']}")

# print(model.config)