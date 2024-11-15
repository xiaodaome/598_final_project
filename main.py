from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image
import requests


url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
image = Image.open(requests.get(url, stream=True).raw)
# image = Image.open(".jpg")
feature_extractor = ViTImageProcessor.from_pretrained('google/vit-large-patch16-384')
model = ViTForImageClassification.from_pretrained('google/vit-large-patch16-384')
print(model.config)
inputs = feature_extractor(images=image, return_tensors="pt")
outputs = model(**inputs)
logits = outputs.logits
predicted_class_idx = logits.argmax(-1).item()
print("Predicted class:", model.config.id2label[predicted_class_idx])
