from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch
import matplotlib
from googletrans import Translator

matplotlib.rcParams["font.sans-serif"] = ["SimHei"]
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt

# 设备设置
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"当前使用设备: {device}")

# 加载BLIP模型
print("正在加载BLIP模型，请稍候...")
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)
model.eval()
print("模型加载完成！")

# 读取图片
img_path = "images/dog.jpg"
image = Image.open(img_path).convert("RGB")

# 生成英文描述
inputs = processor(image, return_tensors="pt").to(device)
with torch.no_grad():
    output = model.generate(**inputs, max_new_tokens=50)
caption_en = processor.decode(output[0], skip_special_tokens=True)
print(f"英文描述: {caption_en}")

# 条件生成
prompt = "a photo of"
inputs_cond = processor(image, prompt, return_tensors="pt").to(device)
with torch.no_grad():
    output_cond = model.generate(**inputs_cond, max_new_tokens=50)
caption_cond = processor.decode(output_cond[0], skip_special_tokens=True)
print(f"条件描述: {caption_cond}")

print("正在翻译为中文...")
translator = Translator()
caption_zh = translator.translate(caption_en, src='en', dest='zh-cn').text
print(f"中文翻译: {caption_zh}")

# 可视化并保存结果
fig, ax = plt.subplots(figsize=(7, 5))
ax.imshow(image)
ax.axis("off")
ax.set_title(
    f"英文: {caption_en}\n中文: {caption_zh}",
    fontsize=11, pad=10
)
plt.tight_layout()
plt.savefig("caption_result.jpg", dpi=150, bbox_inches="tight")
plt.show()
print("结果已保存至 caption_result.jpg")