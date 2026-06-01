import os
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch
import matplotlib
from googletrans import Translator

matplotlib.rcParams["font.sans-serif"] = ["SimHei"]
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt

# ---------------------- 设备设置 ----------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"当前使用设备: {device}")

# ---------------------- 加载BLIP模型 ----------------------
print("正在加载BLIP模型，请稍候...")
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)
model.eval()
print("模型加载完成！\n")

# ---------------------- 定义函数 ----------------------
def generate_caption(image):
    """生成英文描述"""
    inputs = processor(image, return_tensors="pt").to(device)
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=50)
    return processor.decode(output[0], skip_special_tokens=True)

def translate(text):
    translator = Translator()
    return translator.translate(text, src='en', dest='zh-cn').text

# ---------------------- 批量处理 images/ 目录下的所有图片 ----------------------
img_dir = "images"
results = []
exts = {".jpg", ".jpeg", ".png", ".bmp"}

# 获取所有图片文件
img_files = [f for f in os.listdir(img_dir) if os.path.splitext(f)[1].lower() in exts]

print(f"共找到 {len(img_files)} 张图片，开始处理...\n")

for fname in img_files:
    # 读取图片
    img_path = os.path.join(img_dir, fname)
    image = Image.open(img_path).convert("RGB")

    # 生成描述
    caption_en = generate_caption(image)
    caption_zh = translate(caption_en)

    # 打印结果
    print(f"[{fname}]")
    print(f"英文描述: {caption_en}")
    print(f"中文翻译: {caption_zh}\n")

    # 保存结果
    results.append((fname, caption_en, caption_zh))

    # 为每张图片生成可视化结果
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.imshow(image)
    ax.axis("off")
    ax.set_title(
        f"文件: {fname}\n英文: {caption_en}\n中文: {caption_zh}",
        fontsize=11, pad=10
    )
    plt.tight_layout()
    plt.savefig(f"result_{os.path.splitext(fname)[0]}.jpg", dpi=150, bbox_inches="tight")
    plt.close()  # 关闭当前图，避免内存占用

# ---------------------- 保存结果到文本文件 ----------------------
with open("results.txt", "w", encoding="utf-8") as f:
    for fname, en, zh in results:
        f.write(f"文件: {fname}\n")
        f.write(f"英文: {en}\n")
        f.write(f"中文: {zh}\n")
        f.write("-" * 40 + "\n")

print(f"全部完成！")
print(f"1. 可视化图片已保存为: result_*.jpg")
print(f"2. 文本结果已保存至: results.txt")