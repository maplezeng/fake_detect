import re
import matplotlib.pyplot as plt

# Step 1: 读取文件
with open('gossip_record.txt', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Step 2: 初始化容器
epochs = []
val_accs = []
img_accs = []
text_accs = []
vgg_accs = []

real_precision = []
real_recall = []
real_f1 = []

fake_precision = []
fake_recall = []
fake_f1 = []

# 新增：loss容器
mix_loss = []
image_loss = []
text_loss = []
vgg_loss = []

# Step 3: 提取指标
for i, line in enumerate(lines):
    if "Epoch [" in line and "Val_Acc" in line:
        epoch_match = re.search(r"Epoch \[(\d+)/", line)
        acc_match = re.search(r"Val_Acc: ([0-9.]+)", line)
        if epoch_match and acc_match:
            epochs.append(int(epoch_match.group(1)))
            val_str = acc_match.group(1).strip().rstrip('.')  # 清除尾部 .
            val_accs.append(float(val_str))

    if "Single Modalities Accuracy" in line:
        accs = re.findall(r"[0-9.]+", line)
        if len(accs) == 3:
            img_accs.append(float(accs[0]))
            text_accs.append(float(accs[1]))
            vgg_accs.append(float(accs[2]))

    if "------Real News -----------" in line:
        precision = float(re.search(r"\[([0-9.]+)\]", lines[i+1]).group(1))
        recall = float(re.search(r"\[([0-9.]+)\]", lines[i+2]).group(1))
        f1 = float(re.search(r"\[([0-9.]+)\]", lines[i+4]).group(1))
        real_precision.append(precision)
        real_recall.append(recall)
        real_f1.append(f1)

    if "------Fake News -----------" in line:
        precision = float(re.search(r"\[([0-9.]+)\]", lines[i+1]).group(1))
        recall = float(re.search(r"\[([0-9.]+)\]", lines[i+2]).group(1))
        f1 = float(re.search(r"\[([0-9.]+)\]", lines[i+4]).group(1))
        fake_precision.append(precision)
        fake_recall.append(recall)
        fake_f1.append(f1)

    # 新增：提取loss曲线
    if "mix_loss:" in line and "image_loss:" in line:
        try:
            mix = float(re.search(r"mix_loss: ([0-9.]+)", line).group(1))
            img_l = float(re.search(r"image_loss: ([0-9.]+)", line).group(1))
            txt_l = float(re.search(r"text_loss: ([0-9.]+)", line).group(1))
            vgg_l = float(re.search(r"vgg_loss: ([0-9.]+)", line).group(1))
            mix_loss.append(mix)
            image_loss.append(img_l)
            text_loss.append(txt_l)
            vgg_loss.append(vgg_l)
        except:
            pass

# Step 4: 可视化
plt.figure(figsize=(18, 12))

# 1. 模态准确率变化
plt.subplot(3, 2, 1)
plt.plot(epochs, img_accs, label='Image Acc', marker='o')
plt.plot(epochs, text_accs, label='Text Acc', marker='o')
plt.plot(epochs, vgg_accs, label='VGG Acc', marker='o')
plt.title("Single Modality Accuracy per Epoch")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)

# 2. 验证准确率变化
plt.subplot(3, 2, 2)
plt.plot(epochs, val_accs, label='Val Accuracy', color='purple', marker='s')
plt.title("Validation Accuracy per Epoch")
plt.xlabel("Epoch")
plt.ylabel("Val_Acc")
plt.legend()
plt.grid(True)

# 3. Fake vs Real 的 F1 对比
plt.subplot(3, 2, 3)
plt.plot(epochs, real_f1, label='F1 Real', marker='o')
plt.plot(epochs, fake_f1, label='F1 Fake', marker='o')
plt.title("F1 Score: Real vs Fake")
plt.xlabel("Epoch")
plt.ylabel("F1 Score")
plt.legend()
plt.grid(True)

# 4. Fake vs Real 的 Recall 对比
plt.subplot(3, 2, 4)
plt.plot(epochs, real_recall, label='Recall Real', marker='x')
plt.plot(epochs, fake_recall, label='Recall Fake', marker='x')
plt.title("Recall: Real vs Fake")
plt.xlabel("Epoch")
plt.ylabel("Recall")
plt.legend()
plt.grid(True)

# 新增 5. Fake vs Real Precision 对比
plt.subplot(3, 2, 5)
plt.plot(epochs, real_precision, label='Precision Real', marker='^')
plt.plot(epochs, fake_precision, label='Precision Fake', marker='^')
plt.title("Precision: Real vs Fake")
plt.xlabel("Epoch")
plt.ylabel("Precision")
plt.legend()
plt.grid(True)

# 新增 6. Loss 曲线
plt.subplot(3, 2, 6)
plt.plot(range(len(mix_loss)), mix_loss, label='mix_loss')
plt.plot(range(len(image_loss)), image_loss, label='image_loss')
plt.plot(range(len(text_loss)), text_loss, label='text_loss')
plt.plot(range(len(vgg_loss)), vgg_loss, label='vgg_loss')
plt.title("Loss Curve (per Eval Step)")
plt.xlabel("Evaluation Step")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
