import pandas as pd
import ollama
from tqdm import tqdm

# 读取数据
data = pd.read_csv("posts.txt", sep='\t')
data = data[['post_text', 'label']].dropna()
data['label'] = data['label'].map({'fake': 0, 'real': 1})  # 转为数字标签

# 定义基础 Prompt 模板
def prompt_basic_classification(text):
    return f"""你是一名信息判别专家。请判断以下新闻是真新闻还是假新闻，仅回答"fake"或"real"。
新闻内容：{text}
你的判断是："""

def prompt_sentiment_analysis(text):
    return f"""请分析这段文本的情感倾向，是正面（positive）、中性（neutral）还是负面（negative）？
文本内容：{text}
情感倾向是："""

def prompt_with_sentiment(text, sentiment):
    return f"""你是一名信息判别专家。请根据以下新闻内容和它的情感倾向，判断其是真新闻还是假新闻，仅回答"fake"或"real"。
新闻内容：{text}
情感倾向：{sentiment}
你的判断是："""

# 使用模型接口
def get_response(prompt):
    response = ollama.chat(model='deepseek-r1:1.5b', messages=[{"role": "user", "content": prompt}])
    return response['message']['content'].strip().lower()

# 准确率统计函数
def evaluate(preds, labels):
    total = len(labels)
    correct = sum([p == l for p, l in zip(preds, labels)])
    fake_total = sum([l == 0 for l in labels])
    true_total = sum([l == 1 for l in labels])
    fake_correct = sum([p == l == 0 for p, l in zip(preds, labels)])
    true_correct = sum([p == l == 1 for p, l in zip(preds, labels)])
    return {
        "Accuracy": correct / total,
        "Accuracy_fake": fake_correct / fake_total if fake_total else 0,
        "Accuracy_true": true_correct / true_total if true_total else 0,
    }

# 第一轮：只用新闻文本
print("第一轮：真假分类中...")
preds1 = []
for text in tqdm(data['post_text']):
    res = get_response(prompt_basic_classification(text))
    preds1.append(0 if 'fake' in res else 1)
acc1 = evaluate(preds1, data['label'].tolist())
print("第一轮准确率：", acc1)

# 第二轮：情感分析
print("\n第二轮：情感分析中...")
sentiments = []
for text in tqdm(data['post_text']):
    res = get_response(prompt_sentiment_analysis(text))
    sentiments.append(res)

# 第三轮：真假分类 + 情感信息
print("\n第三轮：结合情感后进行真假分类中...")
preds3 = []
for text, sentiment in tqdm(zip(data['post_text'], sentiments), total=len(sentiments)):
    res = get_response(prompt_with_sentiment(text, sentiment))
    preds3.append(0 if 'fake' in res else 1)
acc3 = evaluate(preds3, data['label'].tolist())
print("第三轮准确率：", acc3)

# 比较提升
print("\n比较分析：")
print(f"初始准确率: {acc1['Accuracy']:.2%}")
print(f"情感增强后准确率: {acc3['Accuracy']:.2%}")
if acc3['Accuracy'] > acc1['Accuracy']:
    print("✅ 情感分析后准确率有所提升。")
else:
    print("⚠️ 情感分析未提升整体准确率，可能需优化 prompt 或模型参数。")
