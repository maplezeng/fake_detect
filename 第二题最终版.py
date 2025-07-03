import pandas as pd
import re
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from gensim import corpora, models
from wordcloud import WordCloud
import pyLDAvis.gensim_models
import pyLDAvis
import numpy as np
import ollama

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# -------- 1. 数据准备 --------
data = pd.read_csv("1(1).txt", sep='\t')
texts = data['post_text'].dropna().tolist()

# -------- 2. 数据预处理 --------
nltk.download('stopwords')
nltk.download('wordnet')
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
def remove_urls(text):
    return re.sub(r'http\S+', '', text)
def preprocess(text):
    text = remove_urls(text)
    text = re.sub(r'[^a-zA-Z]', ' ', text).lower()
    words = text.split()
    return [lemmatizer.lemmatize(w) for w in words if w not in stop_words and len(w) > 2]

processed_texts = [preprocess(t) for t in texts]

# -------- 打印预处理结果 --------
for i, (original, processed) in enumerate(zip(texts, processed_texts)):
    print(f"\n【原始文本 {i+1}】:\n{original}")
    print(f"【预处理后】:\n{' '.join(processed)}")


# -------- 3. 构建词袋模型并训练LDA --------
# 创建词典和语料库
dictionary = corpora.Dictionary(processed_texts)
corpus = [dictionary.doc2bow(text) for text in processed_texts]
# 训练LDA模型（设置主题数为3）
lda_model = models.LdaModel(corpus=corpus,
                            id2word=dictionary,
                            num_topics=3,
                            passes=10,
                            random_state=42)
# 打印主题关键词
print("LDA主题关键词：")
for idx, topic in lda_model.print_topics(num_words=10):
    print(f"主题 {idx + 1}: {topic}")

# -------- 4. 可视化分析 --------
# 4.1 LDA交互图
vis = pyLDAvis.gensim_models.prepare(lda_model, corpus, dictionary)
pyLDAvis.save_html(vis, 'lda_twitter_topics.html')
print("✅ 已保存交互式LDA可视化为 lda_twitter_topics.html，请用浏览器打开查看")

# 4.2 词云图
for t in range(lda_model.num_topics):
    plt.figure()
    wc = WordCloud(background_color='white').fit_words(dict(lda_model.show_topic(t, 30)))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis("off")
    plt.title(f"主题 {t + 1}")
    plt.show()

# 4.3 热力图
doc_topic_matrix = [lda_model.get_document_topics(bow) for bow in corpus]
matrix = np.zeros((len(corpus), lda_model.num_topics))

for doc_idx, doc in enumerate(doc_topic_matrix):
    for topic_id, prob in doc:
        matrix[doc_idx][topic_id] = prob

sns.heatmap(matrix, cmap='YlGnBu', xticklabels=[f"主题{i + 1}" for i in range(lda_model.num_topics)])
plt.xlabel("主题")
plt.ylabel("文档索引")
plt.title("文档-主题概率分布热力图")
plt.show()


# -------- 5. 使用大模型DeepSeek-R1做主题内容总结 --------
def analyze_topic_keywords(topic_words, topic_id):
    keywords = ", ".join([w for w, _ in topic_words])
    prompt = f"""请根据以下关键词总结一个Twitter主题的主要内容，不超过150字：
关键词：{keywords}"""

    response = ollama.chat(
        model='deepseek-r1:1.5b',
        messages=[{"role": "user", "content": prompt}]
    )
    print(f"\n🧠 主题 {topic_id + 1} 智能总结：\n{response['message']['content']}\n")


for t in range(lda_model.num_topics):
    topic_words = lda_model.show_topic(t, topn=10)
    analyze_topic_keywords(topic_words, t)
