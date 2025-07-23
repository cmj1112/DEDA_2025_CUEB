# -*- coding: utf-8 -*-
"""
最基础摘要词云：无图案、无透明、自动常见英文停用词和自定义补充
"""

import re
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt

# 1. 路径设置
abstract_path = '/Users/liboyang/PycharmProjects/pythonProject1/暑期课程/Abstract_all.txt'
output_path = '/Users/liboyang/PycharmProjects/pythonProject1/暑期课程/wordcloud_abstract_rect.png'

# 2. 读取与预处理
with open(abstract_path, 'r', encoding='utf-8') as f:
    text = f.read()
text = re.sub(r'\n', ' ', text)
text = re.sub(r'[^a-zA-Z0-9 ]', '', text)
text = text.lower()

# 3. 停用词设置（可扩展）
stopwords = set(STOPWORDS)
stopwords.update({'abstract', 'keywords', 'sep', 'using', 'study', 'result', 'proposed', 'paper', 'one', 'method', 'model'})

# 4. 构建与生成词云
wc = WordCloud(
    background_color='white',
    width=800,
    height=400,
    stopwords=stopwords,
    max_words=200,
    colormap='viridis'
)
wc.generate(text)

# 5. 显示和保存
plt.figure(figsize=(12, 6))
plt.imshow(wc, interpolation='bilinear')
plt.axis('off')
plt.tight_layout(pad=0)
plt.savefig(output_path, dpi=300)
plt.show()

print('矩形学术词云已生成：', output_path)
