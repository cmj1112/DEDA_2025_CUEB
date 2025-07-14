import numpy as np
import pandas as pd
import os
import re
from bs4 import BeautifulSoup as soup
import requests
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import matplotlib.pyplot as plt
import pyLDAvis
import warnings

warnings.filterwarnings("ignore")


def crawl_and_preprocess():
    base_link = 'http://www.wiwi.hu-berlin.de/de/forschung/irtg/results/discussion-papers'
    abs_link = 'https://www.wiwi.hu-berlin.de/de/forschung/irtg/results/'

    # 初始化 abstract_all 列表
    abstract_all = []

    # 获取论文信息
    try:
        request_result = requests.get(base_link, headers={'Connection': 'close'})
        request_result.raise_for_status()  # 检查请求是否成功
        parsed = soup(request_result.content)
        tr_items = parsed.find_all('tr')
        info_list = []

        for item in tr_items:
            link_list = item.find_all('td')
            try:
                paper_title = re.sub(pattern=r'\s+', repl=' ', string=link_list[1].text.strip())
                author = link_list[2].text
                date_of_issue = link_list[3].text
                abstract_link = link_list[5].find('a')['href']
                info_list.append([paper_title, author, date_of_issue, abstract_link])
            except Exception as e:
                print(f"Skipping row due to error: {e}")
                continue

        # 获取摘要文本
        for paper in info_list:
            print(f"Processing: {paper[0]}")
            try:
                paper_abstract_page = requests.get(paper[3], headers={'Connection': 'close'})
                paper_abstract_page.raise_for_status()

                if paper_abstract_page.status_code == 200:
                    abstract_parsed = soup(paper_abstract_page.content)
                    main_part = abstract_parsed.find_all('div', attrs={'id': r'content-core'})[0].text.strip()

                    # 文本清洗
                    main_part = re.sub(r'.+?[Aa]bstract', 'Abstract', main_part)
                    main_part = re.sub(r'JEL [Cc]lassification:.*', '', main_part)
                    main_part = re.sub(r'[A-Za-z][0-9][0-9]?', '', main_part)
                    main_part = re.sub('[\r\n]+', ' ', main_part)

                    abstract_all.append(main_part)
                else:
                    print(f"Failed to access: {paper[3]}")
            except Exception as e:
                print(f"Error processing {paper[0]}: {e}")
                continue

    except Exception as e:
        print(f"Error during crawling: {e}")
        return []

    return abstract_all


def perform_lda(abstracts, n_topics=5, max_features=1000):
    # 文本向量化
    tf_vectorizer = CountVectorizer(
        max_df=0.95,
        min_df=2,
        max_features=max_features,
        stop_words='english'
    )
    tf = tf_vectorizer.fit_transform(abstracts)

    # 训练LDA模型
    lda = LatentDirichletAllocation(
        n_components=n_topics,
        max_iter=5,
        learning_method='online',
        learning_offset=50.,
        random_state=0
    )
    lda.fit(tf)

    return lda, tf_vectorizer, tf


def visualize_lda(lda_model, tf_vectorizer, tf):
    # 获取必要的统计量
    vocab = tf_vectorizer.get_feature_names_out()
    term_frequency = np.asarray(tf.sum(axis=0)).ravel()
    doc_length = np.asarray(tf.sum(axis=1)).ravel()

    # 准备可视化数据
    panel = pyLDAvis.prepare(
        topic_term_dists=lda_model.components_,
        doc_topic_dists=lda_model.transform(tf),
        doc_lengths=doc_length,
        vocab=vocab,
        term_frequency=term_frequency,
        mds='tsne'
    )

    pyLDAvis.save_html(panel, 'lda_visualization.html')

    print("\nTopics and their top words:")
    for topic_idx, topic in enumerate(lda_model.components_):
        print(f"Topic #{topic_idx + 1}:")
        print(" ".join([vocab[i] for i in topic.argsort()[:-10 - 1:-1]]))
        print()


def main():
    print("Starting LDA analysis for IRTG1792 papers...")

    # 1. 爬取和预处理数据
    print("\nStep 1: Crawling and preprocessing abstracts...")
    abstracts = crawl_and_preprocess()
    print(f"\nTotal abstracts collected: {len(abstracts)}")

    if len(abstracts) == 0:
        print("Error: No abstracts found. Check network connection or website structure.")
        return

    # 2. 执行LDA分析
    print("\nStep 2: Performing LDA analysis...")
    n_topics = 5  # 可调整主题数量
    lda_model, tf_vectorizer, tf = perform_lda(abstracts, n_topics=n_topics)

    # 3. 可视化结果
    print("\nStep 3: Visualizing results...")
    visualize_lda(lda_model, tf_vectorizer, tf)
    print("\nVisualization saved to 'lda_visualization.html'")

    print("\nLDA analysis completed!")


if __name__ == "__main__":
    main()
