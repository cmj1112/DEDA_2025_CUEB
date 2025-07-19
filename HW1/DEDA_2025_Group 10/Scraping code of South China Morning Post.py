#!/usr/bin/env python
# coding: utf-8

# In[8]:


import requests
from bs4 import BeautifulSoup
import pandas as pd
import os
from datetime import datetime
from urllib.parse import urljoin

def 获取网页内容(url):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8'
    }
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        response.encoding = response.apparent_encoding
        return response.text
    except Exception as e:
        print(f"请求失败: {e}")
        return None

def 解析网页内容(html, base_url):
    soup = BeautifulSoup(html, 'html.parser')
    文章列表 = []
    
    # 查找所有可能是新闻链接的<a>标签
    for 链接 in soup.find_all('a', href=True):
        href = 链接['href']
        文本 = 链接.get_text().strip()
        
        # 筛选可能是新闻的链接（根据实际情况调整条件）
        if (
            len(文本) > 10 and  # 过滤过短的文本
            not href.startswith(('#', 'javascript:', 'mailto:')) and  # 过滤无效链接
            ('news' in href or 'article' in href or 'story' in href)  # 包含新闻关键词
        ):
            完整链接 = urljoin(base_url, href)
            文章列表.append([文本, 完整链接, datetime.now().strftime("%Y-%m-%d")])
    
    return 文章列表 if 文章列表 else [["未找到新闻链接", "", datetime.now().strftime("%Y-%m-%d")]]

def 保存到CSV(文件路径, 内容):
    try:
        目录 = os.path.dirname(文件路径) or '.'
        os.makedirs(目录, exist_ok=True)
        
        df = pd.DataFrame(内容, columns=["标题", "新闻链接", "提取日期"])
        df.to_csv(文件路径, encoding='utf_8_sig', index=False)
        print(f"已保存 {len(内容)} 条新闻链接到 {文件路径}")
    except Exception as e:
        print(f"保存文件时出错: {e}")

if __name__ == "__main__":
    目标网址 = "https://www.frontpages.com/south-china-morning-post/"
    html内容 = 获取网页内容(目标网址)
    
    if html内容:
        # 调试：保存原始HTML
        with open("debug.html", "w", encoding="utf-8") as f:
            f.write(html内容)
        
        解析结果 = 解析网页内容(html内容, 目标网址)
        保存路径 = os.path.join(os.getcwd(), "南华早报_新闻链接.csv")
        保存到CSV(保存路径, 解析结果)
        
        # 打印示例结果
        print("\n前5条新闻链接：")
        for i, row in enumerate(解析结果[:5], 1):
            print(f"{i}. {row[0]} - {row[1]}")
    else:
        print("无法获取网页内容")

