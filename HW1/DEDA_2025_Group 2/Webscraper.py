#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 01:48:27 2024

@author: Hua LEI, Wolfgang Karl Härdle
Improved on Wed Sep 9 01:29:20 2025

@author: Zeyuan WANG
"""
# Load modules
import requests
from bs4 import BeautifulSoup as soup
# Receiving source code from the China Daily website
scmp_url = 'https://www.chinadaily.com.cn/business/economy'
url_request = requests.get(scmp_url)
# Returns the content of the response
url_content = url_request.content

# Using BeautifulSoup to parse webpage source code
parsed_content = soup(url_content, 'html.parser')
# print(parsed_content)
# Find all news sections
filtered_parts0 = parsed_content.find_all('div', class_="mb10 tw3_01_2")
filtered_parts1 = parsed_content.find_all('span', class_="tw3_01_2_t")
# print(filtered_parts1)
page_info = []

# For loop iterates over every line in text
for section in filtered_parts1:
    #print(section)
    unit_info = {}
    # (1) Filter title
    filtered_part1 = section.find_all('a', shape="rect")
    filtered_part2 = section.find_all('b')
    # (2) Extract the title and link from the section
    news_title = filtered_part1[0].text.strip()
    news_link = filtered_part1[0].get('href').strip()
    news_date= filtered_part2[0].text.strip()
    news_link = f"https:{news_link}"   # adjust the relative link
    # (3) Add all info into the dictionary
    unit_info['news_title'] = news_title
    unit_info['news_link'] = news_link
    unit_info["news_date"] = news_date
    page_info.append(unit_info)

# print(filtered_parts)

import pandas as pd
import openpyxl
# Calling DataFrame constructor on our list
df = pd.DataFrame(page_info, columns=['news_title', 'news_link', 'news_date'])
print(df)

# Exporting to .csv file
df.to_excel(r'C:\Work\DEDA\pythonProject\Chinadaily_business_Scraped_News0709.xlsx', index=False)
