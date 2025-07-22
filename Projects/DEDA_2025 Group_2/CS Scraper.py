import time
import random
import pandas as pd
import os
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException, WebDriverException
from bs4 import BeautifulSoup
import json
from tqdm import tqdm
import sys

# 配置参数
BASE_URL = "https://buff.163.com"
MARKET_URL = "https://buff.163.com/market/csgo"
CATEGORY_GROUP = "knife"  # 匕首分类
MAX_PAGES = 10
MAX_PRODUCTS = 2000

# 账号配置
USERNAME = "15515935162"  # 替换为您的网易账号
PASSWORD = "060317Wzy#0058"  # 替换为您的密码

# Chrome驱动配置
CHROME_DRIVER_PATH = r"C:\Work\ChromeDriver\chromedriver-win64\chromedriver.exe"  # 确保这个文件在当前目录下
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"


def create_browser():
    print("正在初始化浏览器...")
    chrome_options = Options()

    # 添加优化参数
    chrome_options.add_argument(f"user-agent={USER_AGENT}")
    chrome_options.add_argument("--disable-blink-features=AutomationControlled")
    chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
    chrome_options.add_experimental_option('useAutomationExtension', False)
    chrome_options.add_argument("--disable-infobars")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--log-level=3")
    chrome_options.add_argument("--start-maximized")

    try:
        service = Service(executable_path=CHROME_DRIVER_PATH)
        browser = webdriver.Chrome(service=service, options=chrome_options)

        # 隐藏自动化特征
        browser.execute_script(
            "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"
        )

        print("浏览器初始化成功")
        return browser

    except WebDriverException as e:
        print(f"创建浏览器实例时出错: {e}")
        print(f"请确保 {CHROME_DRIVER_PATH} 存在且与您的Chrome版本兼容")
        return None


# 简化版登录函数 - 不检测登录表单
def simple_login(browser):
    if not browser:
        print("浏览器未初始化")
        return False

    print("正在导航至登录页面...")
    browser.get(f"https://buff.163.com/account/login?back_url=/market/csgo")

    # 直接尝试填写表单而不检测元素
    print("尝试填写登录表单...")
    try:
        # 使用更宽松的等待
        time.sleep(3)

        # 尝试填写用户名
        username_fields = browser.find_elements(By.ID, "j_login_email")
        if username_fields:
            username_fields[0].send_keys(USERNAME)
            print("已输入用户名")
        else:
            print("未找到用户名输入框")

        # 尝试填写密码
        password_fields = browser.find_elements(By.ID, "j_login_pwd")
        if password_fields:
            password_fields[0].send_keys(PASSWORD)
            print("已输入密码")
        else:
            print("未找到密码输入框")

        # 尝试点击登录按钮
        login_buttons = browser.find_elements(By.ID, "e_login_btn")
        if login_buttons:
            login_buttons[0].click()
            print("已点击登录按钮")
        else:
            print("未找到登录按钮")

        # 等待登录完成 - 不进行严格检测
        print("等待登录完成...")
        time.sleep(5)

        # 检查是否重定向到市场页面
        if "market" in browser.current_url:
            print("登录成功！已重定向到市场页面")
            return True

        # 检查是否有验证码
        if browser.find_elements(By.CLASS_NAME, "verify-box"):
            print("检测到验证码，请手动完成验证")
            input("完成验证后按回车键继续...")
            return True

        print("登录状态不确定，继续尝试访问市场页面...")
        return True

    except Exception as e:
        print(f"登录过程中出错: {e}")
        return False


# 获取商品列表
def get_product_list(browser):
    if not browser:
        return []

    products = []

    # 遍历每一页
    for page_num in range(1, MAX_PAGES + 1):
        print(f"正在处理第 {page_num} 页...")

        # 构建带页码的URL
        page_url = f"{MARKET_URL}#game=csgo&page_num={page_num}&category_group={CATEGORY_GROUP}&tab=top-bookmarked"
        print(f"导航到: {page_url}")
        browser.get(page_url)

        # 等待商品加载
        time.sleep(3)

        # 解析商品列表
        html = browser.page_source
        soup = BeautifulSoup(html, 'html.parser')

        # 查找所有商品项
        items = soup.select('a[href^="/goods/"]')
        if not items:
            print("未找到商品，可能页面结构已更改")
            break

        print(f"找到 {len(items)} 个商品")

        # 修改商品列表提取部分
        for item in items:
            try:
                href = item['href']
                title = item.get('title', '') or (
                            item.select_one('.detail .name') and item.select_one('.detail .name').get_text(
                        strip=True)) or ""

                # 提取商品ID
                goods_id = href.split('/')[-1].split('?')[0].split('#')[0]

                if goods_id and title:
                    # 构建干净的URL
                    base_goods_url = f"{BASE_URL}/goods/{goods_id}"
                    goods_url = base_goods_url  # 商品页面
                    history_url = base_goods_url + "#tab=history"  # 历史价格页面

                    products.append({
                        'goods_id': goods_id,
                        'title': title.strip(),
                        'url': goods_url,
                        'history_url': history_url
                    })
                    print(f"添加商品: {title[:30]} (ID: {goods_id})")

            except Exception as e:
                print(f"解析商品时出错: {e}")

        # 如果达到最大商品数则停止
        if len(products) >= MAX_PRODUCTS:
            print(f"已达到最大商品数限制 ({MAX_PRODUCTS})")
            break



    print(f"共找到 {len(products)} 个商品")
    return products[:MAX_PRODUCTS]


# 获取商品价格历史
def get_price_history(browser, product):
    if not browser:
        return []

    print(f"正在获取商品 {product['goods_id']} 的价格历史...")
    browser.get(product['history_url'])
    print("当前历史价格页面:", browser.current_url)

    # 等待历史价格加载
    time.sleep(3)

    # 解析历史价格数据
    html = browser.page_source
    soup = BeautifulSoup(html, 'html.parser')

    # 查找历史价格表格
    history_table = soup.find('div', class_='detail-tab-cont')
    if not history_table:
        print(f"未找到商品 {product['goods_id']} 的历史价格表格")
        return []

    # 提取价格数据
    history = []
    rows = history_table.select('tbody tr')

    if not rows:
        print("未找到历史价格行")
        return []

    print(f"找到 {len(rows)} 条历史价格记录")

    for row in rows:
        try:
            date_elem = row.select_one('td:nth-child(6)')
            price_elem = row.select_one('td:nth-child(4) strong') or row.select_one('td:nth-child(4)')

            if date_elem and price_elem:
                date = date_elem.get_text(strip=True)
                price = price_elem.get_text(strip=True)
                history.append({'date': date, 'price': price})
                print(f"添加价格记录: {date} - {price}")
        except Exception as e:
            print(f"解析价格行时出错: {e}")

    return history


# 处理单个商品
def process_product(browser, product):
    if not browser:
        return product

    try:
        # 获取历史价格
        price_history = get_price_history(browser, product)

        # 添加历史价格到商品数据
        product['price_history'] = price_history

        # 添加统计信息
        if price_history:
            try:
                prices = []
                for item in price_history:
                    try:
                        price_str = item['price'].replace('¥', '').replace(',', '').strip()
                        if price_str:  # 确保不是空字符串
                            prices.append(float(price_str))
                    except ValueError:
                        print(f"无法转换价格: {item['price']}")

                if prices:
                    product['history_min'] = min(prices)
                    product['history_max'] = max(prices)
                    product['history_avg'] = sum(prices) / len(prices)
                    product['history_points'] = len(prices)
                else:
                    product['history_min'] = None
                    product['history_max'] = None
                    product['history_avg'] = None
                    product['history_points'] = 0
            except Exception as e:
                print(f"计算价格统计信息时出错: {e}")
                product['history_min'] = None
                product['history_max'] = None
                product['history_avg'] = None
                product['history_points'] = 0
        else:
            product['history_min'] = None
            product['history_max'] = None
            product['history_avg'] = None
            product['history_points'] = 0

        # 添加时间戳
        product['timestamp'] = time.strftime('%Y-%m-%d %H:%M:%S')

        # 随机延迟
        sleep_time = random.uniform(2.0, 5.0)
        print(f"等待 {sleep_time:.1f} 秒后继续...")
        time.sleep(sleep_time)

        return product

    except Exception as e:
        print(f"处理商品 {product['goods_id']} 时出错: {e}")
        return product


# 主函数
def main():
    print("启动网易BUFF饰品历史价格爬虫...")

    # 创建浏览器实例
    browser = create_browser()

    if not browser:
        print("无法创建浏览器实例，程序退出")
        return

    try:
        # 简化登录
        print("=" * 50)
        print("正在尝试登录...")
        login_success = simple_login(browser)

        if not login_success:
            print("尝试直接访问市场页面...")
            browser.get(MARKET_URL)

        print("=" * 50)
        print("开始获取商品列表...")

        # 获取商品列表
        products = get_product_list(browser)

        if not products:
            print("没有找到商品，程序退出")
            return

        print(f"开始处理 {len(products)} 个商品的历史价格...")

        # 处理每个商品的历史价格
        processed_products = []

        # 使用tqdm显示进度条
        for i, product in enumerate(tqdm(products, desc="处理商品", file=sys.stdout)):
            processed = process_product(browser, product)
            processed_products.append(processed)

            # 每处理1个商品保存一次数据
            save_to_json(processed_products, 'data/temp_products.json')
            print(f"已保存临时数据: data/temp_products.json")

            # 每处理5个商品保存一次完整数据
            if (i + 1) % 5 == 0:
                save_to_excel(processed_products, 'data/temp_products.xlsx')
                save_history_to_csv(processed_products, 'data/temp_history.csv')

        # 保存最终结果
        save_to_json(processed_products, 'data/buff_products_full.json')
        save_to_excel(processed_products, 'data/buff_products_summary.xlsx')
        save_history_to_csv(processed_products, 'data/buff_price_history.csv')

        print(f"爬取完成！共处理 {len(processed_products)} 个商品")

    except Exception as e:
        print(f"程序运行出错: {e}")

    finally:
        # 关闭浏览器
        if browser:
            print("正在关闭浏览器...")
            browser.quit()
            print("浏览器已关闭")


# 保存函数
def save_to_json(data, filename):
    try:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"数据已保存到 {filename}")
        return True
    except Exception as e:
        print(f"保存JSON文件时出错: {e}")
        return False


def save_to_excel(data, filename):
    try:
        # 创建简化数据结构
        simplified_data = []
        for item in data:
            simplified = {
                'goods_id': item['goods_id'],
                'title': item['title'],
                'history_min': item.get('history_min', ''),
                'history_max': item.get('history_max', ''),
                'history_avg': item.get('history_avg', ''),
                'history_points': item.get('history_points', 0),
                'url': item['url'],
                'history_url': item['history_url'],
                'timestamp': item.get('timestamp', '')
            }
            simplified_data.append(simplified)

        df = pd.DataFrame(simplified_data)
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        df.to_excel(filename, index=False)
        print(f"数据已保存到 {filename}")
        return True
    except Exception as e:
        print(f"保存Excel文件时出错: {e}")
        return False


def save_history_to_csv(data, filename):
    try:
        all_history = []
        seen_records = set()  # 用于记录已处理的记录

        for product in data:
            if not product.get('price_history'):
                continue

            for history in product['price_history']:
                # 创建唯一标识符 (商品ID + 日期 + 价格)
                record_id = f"{product['goods_id']}-{history['date']}-{history['price']}"

                # 检查是否已处理过
                if record_id in seen_records:
                    continue

                seen_records.add(record_id)

                history_record = {
                    'goods_id': product['goods_id'],
                    'title': product['title'],
                    'price': history['price'],
                    'date': history['date'],
                    'timestamp': product.get('timestamp', '')
                }
                all_history.append(history_record)

        if not all_history:
            print("没有历史价格数据可保存")
            return False

        df = pd.DataFrame(all_history)
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        df.to_csv(filename, index=False)
        print(f"历史价格数据已保存到 {filename}")
        return True
    except Exception as e:
        print(f"保存CSV文件时出错: {e}")
        return False


if __name__ == "__main__":
    # 确保data目录存在
    os.makedirs('data', exist_ok=True)

    # 运行主函数
    main()