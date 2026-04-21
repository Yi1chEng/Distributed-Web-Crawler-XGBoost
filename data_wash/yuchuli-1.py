# 强制设置Python输出编码为UTF-8，彻底解决Windows终端乱码
import io
import sys
import urllib.request
sys.stdout = io.TextIOWrapper(sys.stdout.buffer,encoding='utf8')

# 核心库导入
import json  # JSON数据处理库
import redis  # Redis数据库客户端
import csv  # CSV文件处理库
import re  # 正则表达式库
from tqdm import tqdm  # 进度条库

# ===================== Redis基础配置 =====================
REDIS_HOST = "127.0.0.1"  # Redis服务器地址
REDIS_PORT = 6379  # Redis服务器端口
REDIS_DB = 0  # Redis数据库编号
REDIS_PASSWORD = ""  # 有密码填这里，无则留空
REDIS_LIST_KEY = "sfw:items"  # Redis中存储数据的列表名

# ===================== 输出配置 =====================
OUTPUT_CSV = "cleaned_house_data.csv"  # 清洗后数据保存的CSV文件名
WRITE_BACK_REDIS = True  # 是否将清洗后的数据写回Redis
CLEAN_REDIS_PREFIX = "cleaned_house:"  # 写回Redis时的键前缀

# ===================== 核心函数 =====================
def connect_redis():
    """连接Redis数据库
    
    Returns:
        redis.Redis: Redis连接对象
    
    Raises:
        SystemExit: 如果连接失败则退出程序
    """
    try:
        # 创建Redis连接对象
        r = redis.Redis(
            host=REDIS_HOST,
            port=REDIS_PORT,
            db=REDIS_DB,
            password=REDIS_PASSWORD,
            decode_responses=False  # 不自动解码，保持字节类型
        )
        # 测试连接是否成功
        r.ping()
        print("Redis connection success")
        return r
    except Exception as e:
        print("Redis connection fail: " + str(e))
        sys.exit(1)

def process_single_data(raw_json):
    """清洗单条数据
    
    Args:
        raw_json (str): 原始JSON字符串数据
        
    Returns:
        dict or None: 清洗后的数据字典，如果清洗失败则返回None
    """
    if not raw_json:
        return None
    try:
        # 解析JSON字符串
        data = json.loads(raw_json.strip())
        result = {}
        
        # 核心字段清洗，district行政区
        fields = ["name", "rooms", "area", "address", "district", "sale", "price", "origin_url", "province", "city"]
        for field in fields:
            val = data.get(field, "")
            if isinstance(val, str):
                # 去除字符串两端的空白字符
                result[field] = val.strip()
            elif isinstance(val, list):
                # 清洗列表中的字符串元素
                result[field] = [v.strip() for v in val if isinstance(v, str)]
            else:
                # 保留其他类型的值，None转换为空字符串
                result[field] = val if val is not None else ""
        
        # 字段标准化
        if result["area"]:
            # 统一面积单位和格式
            result["area"] = result["area"].replace("—", "").replace("~", "-").replace("平米", "㎡")
        if result["price"]:
            # 提取价格数字
            price_num = re.findall(r"\d+", result["price"])
            result["price_num"] = int(price_num[0]) if price_num else None
        
        # 过滤空名称数据
        if not result["name"]:
            return None
        return result
    except Exception as e:
        print("Data process fail: " + str(e))
        return None

def read_redis_data(redis_conn):
    """从Redis读取列表数据
    
    Args:
        redis_conn (redis.Redis): Redis连接对象
        
    Returns:
        list: 转换为UTF-8字符串的数据列表
    """
    # 获取列表长度
    list_length = redis_conn.llen(REDIS_LIST_KEY)
    print("Redis list " + REDIS_LIST_KEY + " length: " + str(list_length))
    
    # 批量读取所有元素
    raw_data = redis_conn.lrange(REDIS_LIST_KEY, 0, -1)
    print("Read data count: " + str(len(raw_data)))
    
    # 转换为UTF-8字符串
    str_data = [item.decode("utf-8") for item in raw_data if item is not None]
    return str_data

def save_to_csv_file(cleaned_data):
    """保存清洗后的数据到CSV文件
    
    Args:
        cleaned_data (list): 清洗后的数据列表
    """
    if not cleaned_data:
        print("No valid data to save")
        return
    # 获取所有字段名（从第一条数据中提取）
    headers = list(cleaned_data[0].keys())
    # 使用utf-8-sig编码，确保Excel打开无乱码
    with open(OUTPUT_CSV, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()  # 写入表头
        writer.writerows(cleaned_data)  # 写入数据
    print("CSV saved to: " + OUTPUT_CSV)

# ===================== 主程序 =====================
if __name__ == "__main__":
    # 1. 连接Redis
    print("Step 1: Connecting to Redis...")
    redis_conn = connect_redis()
    
    # 2. 读取数据
    print("\nStep 2: Start reading data from Redis...")
    raw_data_list = read_redis_data(redis_conn)
    if not raw_data_list:
        print("No data read from Redis")
        sys.exit(0)
    print("Total raw data: " + str(len(raw_data_list)))
    
    # 3. 批量清洗数据
    print("\nStep 3: Start processing data...")
    cleaned_list = []
    # 使用tqdm显示处理进度
    for index, raw_str in enumerate(tqdm(raw_data_list, desc="Processing")):
        cleaned_item = process_single_data(raw_str)
        if cleaned_item:
            cleaned_list.append(cleaned_item)
            
            # 可选：写回Redis
            if WRITE_BACK_REDIS:
                try:
                    # 将清洗后的数据转换为JSON字符串
                    json_str = json.dumps(cleaned_item, ensure_ascii=False)
                    # 写回Redis，使用索引作为键的一部分
                    redis_conn.set(CLEAN_REDIS_PREFIX + str(index+1), json_str.encode("utf-8"))
                except Exception as e:
                    print("Write to Redis fail: " + str(e))
    
    # 4. 保存结果到CSV文件
    print("\nStep 4: Saving results to CSV...")
    save_to_csv_file(cleaned_list)
    
    # 5. 输出统计信息
    print("\n===== Process Result =====")
    print("Raw data count: " + str(len(raw_data_list)))
    print("Cleaned data count: " + str(len(cleaned_list)))
    print("Success rate: " + str(round(len(cleaned_list)/len(raw_data_list)*100, 2)) + "%")
    print("Process completed!")