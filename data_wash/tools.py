# 数据清洗和处理工具函数
import pandas as pd
import numpy as np
import re


def clean_price(price_str):
    """
    清洗价格字符串：
    - 提取数字（处理带单位/符号的情况，如"500万"、"¥800000"）
    - 处理缺失/非数字值：返回NaN
    
    Args:
        price_str: 价格字符串
        
    Returns:
        float: 清洗后的价格数值
    """
    if pd.isna(price_str):
        return np.nan
    # 提取所有数字（支持小数）
    nums = re.findall(r'\d+\.?\d*', str(price_str))
    if not nums:
        return np.nan
    # 取第一个有效数字（避免多数字干扰）
    price = float(nums[0])
    # 处理"万"单位（如果有）：比如"500万"→5000000
    if '万' in str(price_str):
        price *= 10000
    return price


def clean_area(area_str):
    """
    清洗面积字符串：
    - 处理区间值（如143-248㎡）：取平均值
    - 处理单值（如100㎡）：提取数字
    - 处理缺失/异常值：返回NaN
    
    Args:
        area_str: 面积字符串
        
    Returns:
        float: 清洗后的面积数值
    """
    if pd.isna(area_str):
        return np.nan
    # 提取所有数字
    nums = re.findall(r'\d+\.?\d*', str(area_str))
    if not nums:
        return np.nan
    # 区间值（如143-248）取平均，单值直接转浮点数
    if len(nums) >= 2:
        return (float(nums[0]) + float(nums[1])) / 2
    else:
        return float(nums[0])


def parse_rooms(room_str):
    """
    提取户型数字，如「四居」→4
    
    Args:
        room_str: 户型字符串
        
    Returns:
        int: 户型数量
    """
    num = re.findall(r'\d+', str(room_str))
    return int(num[0]) if num else 0


def remove_outliers(df, col):
    """
    IQR法剔除异常值，增加数值类型校验
    
    Args:
        df: 数据框
        col: 要处理的列名
        
    Returns:
        DataFrame: 剔除异常值后的数据框
    """
    # 确保列是数值类型
    df[col] = pd.to_numeric(df[col], errors='coerce')
    Q1 = df[col].quantile(0.25)  # 第一四分位数
    Q3 = df[col].quantile(0.75)  # 第三四分位数
    IQR = Q3 - Q1  # 四分位距
    lower_bound = Q1 - 1.5 * IQR  # 下界
    upper_bound = Q3 + 1.5 * IQR  # 上界
    return df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]