from binance.client import Client
from binance.exceptions import BinanceAPIException
from openai import OpenAI
import pandas as pd
import numpy as np
import random
import datetime
import json
import time
import logging
import schedule
import smtplib
import ssl
import requests
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.header import Header
import pytz
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s: %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('trading_analysis.log', encoding='utf-8')
    ]
)

# 配置重试策略
retry_strategy = Retry(
    total=5,
    backoff_factor=1,
    status_forcelist=[500, 502, 503, 504, 429],
    allowed_methods=["HEAD", "GET", "POST", "PUT", "DELETE", "OPTIONS", "TRACE"]
)

# 创建 HTTP 适配器
adapter = HTTPAdapter(
    max_retries=retry_strategy,
    pool_connections=100,
    pool_maxsize=100
)

# 创建会话
session = requests.Session()
session.mount("http://", adapter)
session.mount("https://", adapter)

# 从环境变量获取 API 配置
BINANCE_API_KEY = os.getenv('BINANCE_API_KEY')
BINANCE_API_SECRET = os.getenv('BINANCE_API_SECRET')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
ASSISTANT_ID = os.getenv('ASSISTANT_ID')

# 从环境变量获取邮件配置
EMAIL_CONFIG = {
    'smtp_server': os.getenv('EMAIL_SMTP_SERVER'),
    'smtp_port': int(os.getenv('EMAIL_SMTP_PORT')),
    'sender_email': os.getenv('EMAIL_SENDER'),
    'sender_password': os.getenv('EMAIL_PASSWORD'),
    'receiver_email': os.getenv('EMAIL_RECEIVER')
}

# 初始化客户端
binance_client = Client(
    BINANCE_API_KEY, 
    BINANCE_API_SECRET,
    requests_params={
        'timeout': 30,
        'verify': True
    },
    tld='com'
)
openai_client = OpenAI(api_key=OPENAI_API_KEY)

# 设置时区
TIMEZONE = pytz.timezone('Asia/Shanghai')

def with_retry(func, max_retries=3, delay=5):
    """带重试机制的函数装饰器"""
    def wrapper(*args, **kwargs):
        for attempt in range(max_retries):
            try:
                return func(*args, **kwargs)
            except (BinanceAPIException, requests.exceptions.RequestException) as e:
                if attempt == max_retries - 1:
                    raise
                logging.warning(f"尝试 {attempt + 1}/{max_retries} 失败: {str(e)}")
                time.sleep(delay * (attempt + 1))
        return None
    return wrapper

def get_current_time():
    """获取当前UTC+8时间"""
    return datetime.datetime.now(TIMEZONE)

@with_retry
def get_top_300_symbols():
    """获取成交量前300的合约"""
    exchange_info = binance_client.futures_exchange_info()
    symbols = {symbol_info['symbol'] for symbol_info in exchange_info['symbols']
               if symbol_info['contractType'] == 'PERPETUAL' and symbol_info['quoteAsset'] == 'USDT'}

    tickers = binance_client.futures_ticker()
    df_tickers = pd.DataFrame(tickers)
    df_tickers = df_tickers[df_tickers['symbol'].isin(symbols)]
    df_tickers['volume_usdt'] = pd.to_numeric(df_tickers['volume']) * pd.to_numeric(df_tickers['lastPrice'])
    top_300_symbols = df_tickers.nlargest(300, 'volume_usdt')['symbol'].tolist()
    return top_300_symbols

@with_retry
def get_klines(symbol, interval, limit, end_time=None):
    """获取K线数据的封装函数"""
    return binance_client.futures_klines(
        symbol=symbol,
        interval=interval,
        limit=limit,
        endTime=end_time
    )

def calculate_atr(df, period=14):
    """计算ATR指标"""
    high = df['high'].astype(float)
    low = df['low'].astype(float)
    close = df['close'].astype(float)
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean().iloc[-1]
    return atr

def calculate_indicators(symbols):
    """计算交易对的指标"""
    data_list = []
    now = get_current_time()
    now_utc = now.astimezone(pytz.UTC)
    end_time = int((now_utc.replace(minute=0, second=0, microsecond=0) - pd.Timedelta(hours=1)).timestamp() * 1000)
    end_time_4h = int((now_utc.replace(hour=(now_utc.hour // 4) * 4, minute=0, second=0, microsecond=0) - pd.Timedelta(hours=4)).timestamp() * 1000)

    for symbol in symbols:
        try:
            klines_1h = get_klines(symbol, Client.KLINE_INTERVAL_1HOUR, 24, end_time)
            if not klines_1h or len(klines_1h) < 24:
                continue
                
            df_1h = pd.DataFrame(klines_1h, columns=[
                'open_time', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ])
            df_1h[['open', 'high', 'low', 'close', 'volume']] = df_1h[['open', 'high', 'low', 'close', 'volume']].astype(float)

            df_1h['price_change'] = df_1h['close'].pct_change().abs()
            if df_1h['price_change'].gt(0.1).any():
                continue

            volatility_24h = df_1h['close'].pct_change().std() * np.sqrt(24)

            klines_4h = get_klines(symbol, Client.KLINE_INTERVAL_4HOUR, 14, end_time_4h)
            if not klines_4h or len(klines_4h) < 14:
                continue
                
            df_4h = pd.DataFrame(klines_4h, columns=[
                'open_time', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ])
            df_4h[['open', 'high', 'low', 'close', 'volume']] = df_4h[['open', 'high', 'low', 'close', 'volume']].astype(float)

            atr_4h = calculate_atr(df_4h)
            current_price = df_1h['close'].iloc[-1]
            atr_price_ratio = atr_4h / current_price if current_price != 0 else 0

            price_change_amplitude = abs(df_1h['close'].iloc[-1] - df_1h['close'].iloc[-2]) / df_1h['close'].iloc[-2]

            data_list.append({
                'symbol': symbol,
                'volatility_24h': volatility_24h,
                'atr_price_ratio': atr_price_ratio,
                'price_change_amplitude': price_change_amplitude
            })
        except Exception as e:
            logging.error(f"处理合约 {symbol} 时出错：{e}")
            continue
    return data_list

def weighted_screening(data_list):
    """加权筛选"""
    if not data_list:
        return []
        
    df = pd.DataFrame(data_list)
    
    for col in ['volatility_24h', 'atr_price_ratio', 'price_change_amplitude']:
        if col not in df.columns:
            logging.error(f"列 {col} 不存在于数据中")
            return []
            
        max_val = df[col].max()
        min_val = df[col].min()
        if max_val != min_val:
            df[col + '_norm'] = (df[col] - min_val) / (max_val - min_val)
        else:
            df[col + '_norm'] = 0.0

    df['score'] = (
        df['volatility_24h_norm'] * 0.3 + 
        df['atr_price_ratio_norm'] * 0.3 + 
        df['price_change_amplitude_norm'] * 0.4
    )
    
    df_sorted = df.sort_values(by='score', ascending=False)
    df_filtered_sorted = df_sorted[~df_sorted['symbol'].isin(['BTCUSDT', 'ETHUSDT'])].head(20)
    
    return df_filtered_sorted['symbol'].tolist()

@with_retry
def get_further_data(symbol):
    """获取详细数据"""
    data = {}
    try:
        timeframes = {
            '5m': Client.KLINE_INTERVAL_5MINUTE,
            '15m': Client.KLINE_INTERVAL_15MINUTE,
            '1h': Client.KLINE_INTERVAL_1HOUR,
            '4h': Client.KLINE_INTERVAL_4HOUR
        }

        for tf_name, tf_interval in timeframes.items():
            klines = get_klines(symbol, tf_interval, 40)
            if not klines:
                continue
                
            df = pd.DataFrame(klines, columns=[
                'open_time', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ])
            df = df[['open_time', 'open', 'high', 'low', 'close', 'volume']]
            df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
            
            df['open_time'] = pd.to_datetime(df['open_time'].astype(float), unit='ms')
            df['open_time'] = df['open_time'].dt.tz_localize(pytz.UTC).dt.tz_convert(TIMEZONE)
            df['open_time'] = df['open_time'].dt.strftime('%Y-%m-%d %H:%M:%S')
            
            data[tf_name] = df.to_dict(orient='records')

        depth = binance_client.futures_order_book(symbol=symbol, limit=50)
        data['order_book'] = depth

        funding_rate = binance_client.futures_mark_price(symbol=symbol)
        data['funding_rate'] = {
            'symbol': funding_rate['symbol'],
            'fundingRate': float(funding_rate['lastFundingRate'])
        }

        open_interest = binance_client.futures_open_interest(symbol=symbol)
        data['open_interest'] = {
            'symbol': open_interest['symbol'],
            'openInterest': float(open_interest['openInterest'])
        }

        return data
    except Exception as e:
        logging.error(f"获取 {symbol} 的数据时出错：{e}")
        return None

def process_openai_analysis(json_data):
    """处理GPT分析"""
    try:
        thread = openai_client.beta.threads.create()
        
        message = openai_client.beta.threads.messages.create(
            thread_id=thread.id,
            role="user",
            content=json.dumps(json_data, ensure_ascii=False)
        )

        run = openai_client.beta.threads.runs.create(
            thread_id=thread.id,
            assistant_id=ASSISTANT_ID
        )

        max_wait_time = 240
        start_time = time.time()
        
        while True:
            try:
                run = openai_client.beta.threads.runs.retrieve(
                    thread_id=thread.id, 
                    run_id=run.id
                )
                if run.status == "completed":
                    messages = openai_client.beta.threads.messages.list(thread_id=thread.id)
                    if messages.data:
                        result = messages.data[0].content[0].text.value
                        return result
                elif run.status in ["failed", "cancelled", "expired"]:
                    logging.error(f"GPT分析失败，状态：{run.status}")
                    return None
                    
                if time.time() - start_time > max_wait_time:
                    logging.error("GPT分析超时")
                    return None
                    
                time.sleep(5)
            except Exception as e:
                logging.error(f"检查GPT分析状态时出错：{e}")
                return None

    except Exception as e:
        logging.error(f"GPT分析过程出错：{e}")
        return None

def send_email(subject, content):
    """
