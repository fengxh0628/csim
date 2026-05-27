import requests
import pandas as pd
import time
import sys

# 1. 填入你已经确定的前 100 个币种 Symbol 列表（这里只列举部分作为示例）
with open(sys.argv[1], 'r') as f:
    my_universe = [line.strip().replace('USDT', '') for line in f]

print(f"开始为指定的 {len(my_universe)} 个币种匹配行业标签...")

# 2. 第一步：获取 CoinGecko 的币种全量映射表（因为它的查询需要用到官方币种 ID）
# 这个接口不需要 API Key 即可访问
coins_list_url = "https://api.coingecko.com/api/v3/coins/list"
all_coins = requests.get(coins_list_url).json()

# 将全量表转换为字典，方便通过 Symbol 快速查找 CG_ID
# 例如：{'btc': 'bitcoin', 'sol': 'solana'}
symbol_to_id = {coin['symbol'].lower(): coin['id'] for coin in all_coins}

# 3. 第二步：循环查询每个币种的行业分类
result_list = []

for symbol in my_universe:
    symbol_lower = symbol.lower()
    cg_id = symbol_to_id.get(symbol_lower)
    
    if not cg_id:
        print(f"⚠️ 未找到币种 {symbol} 的 CoinGecko ID，跳过。")
        continue
        
    print(f"正在查询 {symbol} ({cg_id}) 的行业标签...")
    
    # 精准查询单个币种的详情接口
    coin_detail_url = f"https://api.coingecko.com/api/v3/coins/{cg_id}"
    params = {
        "localization": "false",
        "tickers": "false",
        "market_data": "false",
        "community_data": "false",
        "developer_data": "false",
        "sparkline": "false"
    }
    
    try:
        response = requests.get(coin_detail_url, params=params, timeout=10).json()
        # 提取行业分类列表
        categories = response.get('categories', [])
        
        # 过滤掉一些无意义的机构持仓标签（如 FTX Holdings、Alameda Research Portfolio 等）
        clean_categories = [c for c in categories if "Portfolio" not in c and "Holdings" not in c]
        
        # 在截面策略中，我们需要一个币对应一个最核心的“主行业”
        primary_sector = clean_categories[0] if clean_categories else "Other"
        all_sectors_str = ", ".join(clean_categories)
        
        result_list.append({
            "Ticker": symbol,
            "Binance_Symbol": f"{symbol}USDT", # 自动对齐币安永续合约代码
            "Primary_Sector": primary_sector, # 主行业（用于中性化分组）
            "All_Sectors": all_sectors_str     # 备用全量标签
        })
        
    except Exception as e:
        print(f"❌ 查询 {symbol} 失败: {e}")
        
    # 💡 极其重要：免费接口有频次限制，每次查询后强制暂停 2 秒，防止被封 IP
    time.sleep(2)

# 4. 第三步：生成回测用的静态数据库
df_universe_sectors = pd.DataFrame(result_list)
df_universe_sectors.to_csv("my_universe_sectors.csv", index=False)

print("\n🎉 行业标签匹配完成！数据已保存在本地 'my_universe_sectors.csv'")
print(df_universe_sectors.head(10))