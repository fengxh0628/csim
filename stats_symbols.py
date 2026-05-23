import os
import pandas as pd

dump_dir = 'dumps/t_top5'
files = sorted(os.listdir(dump_dir))

long_counts = {}
short_counts = {}

for f in files:
    if not f.endswith('.csv'):
        continue
    ts_str = f.replace('.csv', '')
    if ts_str < '20230101000000':
        continue
    df = pd.read_csv(os.path.join(dump_dir, f), index_col=0)
    df = df.dropna()
    if df.empty:
        continue

    # Sort by alpha
    df_sorted = df.sort_values('alpha')
    
    # Top 5 long
    for sym in df_sorted.tail(5).index:
        long_counts[sym] = long_counts.get(sym, 0) + 1
        
    # Top 5 short
    for sym in df_sorted.head(5).index:
        short_counts[sym] = short_counts.get(sym, 0) + 1

long_df = pd.DataFrame(list(long_counts.items()), columns=['Symbol', 'Days']).sort_values('Days', ascending=False)
short_df = pd.DataFrame(list(short_counts.items()), columns=['Symbol', 'Days']).sort_values('Days', ascending=False)

print("=== Long (Top 1) Frequency ===")
print(long_df.head(20).to_string(index=False))

print("\n=== Short (Bottom 1) Frequency ===")
print(short_df.head(20).to_string(index=False))
