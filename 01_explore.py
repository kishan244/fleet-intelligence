import pandas as pd
import os

# ── Load ──────────────────────────────────────────────────────────────
files = [f for f in os.listdir('data') if f.endswith('.csv')]
df_list = [pd.read_csv(f'data/{f}') for f in files]
df = pd.concat(df_list, ignore_index=True)

print(f"Raw shape: {df.shape}")

# ── Clean ─────────────────────────────────────────────────────────────
# 1. Drop rows with missing end coordinates (only 2408)
df = df.dropna(subset=['end_lat', 'end_lng'])

# 2. Parse datetimes
df['started_at'] = pd.to_datetime(df['started_at'])
df['ended_at']   = pd.to_datetime(df['ended_at'])

# 3. Feature engineering
df['hour']             = df['started_at'].dt.hour
df['day_of_week']      = df['started_at'].dt.dayofweek   # 0=Mon, 6=Sun
df['month']            = df['started_at'].dt.month
df['date']             = df['started_at'].dt.date
df['is_weekend']       = df['day_of_week'].isin([5, 6]).astype(int)
df['trip_duration_min'] = (df['ended_at'] - df['started_at']).dt.total_seconds() / 60

# 4. Drop bad trip durations (negative or > 24 hours)
df = df[(df['trip_duration_min'] > 0) & (df['trip_duration_min'] < 1440)]

# 5. Two versions of the dataset
df_all      = df.copy()                            # all rows (for clustering)
df_stations = df.dropna(subset=['start_station_name'])  # station rows only (for forecasting)

print(f"Cleaned (all):       {df_all.shape}")
print(f"Cleaned (stations):  {df_stations.shape}")

# ── Quick stats ───────────────────────────────────────────────────────
print("\n--- Trip duration stats ---")
print(df_all['trip_duration_min'].describe().round(2))

print("\n--- Rides by hour (top 5) ---")
print(df_all.groupby('hour').size().sort_values(ascending=False).head())

print("\n--- Rider type split ---")
print(df_all['member_casual'].value_counts())

print("\n--- Bike type split ---")
print(df_all['rideable_type'].value_counts())

# ── Save cleaned files ────────────────────────────────────────────────
df_all.to_csv('output/df_all_clean.csv', index=False)
df_stations.to_csv('output/df_stations_clean.csv', index=False)
print("\n✓ Saved to output/")