import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# ── Load ──────────────────────────────────────────────────────────────
df       = pd.read_csv('output/df_stations_clean.csv')
stations = pd.read_csv('output/station_clusters.csv')[['start_station_name', 'cluster', 'cluster_label']]
df       = df.merge(stations, on='start_station_name', how='left')
df       = df.dropna(subset=['cluster'])
df['date'] = pd.to_datetime(df['date'])

print(f"Loaded: {df.shape}")
print(f"Cluster distribution:\n{df['cluster_label'].value_counts()}")

# ── Aggregate: hourly trips per cluster ───────────────────────────────
hourly = (
    df.groupby(['date', 'hour', 'cluster', 'cluster_label'])
    .size()
    .reset_index(name='trips')
)

# ── Add time features ─────────────────────────────────────────────────
hourly['day_of_week'] = pd.to_datetime(hourly['date']).dt.dayofweek
hourly['month']       = pd.to_datetime(hourly['date']).dt.month
hourly['is_weekend']  = hourly['day_of_week'].isin([5, 6]).astype(int)
hourly['is_morning']  = hourly['hour'].between(7, 9).astype(int)
hourly['is_evening']  = hourly['hour'].between(17, 19).astype(int)

# ── Focus: High-Demand Evening Hub (Cluster 4) ────────────────────────
# This is the primary rebalancing risk zone
target_cluster = 4
hd = hourly[hourly['cluster'] == target_cluster].copy().reset_index(drop=True)
print(f"\nHigh-Demand Evening Hub rows: {len(hd)}")

# ── Feature matrix ────────────────────────────────────────────────────
FEATURES = ['hour', 'day_of_week', 'month', 'is_weekend', 'is_morning', 'is_evening']
X = hd[FEATURES]
y = hd['trips']

# Chronological split — last 20% as test (not random, respects time)
split = int(len(hd) * 0.8)
X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]

# ── Model 1: Linear Regression (baseline) ────────────────────────────
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

mae_lr  = mean_absolute_error(y_test, y_pred_lr)
rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))

# ── Model 2: Random Forest (improved) ────────────────────────────────
rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

mae_rf  = mean_absolute_error(y_test, y_pred_rf)
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))

print("\n--- Model Comparison (Cluster 4: High-Demand Evening Hub) ---")
print(f"  Linear Regression  → MAE: {mae_lr:.2f}  RMSE: {rmse_lr:.2f}")
print(f"  Random Forest      → MAE: {mae_rf:.2f}  RMSE: {rmse_rf:.2f}")

# ── Feature importance ────────────────────────────────────────────────
importance = pd.Series(rf.feature_importances_, index=FEATURES).sort_values(ascending=False)
print(f"\n--- Feature Importances ---")
print(importance.round(4))

# ── Chart 1: Actual vs Predicted (7 days of test data) ───────────────
test_plot = hd.iloc[split:split+168].copy()   # 168 hours = 7 days
test_plot['predicted'] = y_pred_rf[:168]

fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(range(168), test_plot['trips'].values,     label='Actual',    color='steelblue',  linewidth=1.5)
ax.plot(range(168), test_plot['predicted'].values, label='Predicted', color='darkorange', linewidth=1.5, linestyle='--')
ax.set_title('High-Demand Evening Hub — Actual vs Predicted Hourly Trips (7-Day Test Window)',
             fontsize=12, fontweight='bold')
ax.set_xlabel('Hour Index (7-Day Window)')
ax.set_ylabel('Number of Trips')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('charts/forecast_actual_vs_predicted.png', dpi=150, bbox_inches='tight')
plt.show()
print("✓ Saved charts/forecast_actual_vs_predicted.png")

# ── Chart 2: Average hourly demand profile per cluster ───────────────
fig, ax = plt.subplots(figsize=(12, 5))
cluster_colors = {
    'Low-Activity Residential': '#adb5bd',
    'Morning Commuter Feeder':  '#2a9d8f',
    'Leisure & Tourist Zone':   '#e9c46a',
    'Weekday Commuter Hub':     '#457b9d',
    'High-Demand Evening Hub':  '#e63946',
}

for label, grp in hourly.groupby('cluster_label'):
    profile = grp.groupby('hour')['trips'].mean()
    ax.plot(profile.index, profile.values,
            label=label, color=cluster_colors.get(label, 'grey'),
            linewidth=2.2, marker='o', markersize=3)

ax.axvspan(7, 9,   alpha=0.08, color='green', label='_Morning peak')
ax.axvspan(17, 19, alpha=0.08, color='red',   label='_Evening peak')
ax.set_title('Average Hourly Demand Profile by Cluster Zone', fontsize=12, fontweight='bold')
ax.set_xlabel('Hour of Day')
ax.set_ylabel('Avg Trips per Hour')
ax.set_xticks(range(0, 24))
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('charts/hourly_demand_profile.png', dpi=150, bbox_inches='tight')
plt.show()
print("✓ Saved charts/hourly_demand_profile.png")

# ── Save model results ────────────────────────────────────────────────
results = pd.DataFrame({
    'metric': ['MAE', 'RMSE'],
    'linear_regression': [mae_lr, rmse_lr],
    'random_forest':     [mae_rf, rmse_rf]
})
results.to_csv('output/forecast_model_results.csv', index=False)

hd['predicted_rf'] = rf.predict(X)
hd.to_csv('output/cluster4_hourly_forecast.csv', index=False)
print("✓ Saved output/forecast_model_results.csv")
print("✓ Saved output/cluster4_hourly_forecast.csv")