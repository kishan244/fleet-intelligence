import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

# ── Load ──────────────────────────────────────────────────────────────
df = pd.read_csv('output/df_stations_clean.csv')
print(f"Loaded: {df.shape}")

# ── Build station-level feature matrix ───────────────────────────────
station = df.groupby('start_station_name').agg(
    total_trips     = ('ride_id',          'count'),
    avg_lat         = ('start_lat',        'mean'),
    avg_lng         = ('start_lng',        'mean'),
    avg_duration    = ('trip_duration_min','mean'),
    weekend_ratio   = ('is_weekend',       'mean'),
    member_ratio    = ('member_casual',    lambda x: (x == 'member').mean()),
).reset_index()

# Peak hour shares
df['morning_peak'] = df['hour'].between(7, 9).astype(int)
df['evening_peak'] = df['hour'].between(17, 19).astype(int)

peak = df.groupby('start_station_name').agg(
    morning_share = ('morning_peak', 'mean'),
    evening_share = ('evening_peak', 'mean'),
).reset_index()

station = station.merge(peak, on='start_station_name')

# Keep only stations with enough trips for reliable clustering
station = station[station['total_trips'] >= 50].reset_index(drop=True)
print(f"Stations after min-trip filter: {len(station)}")

# ── Normalize features ────────────────────────────────────────────────
features = ['total_trips', 'avg_duration', 'weekend_ratio',
            'member_ratio', 'morning_share', 'evening_share']

X = station[features].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ── Elbow + Silhouette to pick k ──────────────────────────────────────
inertia, sil_scores = [], []
K_range = range(2, 9)

for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    inertia.append(km.inertia_)
    sil_scores.append(silhouette_score(X_scaled, labels))

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(K_range, inertia, 'o-', color='steelblue')
axes[0].set_title('Elbow Method')
axes[0].set_xlabel('Number of Clusters k')
axes[0].set_ylabel('Inertia')
axes[0].grid(True, alpha=0.3)

axes[1].plot(K_range, sil_scores, 'o-', color='darkorange')
axes[1].set_title('Silhouette Score')
axes[1].set_xlabel('Number of Clusters k')
axes[1].set_ylabel('Score')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('charts/elbow_silhouette.png', dpi=150, bbox_inches='tight')
plt.show()
print("✓ Saved charts/elbow_silhouette.png")

# Print scores to help pick k
for k, s in zip(K_range, sil_scores):
    print(f"  k={k}  silhouette={s:.4f}")

# ── Final K-Means with k=5 ────────────────────────────────────────────
K_FINAL = 5
km_final = KMeans(n_clusters=K_FINAL, random_state=42, n_init=10)
station['cluster'] = km_final.fit_predict(X_scaled)

# ── Cluster profile table ─────────────────────────────────────────────
profile = station.groupby('cluster').agg(
    n_stations    = ('start_station_name', 'count'),
    total_trips   = ('total_trips',        'mean'),
    morning_share = ('morning_share',      'mean'),
    evening_share = ('evening_share',      'mean'),
    weekend_ratio = ('weekend_ratio',      'mean'),
    member_ratio  = ('member_ratio',       'mean'),
    avg_duration  = ('avg_duration',       'mean'),
).round(3)

print("\n--- Cluster Profiles ---")
print(profile.to_string())

# ── Assign business labels (we will adjust after seeing profile) ───────
labels = {
    0: 'Low-Activity Residential',
    1: 'Morning Commuter Feeder',
    2: 'Leisure & Tourist Zone',
    3: 'Weekday Commuter Hub',
    4: 'High-Demand Evening Hub',
}
station['cluster_label'] = station['cluster'].map(labels)

# ── Geographic scatter plot ───────────────────────────────────────────
COLORS = ['#e63946', '#2a9d8f', '#e9c46a', '#457b9d', '#9b5de5']

fig, ax = plt.subplots(figsize=(10, 12))

for c in range(K_FINAL):
    subset = station[station['cluster'] == c]
    ax.scatter(
        subset['avg_lng'], subset['avg_lat'],
        c=COLORS[c], label=labels[c],
        s=subset['total_trips'] / subset['total_trips'].max() * 120 + 10,
        alpha=0.7, edgecolors='white', linewidths=0.3
    )

ax.set_title('Chicago Divvy Stations — Demand Zone Clusters (k=5)\n'
             'Dot size = relative trip volume', fontsize=13, fontweight='bold')
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
ax.legend(title='Cluster', fontsize=9)
ax.grid(True, alpha=0.2)
plt.tight_layout()
plt.savefig('charts/cluster_map.png', dpi=150, bbox_inches='tight')
plt.show()
print("✓ Saved charts/cluster_map.png")

# ── Save for next steps ───────────────────────────────────────────────
station.to_csv('output/station_clusters.csv', index=False)
print("✓ Saved output/station_clusters.csv")