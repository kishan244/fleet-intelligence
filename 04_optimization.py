import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linprog
import warnings
warnings.filterwarnings('ignore')

# ── Load cluster metadata ─────────────────────────────────────────────
stations = pd.read_csv('output/station_clusters.csv')

cluster_meta = stations.groupby(['cluster', 'cluster_label']).agg(
    n_stations    = ('start_station_name', 'count'),
    avg_trips     = ('total_trips',        'mean'),
    evening_share = ('evening_share',      'mean'),
    lat           = ('avg_lat',            'mean'),
    lng           = ('avg_lng',            'mean'),
).reset_index().sort_values('cluster').reset_index(drop=True)

print("=== Cluster Overview ===")
print(cluster_meta[['cluster_label','n_stations','avg_trips','evening_share']].to_string())

# ── Fleet scenario ────────────────────────────────────────────────────
TOTAL_FLEET          = 1000   # bikes in operational area
COST_PER_KM_PER_BIKE = 0.75  # EUR per km per bike (truck dispatch cost)
n = len(cluster_meta)

# Initial deployment: proportional to station count (naive uniform strategy)
cluster_meta['initial_bikes'] = (
    cluster_meta['n_stations'] / cluster_meta['n_stations'].sum() * TOTAL_FLEET
).round().astype(int)

# Evening demand: proportional to trips × evening_share → scaled to fleet
raw_demand = cluster_meta['n_stations'] * cluster_meta['avg_trips'] * cluster_meta['evening_share']
cluster_meta['demand'] = (raw_demand / raw_demand.sum() * TOTAL_FLEET).round().astype(int)

# Imbalance: positive = surplus, negative = deficit
cluster_meta['imbalance'] = cluster_meta['initial_bikes'] - cluster_meta['demand']

print("\n=== Supply / Demand Scenario (Evening Peak) ===")
print(cluster_meta[['cluster_label','initial_bikes','demand','imbalance']].to_string())

# ── Haversine distance matrix ─────────────────────────────────────────
def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = (np.sin(dlat/2)**2 +
         np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon/2)**2)
    return 2 * R * np.arcsin(np.sqrt(a))

lats = cluster_meta['lat'].values
lngs = cluster_meta['lng'].values
dist_matrix = np.array([[haversine(lats[i], lngs[i], lats[j], lngs[j])
                          for j in range(n)] for i in range(n)])

print("\n=== Distance Matrix (km between cluster centroids) ===")
dist_df = pd.DataFrame(dist_matrix.round(2),
                        index=cluster_meta['cluster_label'],
                        columns=cluster_meta['cluster_label'])
print(dist_df.to_string())

# ── LP formulation ────────────────────────────────────────────────────
# Variables : x[i,j] = bikes moved from cluster i → cluster j  (n×n vector)
# Objective : minimize Σ dist[i,j] * cost_per_km * x[i,j]
# Constraints:
#   (1) Σ_j x[i,j] <= surplus[i]   for each supply zone i
#   (2) Σ_i x[i,j] >= deficit[j]   for each demand zone j  → -Σ_i x[i,j] <= -deficit[j]
#   (3) x[i,j] >= 0 ;  x[i,i] = 0

c_vec   = (dist_matrix * COST_PER_KM_PER_BIKE).flatten()
surplus = np.maximum(0,  cluster_meta['imbalance'].values).astype(float)
deficit = np.maximum(0, -cluster_meta['imbalance'].values).astype(float)

A_ub, b_ub = [], []

# Supply constraints
for i in range(n):
    row = np.zeros(n * n)
    for j in range(n):
        if i != j:
            row[i * n + j] = 1.0
    A_ub.append(row)
    b_ub.append(surplus[i])

# Demand constraints
for j in range(n):
    row = np.zeros(n * n)
    for i in range(n):
        if i != j:
            row[i * n + j] = -1.0
    A_ub.append(row)
    b_ub.append(-deficit[j])

bounds = [(0.0, 0.0) if i == j else (0.0, None)
          for i in range(n) for j in range(n)]

result = linprog(np.array(c_vec), A_ub=np.array(A_ub),
                 b_ub=np.array(b_ub), bounds=bounds, method='highs')

print(f"\n=== LP Status: {result.message} ===")

if result.success:
    x_opt  = result.x.reshape((n, n))
    lp_cost = result.fun
    labels  = cluster_meta['cluster_label'].values

    # ── Rebalancing plan table ────────────────────────────────────────
    print(f"\n✓ Optimal rebalancing cost: €{lp_cost:.2f}")
    plan_rows = []
    for i in range(n):
        for j in range(n):
            if x_opt[i, j] > 0.5:
                bikes = int(round(x_opt[i, j]))
                dist  = dist_matrix[i, j]
                cost  = bikes * dist * COST_PER_KM_PER_BIKE
                plan_rows.append({
                    'From Zone':       labels[i],
                    'To Zone':         labels[j],
                    'Bikes to Move':   bikes,
                    'Distance (km)':   round(dist, 2),
                    'Cost (EUR)':      round(cost, 2)
                })

    plan_df = pd.DataFrame(plan_rows)
    print("\n=== Optimal Rebalancing Plan ===")
    print(plan_df.to_string(index=False))

    # ── Naive cost benchmark ──────────────────────────────────────────
    total_bikes_moved = deficit.sum()
    avg_distance      = dist_matrix[dist_matrix > 0].mean()
    naive_cost        = total_bikes_moved * avg_distance * COST_PER_KM_PER_BIKE
    savings           = naive_cost - lp_cost
    savings_pct       = savings / naive_cost * 100

    print(f"\n=== Cost Comparison ===")
    print(f"  Naive dispatch cost (avg distance): €{naive_cost:.2f}")
    print(f"  LP optimized cost:                  €{lp_cost:.2f}")
    print(f"  Savings:                            €{savings:.2f}  ({savings_pct:.1f}%)")
    print(f"  Total bikes to rebalance:           {int(total_bikes_moved)}")

    # ── Post-rebalancing counts ───────────────────────────────────────
    cluster_meta['final_bikes'] = (
        cluster_meta['initial_bikes']
        - x_opt.sum(axis=1).round()
        + x_opt.sum(axis=0).round()
    ).astype(int)

    # ── Chart: Before / Demand / After ───────────────────────────────
    fig, ax = plt.subplots(figsize=(13, 6))
    x_pos = np.arange(n)
    w = 0.25

    b1 = ax.bar(x_pos - w, cluster_meta['initial_bikes'], w,
                label='Initial (Uniform)', color='#457b9d', alpha=0.85)
    b2 = ax.bar(x_pos,     cluster_meta['demand'],        w,
                label='Evening Demand',   color='#e63946', alpha=0.85)
    b3 = ax.bar(x_pos + w, cluster_meta['final_bikes'],   w,
                label='After LP Rebalancing', color='#2a9d8f', alpha=0.85)

    for bars in [b1, b2, b3]:
        for bar in bars:
            h = int(bar.get_height())
            ax.text(bar.get_x() + bar.get_width()/2, h + 3,
                    str(h), ha='center', va='bottom', fontsize=8)

    ax.set_xticks(x_pos)
    ax.set_xticklabels([l.replace(' ', '\n') for l in cluster_meta['cluster_label']], fontsize=9)
    ax.set_ylabel('Number of Bikes')
    ax.set_title(
        f'Bike Availability: Before vs Demand vs After LP Rebalancing\n'
        f'Fleet = {TOTAL_FLEET} bikes  |  LP Savings vs Naive Dispatch: €{savings:.2f} ({savings_pct:.1f}%)',
        fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2, axis='y')
    plt.tight_layout()
    plt.savefig('charts/rebalancing_before_after.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("✓ Saved charts/rebalancing_before_after.png")

    # ── Save outputs ──────────────────────────────────────────────────
    plan_df.to_csv('output/rebalancing_plan.csv', index=False)
    cluster_meta.to_csv('output/cluster_scenario.csv', index=False)
    print("✓ Saved output/rebalancing_plan.csv")
    print("✓ Saved output/cluster_scenario.csv")

else:
    print("LP could not find a solution. Check constraints.")