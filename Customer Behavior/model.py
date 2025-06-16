import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA

# ----------------------------
# 1. Prepare / load data
# ----------------------------
# Replace this simulation with your actual df_history (historical behavior DataFrame)
# Columns: ['clicks_last_hour', 'avg_time_between_clicks', 'session_length', 
#           'num_failed_logins', 'device_change_rate', 'location_variance',
#           'browser_jump_freq', 'actions_per_session']

np.random.seed(42)
n_history = 200
cluster_A = np.column_stack([
    np.random.poisson(lam=20, size=n_history // 2),
    np.random.exponential(scale=40, size=n_history // 2),
    np.random.uniform(300, 2000, size=n_history // 2),
    np.random.poisson(lam=0.2, size=n_history // 2),
    np.random.poisson(lam=1, size=n_history // 2),
    np.random.poisson(lam=1, size=n_history // 2),
    np.random.poisson(lam=1, size=n_history // 2),
    np.random.poisson(lam=25, size=n_history // 2)
])
cluster_B = np.column_stack([
    np.random.poisson(lam=50, size=n_history // 2),
    np.random.exponential(scale=10, size=n_history // 2),
    np.random.uniform(1000, 3000, size=n_history // 2),
    np.random.poisson(lam=1, size=n_history // 2),
    np.random.poisson(lam=2, size=n_history // 2),
    np.random.poisson(lam=2, size=n_history // 2),
    np.random.poisson(lam=2, size=n_history // 2),
    np.random.poisson(lam=60, size=n_history // 2)
])
history = np.vstack([cluster_A, cluster_B])
columns = [
    'clicks_last_hour', 'avg_time_between_clicks', 'session_length',
    'num_failed_logins', 'device_change_rate', 'location_variance',
    'browser_jump_freq', 'actions_per_session'
]
df_history = pd.DataFrame(history, columns=columns)

# (Optional) New point(s) to check; replace with your actual new data for this user
df_new = pd.DataFrame([
    {
        'clicks_last_hour': 18,
        'avg_time_between_clicks': 45,
        'session_length': 400,
        'num_failed_logins': 0,
        'device_change_rate': 1,
        'location_variance': 1,
        'browser_jump_freq': 1,
        'actions_per_session': 22
    },
    {
        'clicks_last_hour': 200,
        'avg_time_between_clicks': 2,
        'session_length': 50,
        'num_failed_logins': 5,
        'device_change_rate': 5,
        'location_variance': 5,
        'browser_jump_freq': 5,
        'actions_per_session': 220
    }
])

# ---------------------------------------------------
# 2. Standardize historical features
# ---------------------------------------------------
scaler = StandardScaler()
X_hist_scaled = scaler.fit_transform(df_history)  # shape (n_history, n_features)

# ---------------------------------------------------
# 3. Fit Gaussian Mixture Model on historical data
# ---------------------------------------------------
n_components = 2  # or choose via BIC/AIC
gmm = GaussianMixture(n_components=n_components, covariance_type='full', random_state=42)
gmm.fit(X_hist_scaled)

# ---------------------------------------------------
# 4. Compute log-likelihoods and threshold for anomalies
# ---------------------------------------------------
log_likelihoods = gmm.score_samples(X_hist_scaled)  # array of shape (n_history,)
# Choose threshold: e.g., 5th percentile of historical log-likelihoods
threshold = np.percentile(log_likelihoods, 5)
# Identify historical anomalies (for illustration; usually few if threshold low)
is_hist_anomaly = log_likelihoods < threshold

# ---------------------------------------------------
# 5. Prepare new points for anomaly check
# ---------------------------------------------------
if df_new is not None and not df_new.empty:
    X_new_scaled = scaler.transform(df_new)
    loglik_new = gmm.score_samples(X_new_scaled)
    is_new_anomaly = loglik_new < threshold
    # We don't color by cluster; just mark anomalies vs normal
    X_new_pca_flag = True
    X_new_pca = None  # placeholder; will compute after PCA fit below
else:
    is_new_anomaly = np.array([])
    loglik_new = np.array([])
    X_new_pca_flag = False

# ---------------------------------------------------
# 6. Fit PCA on historical scaled data for 2D projection
# ---------------------------------------------------
pca = PCA(n_components=2, random_state=42)
X_hist_pca = pca.fit_transform(X_hist_scaled)  # shape (n_history, 2)

if X_new_pca_flag:
    X_new_pca = pca.transform(X_new_scaled)  # shape (n_new, 2)

# ---------------------------------------------------
# 7. Prepare output directory
# ---------------------------------------------------
output_dir = "plots"
os.makedirs(output_dir, exist_ok=True)
user_id = "user123"  # replace with real user ID if available
plot_filename = os.path.join(output_dir, f"{user_id}_cluster_anomaly_simple.png")

# ---------------------------------------------------
# 8. Plotting: single-color normal cluster + anomalies
# ---------------------------------------------------
plt.figure(figsize=(8, 6))

# a) Plot all historical normal points in one color (e.g., blue)
idx_normal = ~is_hist_anomaly
if np.any(idx_normal):
    plt.scatter(
        X_hist_pca[idx_normal, 0], X_hist_pca[idx_normal, 1],
        c='blue', label='Historical normal behavior', alpha=0.6, edgecolors='w', s=50
    )

# b) Plot historical anomalies in red crosses
if np.any(is_hist_anomaly):
    plt.scatter(
        X_hist_pca[is_hist_anomaly, 0], X_hist_pca[is_hist_anomaly, 1],
        c='red', marker='x', label='Historical anomalies', s=80, linewidths=2
    )

# c) Plot new points: normal vs anomaly
if X_new_pca_flag:
    # Normal new points
    idx_norm_new = ~is_new_anomaly
    if np.any(idx_norm_new):
        plt.scatter(
            X_new_pca[idx_norm_new, 0], X_new_pca[idx_norm_new, 1],
            c='green', marker='o', label='New normal', s=100, edgecolors='k'
        )
    # Anomalous new points
    idx_anom_new = is_new_anomaly
    if np.any(idx_anom_new):
        plt.scatter(
            X_new_pca[idx_anom_new, 0], X_new_pca[idx_anom_new, 1],
            c='red', marker='X', label='New anomalies', s=120, edgecolors='k'
        )

# Optionally, mark the overall GMM “center” (mean of all data) or component means:
# For simplicity, you could mark the overall data centroid:
overall_center = X_hist_pca.mean(axis=0)
plt.scatter(
    overall_center[0], overall_center[1],
    c='black', marker='D', label='Overall centroid', s=80
)

plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.title('User Behavior: Normal Cluster vs Anomalies')
plt.legend(loc='best', fontsize='small')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()

# Save and show
plt.savefig(plot_filename)
print(f"Saved visualization to: {plot_filename}")
plt.show()

# ---------------------------------------------------
# 9. Print summary for layman
# ---------------------------------------------------
print(f"Total historical points: {len(df_history)}")
print(f"Historical anomalies flagged: {np.sum(is_hist_anomaly)}")
if X_new_pca_flag:
    for idx, (ll, is_anom) in enumerate(zip(loglik_new, is_new_anomaly)):
        status = "ANOMALY" if is_anom else "normal"
        print(f"New point {idx}: log-likelihood = {ll:.3f} -> {status}")

print("Interpretation for a layman:")
print("- Blue dots form the main “cloud” of past behavior (the normal cluster).")
print("- Red crosses among past data are rare outliers (historical anomalies).")
print("- Green circles (if any) are new sessions behaving normally within that cloud.")
print("- Red X’s (if any) are new sessions flagged as unusual relative to past behavior.")
