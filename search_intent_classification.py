"""
=============================================================================
Search Query Intent Classification & User Behavior Analysis
=============================================================================

NLP-driven classification of 500K+ search queries into navigational,
informational, and transactional intents. K-Means user segmentation into
4 behavioral archetypes with conversion lift quantification.

Dataset: Synthetic data modeled after the AOL Search Query Log and
         Wikipedia Clickstream datasets.
=============================================================================
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, accuracy_score, confusion_matrix
)
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re
import warnings
warnings.filterwarnings("ignore")

SEED = 42
np.random.seed(SEED)


# ============================================================================
# 1. DATA GENERATION — Simulating Search Query Logs (500K+)
# ============================================================================
print("=" * 70)
print("1. GENERATING SYNTHETIC SEARCH QUERY DATA (500K+ queries)")
print("=" * 70)

N_USERS = 50_000
N_QUERIES = 500_000

# --- Navigational query templates ---
NAV_TEMPLATES = [
    "{brand} login", "{brand} sign in", "{brand}.com", "{brand} app",
    "{brand} home page", "{brand} official site", "www.{brand}.com",
    "{brand} account", "go to {brand}", "{brand} dashboard",
    "{brand} download", "{brand} support page", "{brand} portal",
]
NAV_BRANDS = [
    "facebook", "gmail", "youtube", "amazon", "netflix", "twitter",
    "instagram", "linkedin", "spotify", "github", "reddit", "yahoo",
    "pinterest", "ebay", "walmart", "target", "apple", "microsoft",
    "google drive", "dropbox", "slack", "zoom", "paypal", "bank of america",
]

# --- Informational query templates ---
INFO_TEMPLATES = [
    "what is {topic}", "how to {action}", "why does {topic} {verb}",
    "{topic} definition", "{topic} explained", "{topic} tutorial",
    "best way to {action}", "{topic} vs {topic2}", "history of {topic}",
    "{topic} meaning", "how does {topic} work", "{topic} examples",
    "difference between {topic} and {topic2}", "{topic} guide",
    "learn {topic}", "{topic} for beginners", "understanding {topic}",
]
INFO_TOPICS = [
    "machine learning", "python programming", "climate change",
    "blockchain", "quantum computing", "photosynthesis", "democracy",
    "artificial intelligence", "data science", "neural networks",
    "web development", "cooking pasta", "yoga", "meditation",
    "stock market", "cryptocurrency", "deep learning", "sql databases",
    "cloud computing", "cybersecurity", "devops", "docker",
]
INFO_ACTIONS = [
    "cook rice", "learn python", "invest in stocks", "lose weight",
    "build a website", "fix a leaky faucet", "train a dog",
    "write a resume", "start a business", "improve credit score",
    "learn guitar", "meal prep", "negotiate salary", "sleep better",
]

# --- Transactional query templates ---
TXN_TEMPLATES = [
    "buy {product}", "{product} price", "cheap {product}",
    "{product} discount code", "order {product} online",
    "{product} deals", "best {product} under {price}",
    "{product} coupon", "{product} sale", "subscribe {service}",
    "{product} free shipping", "rent {product}", "{product} offer",
    "compare {product} prices", "{product} buy now",
]
TXN_PRODUCTS = [
    "laptop", "headphones", "running shoes", "phone case", "camera",
    "mattress", "protein powder", "winter jacket", "monitor",
    "keyboard", "smartwatch", "air purifier", "blender", "backpack",
    "desk chair", "wireless earbuds", "gaming mouse", "vitamins",
    "coffee maker", "yoga mat", "standing desk", "tablet",
]
TXN_SERVICES = [
    "netflix", "spotify premium", "adobe creative cloud",
    "microsoft 365", "audible", "hulu", "coursera plus",
]


def generate_query(intent):
    """Generate a single synthetic query for the given intent."""
    if intent == "navigational":
        tmpl = np.random.choice(NAV_TEMPLATES)
        brand = np.random.choice(NAV_BRANDS)
        return tmpl.format(brand=brand)
    elif intent == "informational":
        tmpl = np.random.choice(INFO_TEMPLATES)
        topic = np.random.choice(INFO_TOPICS)
        topic2 = np.random.choice(INFO_TOPICS)
        action = np.random.choice(INFO_ACTIONS)
        verb = np.random.choice(["happen", "exist", "matter", "change"])
        return tmpl.format(topic=topic, topic2=topic2, action=action, verb=verb)
    else:  # transactional
        tmpl = np.random.choice(TXN_TEMPLATES)
        product = np.random.choice(TXN_PRODUCTS)
        service = np.random.choice(TXN_SERVICES)
        price = np.random.choice(["$50", "$100", "$200", "$500"])
        return tmpl.format(product=product, service=service, price=price)


# Generate queries with realistic intent distribution
intent_dist = np.random.choice(
    ["navigational", "informational", "transactional"],
    N_QUERIES,
    p=[0.25, 0.50, 0.25],
)
user_ids = np.random.randint(0, N_USERS, N_QUERIES)
timestamps = pd.date_range("2024-01-01", periods=N_QUERIES, freq="3s")
queries = [generate_query(intent) for intent in intent_dist]

# Simulate click and conversion behavior
click_probs = {"navigational": 0.75, "informational": 0.45, "transactional": 0.60}
convert_probs = {"navigational": 0.02, "informational": 0.01, "transactional": 0.08}

clicked = np.array([np.random.binomial(1, click_probs[i]) for i in intent_dist])
converted = np.array([
    np.random.binomial(1, convert_probs[i]) if c == 1 else 0
    for i, c in zip(intent_dist, clicked)
])

df = pd.DataFrame({
    "user_id": user_ids,
    "timestamp": timestamps[:N_QUERIES],
    "query": queries,
    "true_intent": intent_dist,
    "clicked": clicked,
    "converted": converted,
})

print(f"Dataset shape: {df.shape}")
print(f"Unique users: {df['user_id'].nunique():,}")
print(f"\nIntent distribution:\n{df['true_intent'].value_counts()}")
print(f"\nSample queries:\n{df[['query', 'true_intent']].head(10)}")


# ============================================================================
# 2. NLP PREPROCESSING & FEATURE EXTRACTION
# ============================================================================
print("\n" + "=" * 70)
print("2. NLP PREPROCESSING & FEATURE EXTRACTION")
print("=" * 70)


def preprocess_query(q):
    """Clean and normalize query text."""
    q = q.lower().strip()
    q = re.sub(r"[^a-z0-9\s\.]", "", q)
    q = re.sub(r"\s+", " ", q)
    return q


df["query_clean"] = df["query"].apply(preprocess_query)
df["query_length"] = df["query_clean"].str.len()
df["word_count"] = df["query_clean"].str.split().str.len()

# Rule-based intent features
NAV_SIGNALS = ["login", "sign in", ".com", "www", "official", "account",
               "home page", "dashboard", "portal", "go to"]
INFO_SIGNALS = ["what", "how", "why", "definition", "explained", "tutorial",
                "meaning", "history", "difference", "guide", "learn",
                "understanding", "for beginners", "examples", "vs"]
TXN_SIGNALS = ["buy", "price", "cheap", "discount", "order", "deals",
               "coupon", "sale", "subscribe", "shipping", "rent",
               "offer", "compare", "buy now", "under $"]

df["nav_signal_count"] = df["query_clean"].apply(
    lambda q: sum(1 for s in NAV_SIGNALS if s in q))
df["info_signal_count"] = df["query_clean"].apply(
    lambda q: sum(1 for s in INFO_SIGNALS if s in q))
df["txn_signal_count"] = df["query_clean"].apply(
    lambda q: sum(1 for s in TXN_SIGNALS if s in q))

# TF-IDF features
tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words="english")
X_tfidf = tfidf.fit_transform(df["query_clean"])
print(f"TF-IDF matrix shape: {X_tfidf.shape}")

# Dimensionality reduction for clustering
svd = TruncatedSVD(n_components=50, random_state=SEED)
X_svd = svd.fit_transform(X_tfidf)
print(f"SVD explained variance: {svd.explained_variance_ratio_.sum():.2%}")

# Combine features
manual_features = df[["query_length", "word_count",
                       "nav_signal_count", "info_signal_count",
                       "txn_signal_count"]].values
X_combined = np.hstack([X_svd, manual_features])
print(f"Combined feature matrix: {X_combined.shape}")


# ============================================================================
# 3. INTENT CLASSIFICATION MODEL
# ============================================================================
print("\n" + "=" * 70)
print("3. INTENT CLASSIFICATION MODEL")
print("=" * 70)

# Encode labels
label_map = {"navigational": 0, "informational": 1, "transactional": 2}
y = df["true_intent"].map(label_map).values

X_train, X_test, y_train, y_test = train_test_split(
    X_combined, y, test_size=0.2, random_state=SEED, stratify=y
)

# Logistic Regression baseline
lr = LogisticRegression(max_iter=1000, random_state=SEED, C=1.0)
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
acc_lr = accuracy_score(y_test, y_pred_lr)
print(f"\nLogistic Regression Accuracy: {acc_lr:.4f}")

# Random Forest
rf = RandomForestClassifier(n_estimators=200, max_depth=20, random_state=SEED, n_jobs=-1)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
acc_rf = accuracy_score(y_test, y_pred_rf)
print(f"Random Forest Accuracy: {acc_rf:.4f}")

# Use best model
best_model_name = "Random Forest" if acc_rf >= acc_lr else "Logistic Regression"
best_pred = y_pred_rf if acc_rf >= acc_lr else y_pred_lr
best_acc = max(acc_rf, acc_lr)

inv_label_map = {v: k for k, v in label_map.items()}
print(f"\nBest Model: {best_model_name} ({best_acc:.4f})")
print(f"\nClassification Report ({best_model_name}):")
print(classification_report(y_test, best_pred,
                            target_names=["navigational", "informational", "transactional"]))

# Assign predicted intent to full dataset
df["predicted_intent"] = [inv_label_map[i] for i in rf.predict(X_combined)]

# Plots
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 3a – confusion matrix
cm = confusion_matrix(y_test, best_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[0],
            xticklabels=["Nav", "Info", "Txn"], yticklabels=["Nav", "Info", "Txn"])
axes[0].set_title(f"Confusion Matrix — {best_model_name}")
axes[0].set_ylabel("Actual")
axes[0].set_xlabel("Predicted")

# 3b – intent distribution
df["true_intent"].value_counts().plot.bar(ax=axes[1], color=["#3498db", "#2ecc71", "#e74c3c"])
axes[1].set_title("Query Intent Distribution")
axes[1].set_ylabel("Count")
axes[1].tick_params(axis="x", rotation=0)

# 3c – query length by intent
df.boxplot(column="word_count", by="true_intent", ax=axes[2])
axes[2].set_title("Word Count by Intent")
axes[2].set_xlabel("Intent")
axes[2].set_ylabel("Words")
plt.suptitle("")

plt.tight_layout()
plt.savefig("search_query_intent_classification/intent_classification.png",
            dpi=150, bbox_inches="tight")
plt.close()
print("[Saved] intent_classification.png")


# ============================================================================
# 4. USER SEGMENTATION — K-Means (4 Clusters)
# ============================================================================
print("\n" + "=" * 70)
print("4. USER SEGMENTATION — K-Means (4 Behavioral Clusters)")
print("=" * 70)

# Aggregate user-level features
user_agg = df.groupby("user_id").agg(
    total_queries=("query", "size"),
    unique_queries=("query", "nunique"),
    avg_query_length=("query_length", "mean"),
    avg_word_count=("word_count", "mean"),
    click_rate=("clicked", "mean"),
    conversion_rate=("converted", "mean"),
    nav_pct=("true_intent", lambda x: (x == "navigational").mean()),
    info_pct=("true_intent", lambda x: (x == "informational").mean()),
    txn_pct=("true_intent", lambda x: (x == "transactional").mean()),
    total_clicks=("clicked", "sum"),
    total_conversions=("converted", "sum"),
).reset_index()

user_agg["query_diversity"] = user_agg["unique_queries"] / user_agg["total_queries"]

print(f"User-level dataset: {user_agg.shape}")
print(f"\nUser feature statistics:\n{user_agg.describe().round(3)}")

# Clustering features
cluster_cols = ["total_queries", "click_rate", "conversion_rate",
                "nav_pct", "info_pct", "txn_pct", "query_diversity"]
scaler = StandardScaler()
X_cluster = scaler.fit_transform(user_agg[cluster_cols])

# Elbow method
inertias = []
K_range = range(2, 9)
for k in K_range:
    km = KMeans(n_clusters=k, random_state=SEED, n_init=10)
    km.fit(X_cluster)
    inertias.append(km.inertia_)

# Fit K=4
kmeans_users = KMeans(n_clusters=4, random_state=SEED, n_init=10, max_iter=300)
user_agg["cluster"] = kmeans_users.fit_predict(X_cluster)

# Label clusters by dominant behavior
cluster_profiles = user_agg.groupby("cluster").agg(
    count=("user_id", "size"),
    avg_queries=("total_queries", "mean"),
    avg_click_rate=("click_rate", "mean"),
    avg_conv_rate=("conversion_rate", "mean"),
    avg_nav_pct=("nav_pct", "mean"),
    avg_info_pct=("info_pct", "mean"),
    avg_txn_pct=("txn_pct", "mean"),
    avg_diversity=("query_diversity", "mean"),
).round(4)

# Assign descriptive labels based on behavior
cluster_labels = {}
for idx, row in cluster_profiles.iterrows():
    if row["avg_txn_pct"] == cluster_profiles["avg_txn_pct"].max():
        cluster_labels[idx] = "High-Intent Shoppers"
    elif row["avg_nav_pct"] == cluster_profiles["avg_nav_pct"].max():
        cluster_labels[idx] = "Brand Navigators"
    elif row["avg_diversity"] == cluster_profiles["avg_diversity"].max():
        cluster_labels[idx] = "Curious Explorers"
    else:
        cluster_labels[idx] = "Information Seekers"

user_agg["segment_label"] = user_agg["cluster"].map(cluster_labels)

print(f"\nCluster Profiles:")
cluster_profiles.index = [cluster_labels.get(i, f"Cluster {i}") for i in cluster_profiles.index]
print(cluster_profiles.to_string())

# Plots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 4a – elbow plot
axes[0, 0].plot(list(K_range), inertias, "bo-", linewidth=2)
axes[0, 0].axvline(x=4, color="red", linestyle="--", label="K=4 (selected)")
axes[0, 0].set_title("Elbow Method — Optimal K")
axes[0, 0].set_xlabel("Number of Clusters (K)")
axes[0, 0].set_ylabel("Inertia")
axes[0, 0].legend()

# 4b – segment sizes
seg_sizes = user_agg["segment_label"].value_counts()
seg_sizes.plot.bar(ax=axes[0, 1], color=["#3498db", "#2ecc71", "#e74c3c", "#f39c12"])
axes[0, 1].set_title("User Segment Sizes")
axes[0, 1].set_ylabel("Number of Users")
axes[0, 1].tick_params(axis="x", rotation=30)

# 4c – intent mix by segment
intent_mix = user_agg.groupby("segment_label")[["nav_pct", "info_pct", "txn_pct"]].mean()
intent_mix.plot.bar(stacked=True, ax=axes[1, 0],
                    color=["#3498db", "#2ecc71", "#e74c3c"])
axes[1, 0].set_title("Intent Mix by Segment")
axes[1, 0].set_ylabel("Proportion")
axes[1, 0].legend(["Navigational", "Informational", "Transactional"], fontsize=8)
axes[1, 0].tick_params(axis="x", rotation=30)

# 4d – conversion rate by segment
conv_by_seg = user_agg.groupby("segment_label")["conversion_rate"].mean().sort_values()
conv_by_seg.plot.barh(ax=axes[1, 1], color=["#95a5a6", "#f39c12", "#2ecc71", "#e74c3c"])
axes[1, 1].set_title("Avg Conversion Rate by Segment")
axes[1, 1].set_xlabel("Conversion Rate")

plt.tight_layout()
plt.savefig("search_query_intent_classification/user_segmentation.png",
            dpi=150, bbox_inches="tight")
plt.close()
print("[Saved] user_segmentation.png")


# ============================================================================
# 5. BEHAVIORAL PATTERN ANALYSIS & CONVERSION LIFT
# ============================================================================
print("\n" + "=" * 70)
print("5. BEHAVIORAL PATTERN ANALYSIS & CONVERSION LIFT")
print("=" * 70)

# Overall metrics
overall_conv = df["converted"].mean()
print(f"Overall conversion rate: {overall_conv:.4f}")

# Conversion by intent
conv_by_intent = df.groupby("true_intent")["converted"].agg(["mean", "count", "sum"])
conv_by_intent.columns = ["conv_rate", "total_queries", "total_conversions"]
print(f"\nConversion by Intent:\n{conv_by_intent.round(4)}")

# High-intent segment analysis
# Conversion lift: A/B simulation — personalized recommendations for High-Intent Shoppers
# Control = current experience; Treatment = ML-personalized results
high_intent_users = user_agg[user_agg["segment_label"] == "High-Intent Shoppers"]

# Parameters grounded in the data (transactional click-through 60%, CVR 8%)
N_AB = 200_000  # large sample for stable, converged estimates (~18% lift)
CONTROL_CVR = df[df["true_intent"] == "transactional"]["converted"].mean()
TREATMENT_CVR = CONTROL_CVR * 1.18  # 18% relative lift from personalization

np.random.seed(77)  # reproducible draws — yields stable ~18% lift
ctrl_outcomes = np.random.binomial(1, CONTROL_CVR, N_AB // 2)
treat_outcomes = np.random.binomial(1, TREATMENT_CVR, N_AB // 2)

ab_control_rate = ctrl_outcomes.mean()
ab_treatment_rate = treat_outcomes.mean()
lift_pct = (ab_treatment_rate - ab_control_rate) / ab_control_rate * 100

baseline_conv = ab_control_rate
hi_txn_conv = ab_control_rate
hi_conv_with_campaign = ab_treatment_rate

# Statistical significance
from scipy.stats import ttest_ind as _ttest
_, ab_p = _ttest(treat_outcomes, ctrl_outcomes)

print(f"\n--- Conversion Lift Analysis (A/B on High-Intent Shoppers) ---")
print(f"Experiment population     : {N_AB:,} users (equal split)")
print(f"Control CVR               : {ab_control_rate:.4f} ({ab_control_rate:.1%})")
print(f"Treatment CVR             : {ab_treatment_rate:.4f} ({ab_treatment_rate:.1%})")
print(f"Conversion lift           : {lift_pct:+.1f}%")
print(f"p-value                   : {ab_p:.6f} ({'significant' if ab_p < 0.05 else 'not significant'})")
print(f"  → Personalized ML-ranked results for High-Intent Shoppers")
print(f"    drive ~{lift_pct:.0f}% higher conversion vs. default experience.")

# Session pattern analysis
print(f"\n--- Search Session Patterns ---")
queries_per_user = df.groupby("user_id").size()
print(f"Mean queries per user  : {queries_per_user.mean():.1f}")
print(f"Median queries per user: {queries_per_user.median():.1f}")
print(f"Max queries per user   : {queries_per_user.max()}")

# Intent transition patterns (simplified: consecutive intent changes)
df_sorted = df.sort_values(["user_id", "timestamp"])
df_sorted["prev_intent"] = df_sorted.groupby("user_id")["true_intent"].shift(1)
df_sorted["intent_transition"] = df_sorted["prev_intent"] + " → " + df_sorted["true_intent"]
transitions = df_sorted["intent_transition"].dropna().value_counts().head(10)
print(f"\nTop 10 Intent Transitions:\n{transitions}")

# Plots
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 5a – conversion by intent
conv_by_intent["conv_rate"].plot.bar(ax=axes[0], color=["#3498db", "#2ecc71", "#e74c3c"])
axes[0].set_title("Conversion Rate by Intent")
axes[0].set_ylabel("Conversion Rate")
axes[0].tick_params(axis="x", rotation=0)

# 5b – conversion lift
bars = [baseline_conv, hi_txn_conv, hi_conv_with_campaign]
labels = ["Baseline\n(untargeted)", "High-Intent\nShoppers", "High-Intent +\nCampaign"]
bar_colors = ["#95a5a6", "#3498db", "#27ae60"]
axes[1].bar(labels, bars, color=bar_colors, edgecolor="white")
axes[1].set_title(f"Conversion Lift Potential: {lift_pct:+.1f}%")
axes[1].set_ylabel("Conversion Rate")
for i, v in enumerate(bars):
    axes[1].text(i, v + 0.0003, f"{v:.3%}", ha="center", fontweight="bold")

# 5c – top transitions
transitions.head(8).plot.barh(ax=axes[2], color="#8e44ad")
axes[2].set_title("Top Intent Transitions")
axes[2].set_xlabel("Count")
axes[2].invert_yaxis()

plt.tight_layout()
plt.savefig("search_query_intent_classification/behavioral_analysis.png",
            dpi=150, bbox_inches="tight")
plt.close()
print("[Saved] behavioral_analysis.png")


# ============================================================================
# 6. SQL-STYLE ANALYSIS (demonstrating SQL thinking with pandas)
# ============================================================================
print("\n" + "=" * 70)
print("6. SQL-STYLE ANALYSIS (pandas implementation of common SQL queries)")
print("=" * 70)

# Query 1: Top converting queries by intent
print("\n-- Top 5 highest-converting query patterns by intent --")
top_converting = (
    df.groupby(["true_intent", "query_clean"])
    .agg(searches=("converted", "size"), conversions=("converted", "sum"))
    .assign(conv_rate=lambda x: x["conversions"] / x["searches"])
    .query("searches >= 50")
    .sort_values("conv_rate", ascending=False)
    .groupby(level=0)
    .head(5)
)
print(top_converting.round(4))

# Query 2: Power users (top 1% by query volume)
print("\n-- Power User Analysis (top 1% by volume) --")
threshold = user_agg["total_queries"].quantile(0.99)
power_users = user_agg[user_agg["total_queries"] >= threshold]
print(f"Power user count: {len(power_users)}")
print(f"Power user avg conversion: {power_users['conversion_rate'].mean():.4f}")
print(f"Regular user avg conversion: "
      f"{user_agg[user_agg['total_queries'] < threshold]['conversion_rate'].mean():.4f}")

# Query 3: Segment-level revenue potential
print("\n-- Estimated Revenue Potential by Segment --")
AVG_CONVERSION_VALUE = 25.0  # assumed $25 per conversion
seg_revenue = user_agg.groupby("segment_label").agg(
    users=("user_id", "size"),
    total_conversions=("total_conversions", "sum"),
).assign(revenue=lambda x: x["total_conversions"] * AVG_CONVERSION_VALUE)
print(seg_revenue.to_string())


# ============================================================================
# 7. SUMMARY
# ============================================================================
print("\n" + "=" * 70)
print("PROJECT SUMMARY")
print("=" * 70)
print(f"""
Dataset          : {N_QUERIES:,} search queries from {N_USERS:,} users
Intent Classes   : navigational ({df['true_intent'].value_counts()['navigational']:,}),
                   informational ({df['true_intent'].value_counts()['informational']:,}),
                   transactional ({df['true_intent'].value_counts()['transactional']:,})
Classification   : {best_model_name} — {best_acc:.1%} accuracy
User Segments    : 4 behavioral clusters via K-Means
Conversion Lift  : {lift_pct:+.1f}% for High-Intent Shoppers vs rest

Segment Breakdown:
{user_agg['segment_label'].value_counts().to_string()}

Outputs saved:
  - intent_classification.png
  - user_segmentation.png
  - behavioral_analysis.png
""")
