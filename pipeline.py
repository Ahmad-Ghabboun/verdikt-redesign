#!/usr/bin/env python3
"""
LMSYS Chatbot Arena — Complete Data Science Pipeline
Steps 1–4: Data Cleaning & Feature Engineering, EDA, Modeling, SHAP
"""

import os
import json
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV

# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def safe_name(s):
    """Create filesystem-safe model name."""
    return (s.replace(' ', '_')
             .replace('(', '')
             .replace(')', '')
             .replace('/', '_'))


def parse_json_col(val):
    """Parse JSON-encoded list column into a single joined string."""
    if pd.isna(val):
        return ""
    try:
        parsed = json.loads(val)
        if isinstance(parsed, list):
            return " ".join(str(x) for x in parsed)
        return str(parsed)
    except Exception:
        return str(val)


# ═══════════════════════════════════════════════════════════════════
# STEP 1 — DATA CLEANING & FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 1: DATA CLEANING & FEATURE ENGINEERING")
print("=" * 60)

os.makedirs('models',    exist_ok=True)
os.makedirs('eda_plots', exist_ok=True)
os.makedirs('shap_plots', exist_ok=True)

# 1.1  Load
print("\n[1/8] Loading dataset…")
DATA_PATH = '/Users/ahmadghabboun/Desktop/522 assignment/lmsys-chatbot-arena/train.csv'
df = pd.read_csv(DATA_PATH)
print(f"  Loaded {len(df):,} rows × {len(df.columns)} columns")

# 1.2  Sample
if len(df) > 50_000:
    df = df.sample(30_000, random_state=42).reset_index(drop=True)
    print(f"  Sampled to 30,000 rows (random_state=42)")

# 1.3  Parse JSON text columns
print("\n[2/8] Parsing JSON text columns…")
df['prompt_text']      = df['prompt'].apply(parse_json_col)
df['response_a_text']  = df['response_a'].apply(parse_json_col)
df['response_b_text']  = df['response_b'].apply(parse_json_col)
print("  Prompt, Response A, Response B parsed")

# 1.4  Target encoding: 0=A wins, 1=B wins, 2=tie
print("\n[3/8] Encoding target variable…")

def encode_winner(row):
    if row['winner_model_a'] == 1:
        return 0
    elif row['winner_model_b'] == 1:
        return 1
    else:
        return 2

df['target'] = df.apply(encode_winner, axis=1)
y = df['target']
print(f"  Counts: {dict(y.value_counts().sort_index())}  (0=A wins, 1=B wins, 2=tie)")

# 1.5  Structured features
print("\n[4/8] Engineering structured features…")
df['prompt_length']     = df['prompt_text'].str.len()
df['response_a_length'] = df['response_a_text'].str.len()
df['response_b_length'] = df['response_b_text'].str.len()
df['length_difference'] = (df['response_a_length'] - df['response_b_length']).abs()
df['length_ratio']      = df['response_a_length'] / (df['response_b_length'] + 1)
df['prompt_word_count'] = df['prompt_text'].str.split().str.len()
df['response_a_word_count'] = df['response_a_text'].str.split().str.len()
df['response_b_word_count'] = df['response_b_text'].str.split().str.len()
df['word_count_difference'] = (df['response_a_word_count'] - df['response_b_word_count']).abs()
df['word_count_ratio']      = df['response_a_word_count'] / (df['response_b_word_count'] + 1)
df['longer_response']       = (df['response_a_length'] > df['response_b_length']).astype(int)

# Regex features
code_pat = r"(`{3}|`)"
list_pat = r"(?m)(^\s*[-*]\s+|^\s*\d+\.\s+)"

df['response_a_has_code'] = df['response_a_text'].str.contains(code_pat, regex=True).astype(int)
df['response_b_has_code'] = df['response_b_text'].str.contains(code_pat, regex=True).astype(int)
df['response_a_has_list'] = df['response_a_text'].str.contains(list_pat, regex=True).astype(int)
df['response_b_has_list'] = df['response_b_text'].str.contains(list_pat, regex=True).astype(int)

STRUCT_COLS = [
    'prompt_length', 'response_a_length', 'response_b_length',
    'length_difference', 'length_ratio', 'prompt_word_count',
    'response_a_word_count', 'response_b_word_count',
    'word_count_difference', 'word_count_ratio', 'longer_response',
    'response_a_has_code', 'response_b_has_code',
    'response_a_has_list', 'response_b_has_list',
]

model_a_dummies = pd.get_dummies(df['model_a'], prefix='model_a')
model_b_dummies = pd.get_dummies(df['model_b'], prefix='model_b')
X_structured = pd.concat([df[STRUCT_COLS], model_a_dummies, model_b_dummies], axis=1)
print(f"  Structured matrix: {X_structured.shape}")

# Save metadata for Streamlit
top_models = (pd.concat([df['model_a'], df['model_b']])
                .value_counts().head(5).index.tolist())
joblib.dump(top_models, 'models/top_models.joblib')
joblib.dump(list(model_a_dummies.columns) + list(model_b_dummies.columns),
            'models/model_dummies_columns.joblib')
joblib.dump(list(X_structured.columns), 'models/structured_feature_columns.joblib')
print("  Saved Streamlit metadata")

# 1.6  Embeddings
print("\n[5/8] Creating semantic embeddings (all-MiniLM-L6-v2)…")
from sentence_transformers import SentenceTransformer
st_model = SentenceTransformer('all-MiniLM-L6-v2')

embedding_configs = [
    ('prompt_text', 'embeddings.npy', 'embeddings'),
    ('response_a_text', 'embeddings_response_a.npy', 'embeddings_a'),
    ('response_b_text', 'embeddings_response_b.npy', 'embeddings_b')
]

loaded_embeddings = {}

for col, fname, key in embedding_configs:
    if os.path.exists(fname):
        print(f"  Loading cached {key} from {fname}…")
        loaded_embeddings[key] = np.load(fname)
    else:
        print(f"  Encoding {col} (batch_size=64)…")
        emb = st_model.encode(df[col].tolist(), show_progress_bar=True, batch_size=64)
        np.save(fname, emb)
        loaded_embeddings[key] = emb
        print(f"  Saved {fname}")

# 1.7  PCA 384 → 50
print("\n[6/8] PCA Reduction…")
from sklearn.decomposition import PCA

# Prompt PCA (50 dims)
pca_prompt = PCA(n_components=50, random_state=42)
emb_prompt_red = pca_prompt.fit_transform(loaded_embeddings['embeddings'])
joblib.dump(pca_prompt, 'models/pca.joblib')

# Response A PCA (30 dims)
pca_res_a = PCA(n_components=30, random_state=42)
emb_a_red = pca_res_a.fit_transform(loaded_embeddings['embeddings_a'])
joblib.dump(pca_res_a, 'models/pca_response_a.joblib')

# Response B PCA (30 dims)
pca_res_b = PCA(n_components=30, random_state=42)
emb_b_red = pca_res_b.fit_transform(loaded_embeddings['embeddings_b'])
joblib.dump(pca_res_b, 'models/pca_response_b.joblib')

print("  Saved PCA models (prompt=50d, res_a=30d, res_b=30d)")

emb_cols_p = [f'emb_prompt_{i}' for i in range(50)]
emb_cols_a = [f'emb_res_a_{i}' for i in range(30)]
emb_cols_b = [f'emb_res_b_{i}' for i in range(30)]

X_emb = pd.DataFrame(
    np.hstack([emb_prompt_red, emb_a_red, emb_b_red]),
    columns=emb_cols_p + emb_cols_a + emb_cols_b,
    index=X_structured.index
)

# 1.8  Concatenate & save
print("\n[7/8] Building final feature matrix…")
X = pd.concat([X_structured, X_emb], axis=1).fillna(0)
print(f"  Final X shape: {X.shape}")

print("\n[8/8] Saving X and y as parquet…")
X.to_parquet('X_features.parquet', index=False)
y.to_frame('target').to_parquet('y_target.parquet', index=False)
joblib.dump(list(X.columns), 'models/all_feature_columns.joblib')
print("  Saved X_features.parquet, y_target.parquet, models/all_feature_columns.joblib")

print("\n✓ STEP 1 COMPLETE")


# ═══════════════════════════════════════════════════════════════════
# STEP 2 — DESCRIPTIVE ANALYTICS (EDA)
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 2: DESCRIPTIVE ANALYTICS (EDA)")
print("=" * 60)

LABEL_MAP = {0: 'Model A Wins', 1: 'Model B Wins', 2: 'Tie'}
df['winner_label'] = y.map(LABEL_MAP)
PALETTE = {'Model A Wins': '#4C72B0', 'Model B Wins': '#DD8452', 'Tie': '#55A868'}

# ── Plot 1: Target distribution ──────────────────────────────
print("\n[Plot 1/6] Target distribution bar chart…")
fig, ax = plt.subplots(figsize=(8, 5))
counts = y.value_counts().sort_index()
labels = [LABEL_MAP[i] for i in counts.index]
colors = [list(PALETTE.values())[i] for i in counts.index]
bars = ax.bar(labels, counts.values, color=colors, edgecolor='white', linewidth=1.2)
for bar, cnt in zip(bars, counts.values):
    pct = cnt / len(y) * 100
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(counts.values) * 0.01,
            f'{pct:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
ax.set_title('Target Distribution: Winner Outcomes', fontsize=14, fontweight='bold')
ax.set_ylabel('Count')
ax.set_xlabel('Outcome')
ax.set_ylim(0, max(counts.values) * 1.12)
plt.tight_layout()
plt.savefig('eda_plots/01_target_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved eda_plots/01_target_distribution.png")

# ── Plot 2: Response length histogram ────────────────────────
print("[Plot 2/6] Response length distribution…")
fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(df['response_a_length'].clip(upper=6000), bins=60,
        alpha=0.6, color='#4C72B0', label='Response A', density=True)
ax.hist(df['response_b_length'].clip(upper=6000), bins=60,
        alpha=0.6, color='#DD8452', label='Response B', density=True)
ax.set_title('Response Length Distribution: A vs B', fontsize=14, fontweight='bold')
ax.set_xlabel('Character Count (capped at 6,000)')
ax.set_ylabel('Density')
ax.legend(fontsize=11)
plt.tight_layout()
plt.savefig('eda_plots/02_response_length_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved eda_plots/02_response_length_distribution.png")

# ── Plot 3: Length difference boxplot ────────────────────────
print("[Plot 3/6] Length difference vs outcome boxplot…")
order = ['Model A Wins', 'Model B Wins', 'Tie']
groups_data = [
    df[df['winner_label'] == lbl]['length_difference'].clip(upper=5000).values
    for lbl in order
]
fig, ax = plt.subplots(figsize=(9, 5))
bp = ax.boxplot(groups_data, labels=order, patch_artist=True, notch=False)
for patch, color in zip(bp['boxes'], list(PALETTE.values())):
    patch.set_facecolor(color)
    patch.set_alpha(0.75)
ax.set_title('|Response A − Response B| Length by Winner', fontsize=13, fontweight='bold')
ax.set_ylabel('Length Difference in Characters (capped at 5,000)')
ax.set_xlabel('Winner Outcome')
plt.tight_layout()
plt.savefig('eda_plots/03_length_difference_boxplot.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved eda_plots/03_length_difference_boxplot.png")

# ── Plot 4: Top 10 model pairs ───────────────────────────────
print("[Plot 4/6] Top 10 model pairs…")
df['model_pair'] = df['model_a'].str[:22] + ' vs ' + df['model_b'].str[:22]
top_pairs = df['model_pair'].value_counts().head(10).sort_values()
fig, ax = plt.subplots(figsize=(11, 6))
ax.barh(range(len(top_pairs)), top_pairs.values,
        color='#4C72B0', edgecolor='white')
ax.set_yticks(range(len(top_pairs)))
ax.set_yticklabels(top_pairs.index, fontsize=9)
for i, v in enumerate(top_pairs.values):
    ax.text(v + 2, i, str(v), va='center', fontsize=9)
ax.set_title('Top 10 Most Frequent Model Pairs', fontsize=14, fontweight='bold')
ax.set_xlabel('Count')
plt.tight_layout()
plt.savefig('eda_plots/04_top_model_pairs.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved eda_plots/04_top_model_pairs.png")

# ── Plot 5: Prompt length violin ─────────────────────────────
print("[Plot 5/6] Prompt length violin plot…")
plot_df = df[['prompt_length', 'winner_label']].copy()
plot_df['prompt_length'] = plot_df['prompt_length'].clip(upper=4000)
fig, ax = plt.subplots(figsize=(9, 5))
sns.violinplot(data=plot_df, x='winner_label', y='prompt_length', ax=ax,
               palette=PALETTE, order=order, inner='quartile', cut=0)
ax.set_title('Prompt Length Distribution by Winner Outcome',
             fontsize=14, fontweight='bold')
ax.set_xlabel('Winner Outcome')
ax.set_ylabel('Prompt Character Length (capped at 4,000)')
plt.tight_layout()
plt.savefig('eda_plots/05_prompt_length_violin.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved eda_plots/05_prompt_length_violin.png")

# ── Plot 6: Correlation heatmap (structured numeric) ─────────
print("[Plot 6/6] Correlation heatmap…")
corr = X_structured[STRUCT_COLS].corr()
fig, ax = plt.subplots(figsize=(9, 7))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
            center=0, ax=ax, square=True, linewidths=0.5,
            annot_kws={'size': 10})
ax.set_title('Correlation Heatmap: Structured Features',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('eda_plots/06_correlation_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved eda_plots/06_correlation_heatmap.png")

print("\nEDA complete")


# ═══════════════════════════════════════════════════════════════════
# STEP 3 — PREDICTIVE MODELING
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 3: PREDICTIVE MODELING (5-fold CV)")
print("=" * 60)

from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb

X = pd.read_parquet('X_features.parquet')
y = pd.read_parquet('y_target.parquet')['target']

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
SCORING = ['accuracy', 'f1_weighted', 'neg_log_loss']

MODELS = {
    'Logistic Regression': LogisticRegression(
        max_iter=1000, random_state=42, n_jobs=-1),
    'Lasso (L1)': LogisticRegression(
        penalty='l1', solver='saga', max_iter=2000, random_state=42, n_jobs=-1),
    'Ridge (L2)': LogisticRegression(
        penalty='l2', max_iter=1000, random_state=42, n_jobs=-1),
    'CART Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(
        n_estimators=100, random_state=42, n_jobs=-1),
    'LightGBM': lgb.LGBMClassifier(
        random_state=42, n_jobs=-1, verbose=-1),
}

# Hyperparameter tuning for LightGBM
print("\n  Tuning LightGBM with GridSearchCV…")
lgb_params = {
    'learning_rate': [0.05, 0.1],
    'n_estimators': [100, 200],
    'num_leaves': [31, 63],
}
lgb_gs = GridSearchCV(lgb.LGBMClassifier(random_state=42, n_jobs=-1, verbose=-1),
                      lgb_params, cv=3, scoring='f1_weighted', n_jobs=1)
lgb_gs.fit(X, y)
MODELS['LightGBM'] = lgb_gs.best_estimator_
print(f"    Best LightGBM params: {lgb_gs.best_params_}")

results = []
for name, model in MODELS.items():
    print(f"\n  Training: {name}…")
    cv_res = cross_validate(model, X, y, cv=cv, scoring=SCORING, n_jobs=1)
    acc = cv_res['test_accuracy'].mean()
    f1  = cv_res['test_f1_weighted'].mean()
    ll  = -cv_res['test_neg_log_loss'].mean()
    print(f"    Accuracy={acc:.4f}  F1={f1:.4f}  LogLoss={ll:.4f}")
    results.append({
        'Model': name,
        'Accuracy':    round(acc, 4),
        'F1_Weighted': round(f1,  4),
        'Log_Loss':    round(ll,  4),
    })
    # Retrain on full data and save
    model.fit(X, y)
    fname = f'models/{safe_name(name)}.joblib'
    joblib.dump(model, fname)
    print(f"    Saved {fname}")

results_df = (pd.DataFrame(results)
                .sort_values('F1_Weighted', ascending=False)
                .reset_index(drop=True))
results_df.to_csv('model_comparison.csv', index=False)
print("\nModel comparison:")
print(results_df.to_string(index=False))
print("\n  Saved model_comparison.csv")
print("\n✓ STEP 3 COMPLETE")


# ═══════════════════════════════════════════════════════════════════
# STEP 3b — MLP NEURAL NETWORK
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 3b: MLP NEURAL NETWORK (Keras/TensorFlow)")
print("=" * 60)

os.environ.setdefault('KERAS_BACKEND', 'torch')
import keras
from sklearn.model_selection import train_test_split as _tts
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, log_loss as _log_loss

print("\n[1/5] Loading features and preparing train/test split…")
X_mlp = pd.read_parquet('X_features.parquet')
y_mlp = pd.read_parquet('y_target.parquet')['target']

X_train_m, X_test_m, y_train_m, y_test_m = _tts(
    X_mlp, y_mlp, test_size=0.2, random_state=42, stratify=y_mlp)
print(f"  Train: {X_train_m.shape}  Test: {X_test_m.shape}")

scaler_mlp = StandardScaler()
X_train_sc = scaler_mlp.fit_transform(X_train_m)
X_test_sc  = scaler_mlp.transform(X_test_m)
joblib.dump(scaler_mlp, 'models/mlp_scaler.joblib')
print("  Saved models/mlp_scaler.joblib")

print("\n[2/5] Building MLP architecture…")
keras.utils.set_random_seed(42)
n_features = X_train_sc.shape[1]
mlp_model = keras.Sequential([
    keras.layers.Input(shape=(n_features,)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(3, activation='softmax'),
], name='mlp_preference_predictor')
mlp_model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'],
)
mlp_model.summary()

print("\n[3/5] Training with EarlyStopping (max 50 epochs, batch_size=64)…")
_early_stop = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    min_delta=0.001,
    restore_best_weights=True,
    verbose=1,
)
mlp_history = mlp_model.fit(
    X_train_sc, y_train_m,
    epochs=50,
    batch_size=64,
    validation_split=0.1,
    callbacks=[_early_stop],
    verbose=1,
)
_stopped_epoch = len(mlp_history.history['loss'])
_best_epoch    = _stopped_epoch - _early_stop.patience if _stopped_epoch < 50 else _stopped_epoch
print(f"  Early stopping triggered at epoch {_stopped_epoch} (best weights from epoch {_best_epoch})")
# Save epoch number so app.py can display it
import json as _json
with open('models/mlp_training_meta.json', 'w') as _f:
    _json.dump({'stopped_epoch': _stopped_epoch, 'best_epoch': _best_epoch, 'max_epochs': 50}, _f)

print("\n[4/5] Saving training history plot…")
_fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(14, 5))
_ax1.plot(mlp_history.history['loss'],     label='Train Loss',     color='#4C72B0', linewidth=2)
_ax1.plot(mlp_history.history['val_loss'], label='Val Loss',       color='#DD8452', linewidth=2, linestyle='--')
_ax1.axvline(x=_best_epoch - 1, color='#FF6B6B', linestyle=':', linewidth=1.5, label=f'Best epoch ({_best_epoch})')
_ax1.set_title('Loss', fontsize=13, fontweight='bold')
_ax1.set_xlabel('Epoch'); _ax1.set_ylabel('Loss')
_ax1.legend(fontsize=10); _ax1.grid(True, alpha=0.3)
_ax2.plot(mlp_history.history['accuracy'],     label='Train Accuracy', color='#4C72B0', linewidth=2)
_ax2.plot(mlp_history.history['val_accuracy'], label='Val Accuracy',   color='#DD8452', linewidth=2, linestyle='--')
_ax2.axvline(x=_best_epoch - 1, color='#FF6B6B', linestyle=':', linewidth=1.5, label=f'Best epoch ({_best_epoch})')
_ax2.set_title('Accuracy', fontsize=13, fontweight='bold')
_ax2.set_xlabel('Epoch'); _ax2.set_ylabel('Accuracy')
_ax2.legend(fontsize=10); _ax2.grid(True, alpha=0.3)
plt.suptitle(
    f'MLP Neural Network — Training History (stopped epoch {_stopped_epoch}/{50}, EarlyStopping patience=5)',
    fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('eda_plots/mlp_training_history.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved eda_plots/mlp_training_history.png")

print("\n[5/5] Evaluating on test set and updating model_comparison.csv…")
_y_proba_m = mlp_model.predict(X_test_sc, verbose=0)
_y_pred_m  = _y_proba_m.argmax(axis=1)
mlp_acc = accuracy_score(y_test_m, _y_pred_m)
mlp_f1  = f1_score(y_test_m, _y_pred_m, average='weighted')
mlp_ll  = _log_loss(y_test_m, _y_proba_m)
print(f"  Accuracy={mlp_acc:.4f}  F1={mlp_f1:.4f}  LogLoss={mlp_ll:.4f}")

mlp_model.save('models/mlp_model.keras')
print("  Saved models/mlp_model.keras")

_comp = pd.read_csv('model_comparison.csv')
_comp = _comp[_comp['Model'] != 'MLP Neural Network']   # remove stale row if re-run
_mlp_row = pd.DataFrame([{
    'Model':       'MLP Neural Network',
    'Accuracy':    round(mlp_acc, 4),
    'F1_Weighted': round(mlp_f1,  4),
    'Log_Loss':    round(mlp_ll,  4),
}])
_comp = (pd.concat([_comp, _mlp_row], ignore_index=True)
           .sort_values('F1_Weighted', ascending=False)
           .reset_index(drop=True))
_comp.to_csv('model_comparison.csv', index=False)
print("  Updated model_comparison.csv:")
print(_comp.to_string(index=False))

print("\n✓ STEP 3b COMPLETE")


# ═══════════════════════════════════════════════════════════════════
# STEP 4 — SHAP EXPLAINABILITY
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 4: SHAP EXPLAINABILITY")
print("=" * 60)

import shap
from sklearn.model_selection import train_test_split

best_model_name = results_df.iloc[0]['Model']
print(f"\n  Best model by F1: {best_model_name}")

# Structured features only (exclude emb_* columns)
struct_only_cols = [c for c in X.columns if not c.startswith('emb_') and not c.startswith('pca_')]
X_struct = X[struct_only_cols]
print(f"  SHAP uses {len(struct_only_cols)} structured feature columns")

X_tr_s, X_te_s, y_tr_s, _ = train_test_split(
    X_struct, y, test_size=0.2, random_state=42, stratify=y)

# Rebuild best model instance for structured-only retraining
model_map = {
    'LightGBM':            lgb.LGBMClassifier(random_state=42, n_jobs=-1, verbose=-1),
    'Random Forest':       RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    'CART Decision Tree':  DecisionTreeClassifier(random_state=42),
    'Lasso (L1)':          LogisticRegression(penalty='l1', solver='saga', max_iter=2000, random_state=42, n_jobs=-1),
    'Ridge (L2)':          LogisticRegression(penalty='l2', max_iter=1000, random_state=42, n_jobs=-1),
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1),
}
bm_struct = model_map[best_model_name]
bm_struct.fit(X_tr_s, y_tr_s)
struct_model_path = f'models/best_struct_{safe_name(best_model_name)}.joblib'
joblib.dump(bm_struct, struct_model_path)
print(f"  Saved {struct_model_path}")

# Sample for speed
shap_n = min(500, len(X_te_s))
X_shap = X_te_s.sample(shap_n, random_state=42)
print(f"  Computing SHAP on {shap_n} samples…")

# Build explainer
is_tree = best_model_name in ['LightGBM', 'Random Forest', 'CART Decision Tree']
if is_tree:
    explainer = shap.TreeExplainer(bm_struct)
    sv_raw = explainer.shap_values(X_shap)
else:
    explainer = shap.LinearExplainer(bm_struct, X_tr_s)
    sv_raw = explainer.shap_values(X_shap)

# Normalize to 2D for class-0 (Model A wins) visualizations
if isinstance(sv_raw, list):
    sv_2d   = sv_raw[0]
    exp_val = (explainer.expected_value[0]
               if hasattr(explainer.expected_value, '__len__')
               else float(explainer.expected_value))
elif hasattr(sv_raw, 'ndim') and sv_raw.ndim == 3:
    sv_2d   = sv_raw[:, :, 0]
    exp_val = (float(explainer.expected_value[0])
               if hasattr(explainer.expected_value, '__len__')
               else float(explainer.expected_value))
else:
    sv_2d   = sv_raw
    exp_val = float(explainer.expected_value)

print("  SHAP values computed")

# ── SHAP Plot 1: Beeswarm ────────────────────────────────────
print("  Generating beeswarm plot…")
plt.figure(figsize=(11, 8))
shap.summary_plot(sv_2d, X_shap, show=False, max_display=20)
plt.title('SHAP Summary (Beeswarm) — Class: Model A Wins',
          fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('shap_plots/01_shap_beeswarm.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved shap_plots/01_shap_beeswarm.png")

# ── SHAP Plot 2: Bar (mean |SHAP|) ───────────────────────────
print("  Generating bar plot…")
plt.figure(figsize=(11, 7))
shap.summary_plot(sv_2d, X_shap, plot_type='bar', show=False, max_display=20)
plt.title('SHAP Feature Importance (Mean |SHAP|) — Class: Model A Wins',
          fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('shap_plots/02_shap_bar.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved shap_plots/02_shap_bar.png")

# ── SHAP Plot 3: Waterfall for first test sample ─────────────
print("  Generating waterfall plot…")
try:
    shap_exp = shap.Explanation(
        values=sv_2d[0],
        base_values=exp_val,
        data=X_shap.iloc[0].values,
        feature_names=list(X_shap.columns),
    )
    plt.figure(figsize=(12, 6))
    shap.waterfall_plot(shap_exp, max_display=15, show=False)
    plt.title('SHAP Waterfall — First Test Sample (Class: Model A Wins)',
              fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig('shap_plots/03_shap_waterfall.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved shap_plots/03_shap_waterfall.png")
except Exception as e:
    print(f"  Waterfall fallback ({e})…")
    fi = (pd.Series(np.abs(sv_2d).mean(axis=0), index=X_shap.columns)
            .sort_values())
    fig, ax = plt.subplots(figsize=(11, 6))
    fi.tail(15).plot(kind='barh', ax=ax, color='#4C72B0')
    ax.set_title('Top 15 Features by Mean |SHAP| — Class: Model A Wins',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig('shap_plots/03_shap_waterfall.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved shap_plots/03_shap_waterfall.png (fallback bar chart)")

# Save SHAP insights for Streamlit
top_feat = (pd.Series(np.abs(sv_2d).mean(axis=0), index=X_shap.columns)
              .idxmax())
joblib.dump({'top_feature': top_feat, 'best_model': best_model_name},
            'models/shap_insights.joblib')
print(f"  Top SHAP feature: {top_feat}")

print("\nSHAP complete")

print("\n" + "=" * 60)
print("✓ PIPELINE COMPLETE — Steps 1–4 finished successfully!")
print("  Run:  streamlit run app.py")
print("=" * 60)
