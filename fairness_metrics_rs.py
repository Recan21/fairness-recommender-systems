"""
fairness_metrics_rs.py  -  MovieLens 1M
Bachelor Thesis  -  Fairness Metrics for Recommender Systems

Laedt Ergebnisse aus simulation.py (results/simulation_results.pkl)
und berechnet alle Fairness-Metriken auf den Empfehlungen.

VORAUSSETZUNG: simulation.py muss zuerst gelaufen sein.

PROVIDER-SIDE:
  Gini, Gini_our, Entropy (Entour), APLT, Jain, FSat, MinSkew, MaxSkew, Coverage

GROUP FAIRNESS:
  Hit Rate Gap (Gender), Hit Rate Gap (Age max), nDCG Gap (Gender)

INDIVIDUAL FAIRNESS (CF only, aus Pickle):
  U_val, U_over, U_under  (Gender + Age max pairwise)

OUTPUTS:
  Bestehend (Popularity + CF):
    plots/fairness_rs/    results/rs/
  Neu (alle 5 RS):
    plots/fairness_rs_extended/final/    results/rs_extended/
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from collections import defaultdict

# =============================================================================
# 0. CONFIG
# =============================================================================
DATA_DIR       = "ml-1m/ml-1m"
OUTPUT_DIR     = "plots/fairness_rs"
OUTPUT_EXT     = "plots/fairness_rs_extended"
OUTPUT_EXT_FIN = os.path.join(OUTPUT_EXT, "final")
CSV_DIR        = "results/rs"
CSV_EXT        = "results/rs_extended"
PICKLE_PATH    = "results/simulation_results.pkl"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_EXT, exist_ok=True)
os.makedirs(OUTPUT_EXT_FIN, exist_ok=True)
os.makedirs(CSV_DIR, exist_ok=True)
os.makedirs(CSV_EXT, exist_ok=True)

TOP_N          = 10
NDCG_THRESHOLD = 4.0
AGE_ORDER      = ["Under 18", "18-24", "25-34", "35-44", "45-49", "50-55", "56+"]
DPI            = 150
AGE_MAP = {1:"Under 18",18:"18-24",25:"25-34",35:"35-44",
           45:"45-49",50:"50-55",56:"56+"}

# Farben und Stile pro RS (NextItem_k1 entfernt)
RS_STYLES = {
    "Popularity":   {"color": "#E07B39", "marker": "o", "ls": "-",  "lw": 2.2},
    "CF":           {"color": "#2166AC", "marker": "s", "ls": "-",  "lw": 2.2},
    "Random":       {"color": "#2ca02c", "marker": "^", "ls": "--", "lw": 1.8},
    "FairnessExp":  {"color": "#9467bd", "marker": "D", "ls": "--", "lw": 1.8},
    "NextItem_k10": {"color": "#8c564b", "marker": "P", "ls": ":",  "lw": 1.8},
}

# =============================================================================
# 1. DATEN LADEN
# =============================================================================
print("Loading data ...")

users = pd.read_csv(
    os.path.join(DATA_DIR, "users.dat"), sep="::", engine="python",
    names=["UserID","Gender","Age","Occupation","Zip"], dtype={"Zip":str})
ratings = pd.read_csv(
    os.path.join(DATA_DIR, "ratings.dat"), sep="::", engine="python",
    names=["UserID","MovieID","Rating","Timestamp"])

users["AgeLabel"] = users["Age"].map(AGE_MAP)
ratings["Date"]   = pd.to_datetime(ratings["Timestamp"], unit="s")

df = ratings.merge(users[["UserID","Gender","AgeLabel"]], on="UserID", how="left")
df = df.sort_values("Timestamp").reset_index(drop=True)

total_ratings = len(df)
N_SPLITS      = 10
split_size    = total_ratings // N_SPLITS
df["Bucket"]  = np.clip(np.arange(len(df)) // split_size, 0, N_SPLITS - 1) + 1

user_gender_dict = users.set_index("UserID")["Gender"].to_dict()
user_age_dict    = users.set_index("UserID")["AgeLabel"].to_dict()

print(f"  {total_ratings:,} ratings geladen")

# =============================================================================
# 2. PICKLE LADEN
# =============================================================================
print(f"\nLoading simulation results from {PICKLE_PATH} ...")

if not os.path.exists(PICKLE_PATH):
    raise FileNotFoundError(
        f"{PICKLE_PATH} nicht gefunden.\n"
        "Bitte zuerst simulation.py laufen lassen!"
    )

with open(PICKLE_PATH, "rb") as f:
    saved = pickle.load(f)

simulation_results = saved["simulation_results"]
all_movie_ids      = saved["all_movie_ids"]
pct_labels         = saved["pct_labels"]
n_all_films        = len(all_movie_ids)

print(f"  {len(simulation_results)} Splits  |  {n_all_films} Filme\n")

# =============================================================================
# 3. FAIRNESS-METRIKEN FUNKTIONEN
# =============================================================================

# --- Provider-Side ---

def gini(arr):
    """
    Gini Koeffizient der Empfehlungsverteilung.
    j = Rank des Films (j=1=wenigste, j=n=meiste Empfehlungen)
    Ex_j = Anzahl Empfehlungen Film j
    0 = fair, 1 = unfair.
    """
    arr = np.sort(arr.astype(float))
    n, total = len(arr), arr.sum()
    if n == 0 or total == 0: return np.nan
    j = np.arange(1, n + 1)
    return float(np.sum((2*j - n - 1) * arr) / (n * total))


def gini_our(arr, n_all_films, n_users):
    """
    Korrigierter Gini (Rampisela et al. 2024, Eq. 16-17).
    Normalisiert auf [0,1]: 0 = most fair, 1 = most unfair @k.
    Ginimax = 1 - k/n
    Ginimin = (n - km mod n)(km mod n) / (km x n)
    """
    km       = n_users * TOP_N
    gini_max = 1.0 - TOP_N / n_all_films
    gini_min = ((n_all_films - km % n_all_films) * (km % n_all_films)
                / (km * n_all_films))
    g = gini(arr)
    if np.isnan(g) or (gini_max - gini_min) == 0: return np.nan
    return float(np.clip((g - gini_min) / (gini_max - gini_min), 0, 1))


def entropy_rs(arr):
    """
    Entropy (Entour): nur empfohlene Filme (vermeidet log(0)).
    p(i) = rec_i / sum rec_j
    Log-Basis = n -> Bereich [0,1]. 0=unfair, 1=fair.
    """
    counts  = arr[arr > 0]
    n_total = len(arr)
    if len(counts) == 0 or n_total <= 1: return np.nan
    p = counts / counts.sum()
    return float(-np.sum(p * np.log(p)) / np.log(n_total))


def aplt(arr):
    """
    Adapted APLT: Anteil Empfehlungen auf Long-Tail-Filme.
    Gamma = Filme mit rec_i < Durchschnitt (Long-Tail).
    APLT = sum_{i in Gamma} rec_i / sum_{j in I} rec_j
    0=unfair, hoeher=fairer.
    """
    total = arr.sum()
    if total == 0: return np.nan
    long_tail = arr[arr < arr.mean()]
    return float(long_tail.sum() / total)


def jain(arr, n_all_films, km):
    """
    Jain's Index.
    Jain = (km)^2 / (n x sum_i rec_i^2)
    0=unfair, 1=fair.
    km = n_users x TOP_N (60,400)
    """
    if km == 0: return np.nan
    sum_sq = np.sum(arr ** 2)
    if sum_sq == 0: return np.nan
    return float((km ** 2) / (n_all_films * sum_sq))


def fsat(arr, n_all_films, km):
    """
    Fraction of Satisfied Items.
    Film 'satisfied' wenn rec_i >= floor(km/n) (Maximin Share).
    FSat = (1/n) sum delta(rec_i >= floor(km/n))
    0=unfair, 1=fair.
    """
    maximin = int(km / n_all_films)
    return float((arr >= maximin).sum() / n_all_films)


def skews(arr, n_all_films, epsilon=1e-10):
    """
    MinSkew und MaxSkew.
    p_f(v) = 1/n  (faire Verteilung)
    p(v)   = rec_v / sum rec_j  (Laplace geglättet)
    Skew(v) = log(p_f(v) / p(v))
      = 0  -> faire Share
      < 0  -> mehr als fair (bevorzugt)
      > 0  -> weniger als fair (benachteiligt)
    MinSkew = min aller Skews (meistbevorzugter Film)
    MaxSkew = max aller Skews (meistbenachteiligter Film)
    """
    km = arr.sum()
    if km == 0: return np.nan, np.nan
    p_f      = 1.0 / n_all_films
    p_smooth = (arr + epsilon) / (km + n_all_films * epsilon)
    skew     = np.log(p_f / p_smooth)
    return float(skew.min()), float(skew.max())


def coverage(arr, n_all_films):
    """Coverage (QF): Anteil Filme die mindestens 1x empfohlen wurden."""
    return float((arr > 0).sum() / n_all_films)


# --- Group Fairness ---

def _hr_by_group(recs, future_df, attr_dict, denom=TOP_N):
    """
    Hit Rate pro Gruppe.
    denom = TOP_N fuer standard RS; "actual" fuer Next-Item (variable Listenlänge).
    """
    fut    = future_df.groupby("UserID")["MovieID"].apply(set).to_dict()
    g_hits = defaultdict(list)
    for uid, rec_list in recs.items():
        if not rec_list: continue
        g = attr_dict.get(uid)
        if g is None: continue
        hits = len(set(rec_list) & fut.get(uid, set()))
        d    = len(rec_list) if denom == "actual" else denom
        g_hits[g].append(hits / d)
    return {g: float(np.mean(v)) for g, v in g_hits.items()}


def hr_gap_gender(recs, future_df, denom=TOP_N):
    """|HR_male - HR_female|. 0=fair."""
    by_g = _hr_by_group(recs, future_df, user_gender_dict, denom)
    m, f = by_g.get('M', np.nan), by_g.get('F', np.nan)
    return float(abs(m - f)) if not (np.isnan(m) or np.isnan(f)) else np.nan


def hr_gap_age_max(recs, future_df, denom=TOP_N):
    """Max pairwise |HR_A - HR_B|. 0=fair."""
    by_g  = _hr_by_group(recs, future_df, user_age_dict, denom)
    valid = {k: v for k, v in by_g.items() if not np.isnan(v)}
    if len(valid) < 2: return np.nan
    grps = list(valid.keys())
    return float(max(abs(valid[a]-valid[b])
                     for i,a in enumerate(grps) for b in grps[i+1:]))


def _ndcg_by_group(recs, future_df, attr_dict, threshold):
    """nDCG@TOP_N pro Gruppe. Relevanz = Rating >= threshold."""
    fut_r = (future_df.groupby("UserID")
             .apply(lambda x: dict(zip(x["MovieID"], x["Rating"]))).to_dict())
    g_ndcg = defaultdict(list)
    for uid, rec_list in recs.items():
        if not rec_list: continue
        g = attr_dict.get(uid)
        if g is None: continue
        uf = fut_r.get(uid, {})
        dcg = sum(1.0/np.log2(p+2) for p,m in enumerate(rec_list)
                  if uf.get(m, 0) >= threshold)
        n_rel = min(sum(1 for rv in uf.values() if rv >= threshold), len(rec_list))
        idcg  = sum(1.0/np.log2(i+2) for i in range(n_rel))
        g_ndcg[g].append(dcg/idcg if idcg > 0 else 0.0)
    return {g: float(np.mean(v)) for g, v in g_ndcg.items()}


def ndcg_gap_gender(recs, future_df):
    """|nDCG_male - nDCG_female|. 0=fair."""
    by_g = _ndcg_by_group(recs, future_df, user_gender_dict, NDCG_THRESHOLD)
    m, f = by_g.get('M', np.nan), by_g.get('F', np.nan)
    return float(abs(m - f)) if not (np.isnan(m) or np.isnan(f)) else np.nan


def compute_hit_rate_at_1(recs, future_df):
    """
    Hit Rate@1: Trefferquote wenn nur die erste Empfehlung pro User gezählt wird.
    Misst wie gut der best-ranked Vorschlag jedes RS tatsächlich zutrifft.
    """
    future_per_user = (future_df.groupby("UserID")["MovieID"]
                                .apply(set).to_dict())
    hits = []
    for uid, rec_list in recs.items():
        if not rec_list: continue
        future_movies = future_per_user.get(uid, set())
        hits.append(1.0 if rec_list[0] in future_movies else 0.0)
    return float(np.mean(hits)) if hits else np.nan


# =============================================================================
# 4. METRIKEN BERECHNEN — BESTEHEND (Popularity + CF)
# =============================================================================
print(f"Computing fairness metrics for {len(simulation_results)} splits ...\n")

rows_all = []

for r in simulation_results:
    t         = r["t"]
    pct_label = r["split"]
    n_users   = r["n_users"]
    km        = n_users * TOP_N

    pop_arr = r["pop_counts"].values.astype(float)
    cf_arr  = r["cf_counts"].values.astype(float)

    # Nur Filme berücksichtigen die bis Split t bereits im System sind
    # (mindestens 1 echtes Rating haben) → fairer Vergleich pro Split
    # Filme die noch nicht existieren sollen nicht als "nicht empfohlen" bestraft werden
    active_movie_ids = set(df[df["Bucket"] <= t]["MovieID"].unique())
    active_mask      = np.array([mid in active_movie_ids for mid in all_movie_ids])
    n_active         = int(active_mask.sum())

    pop_arr_act = pop_arr[active_mask]
    cf_arr_act  = cf_arr[active_mask]

    # Provider-Side — n_active statt n_all_films (3706)
    def _prov(arr):
        g, g_ou = gini(arr), gini_our(arr, n_active, n_users)
        ent     = entropy_rs(arr)
        ap      = aplt(arr)
        ja      = jain(arr, n_active, km)
        fs      = fsat(arr, n_active, km)
        mn, mx  = skews(arr, n_active)
        cov     = coverage(arr, n_active)
        return g, g_ou, ent, ap, ja, fs, mn, mx, cov

    pop_g, pop_gou, pop_ent, pop_ap, pop_ja, pop_fs, pop_mn, pop_mx, pop_cov = _prov(pop_arr_act)
    cf_g,  cf_gou,  cf_ent,  cf_ap,  cf_ja,  cf_fs,  cf_mn,  cf_mx,  cf_cov  = _prov(cf_arr_act)

    print(f"  t={t:2d}  {pct_label}  |  "
          f"Active films: {n_active}/{n_all_films}  |  "
          f"Gini Pop={pop_g:.3f}  CF={cf_g:.3f}  |  Ent CF={cf_ent:.3f}")

    if t < N_SPLITS:
        future_df = df[df["Bucket"] > t]
        pop_hr_g = hr_gap_gender(r["pop_recs"], future_df)
        cf_hr_g  = hr_gap_gender(r["cf_recs"],  future_df)
        pop_hr_a = hr_gap_age_max(r["pop_recs"], future_df)
        cf_hr_a  = hr_gap_age_max(r["cf_recs"],  future_df)
        pop_ndcg = ndcg_gap_gender(r["pop_recs"], future_df)
        cf_ndcg  = ndcg_gap_gender(r["cf_recs"],  future_df)
    else:
        pop_hr_g = cf_hr_g = pop_hr_a = cf_hr_a = pop_ndcg = cf_ndcg = np.nan

    base = dict(split=pct_label, t=t, n_users=n_users, km=km,
                n_active_films=n_active,
                hit_rate_pop=r["pop_hit_rate"], hit_rate_cf=r["cf_hit_rate"])

    rows_all.append({**base, "rs": "Popularity",
        "gini": pop_g, "gini_our": pop_gou, "entropy": pop_ent,
        "aplt": pop_ap, "jain": pop_ja, "fsat": pop_fs,
        "minskew": pop_mn, "maxskew": pop_mx, "coverage": pop_cov,
        "hr_gap_gender": pop_hr_g, "hr_gap_age": pop_hr_a, "ndcg_gap_gender": pop_ndcg,
        "u_val_gender":  np.nan, "u_over_gender":  np.nan, "u_under_gender":  np.nan,
        "u_val_age":     np.nan, "u_over_age":     np.nan, "u_under_age":     np.nan,
    })

    rows_all.append({**base, "rs": "CF",
        "gini": cf_g, "gini_our": cf_gou, "entropy": cf_ent,
        "aplt": cf_ap, "jain": cf_ja, "fsat": cf_fs,
        "minskew": cf_mn, "maxskew": cf_mx, "coverage": cf_cov,
        "hr_gap_gender": cf_hr_g, "hr_gap_age": cf_hr_a, "ndcg_gap_gender": cf_ndcg,
        "u_val_gender":  r.get("u_val_gender",   np.nan),
        "u_over_gender": r.get("u_over_gender",  np.nan),
        "u_under_gender":r.get("u_under_gender", np.nan),
        "u_val_age":     r.get("u_val_age",      np.nan),
        "u_over_age":    r.get("u_over_age",     np.nan),
        "u_under_age":   r.get("u_under_age",    np.nan),
    })

result     = pd.DataFrame(rows_all)
pop_result = result[result["rs"] == "Popularity"].reset_index(drop=True)
cf_result  = result[result["rs"] == "CF"].reset_index(drop=True)

# =============================================================================
# 5. CSV EXPORT + KORRELATIONEN (bestehend)
# =============================================================================
result.to_csv(    os.path.join(CSV_DIR, "fairness_rs_all.csv"),        index=False)
pop_result.to_csv(os.path.join(CSV_DIR, "fairness_rs_popularity.csv"), index=False)
cf_result.to_csv( os.path.join(CSV_DIR, "fairness_rs_cf.csv"),         index=False)

metric_cols_provider = ["gini", "gini_our", "entropy", "aplt", "jain", "fsat",
                        "minskew", "maxskew", "coverage"]
metric_cols_group    = ["hr_gap_gender", "hr_gap_age", "ndcg_gap_gender"]
metric_cols_cf_only  = ["u_val_gender", "u_over_gender", "u_under_gender",
                        "u_val_age",    "u_over_age",    "u_under_age"]

cf_metric_cols  = metric_cols_provider + metric_cols_group + metric_cols_cf_only
cf_corr_full    = cf_result[cf_metric_cols].corr().round(4)
mask_cf         = np.triu(np.ones(cf_corr_full.shape, dtype=bool), k=1)
cf_corr_lt      = cf_corr_full.mask(mask_cf)

pop_metric_cols = metric_cols_provider + metric_cols_group
pop_corr_full   = pop_result[pop_metric_cols].corr().round(4)
mask_pop        = np.triu(np.ones(pop_corr_full.shape, dtype=bool), k=1)
pop_corr_lt     = pop_corr_full.mask(mask_pop)

cf_corr_lt.to_csv( os.path.join(CSV_DIR, "fairness_rs_correlations_cf.csv"))
pop_corr_lt.to_csv(os.path.join(CSV_DIR, "fairness_rs_correlations_popularity.csv"))

print(f"\nCSVs -> {CSV_DIR}/")
print(f"  fairness_rs_all.csv, fairness_rs_popularity.csv, fairness_rs_cf.csv")
print(f"  fairness_rs_correlations_cf.csv, fairness_rs_correlations_popularity.csv")

# =============================================================================
# 6. PLOTTING HELPERS (SHARED)
# =============================================================================
x_pos = list(range(N_SPLITS))
C_POP, C_CF = "#E07B39", "#2166AC"

def _style_single(ax, title, ylabel, note=""):
    """Styling fuer einzelne, eigenstaendige Plots."""
    ax.set_title(title, fontsize=12, fontweight="bold", loc="left", pad=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_xlabel("Cumulative Data Split (time-ordered)", fontsize=10)
    ax.set_xticks(x_pos); ax.set_xticklabels(pct_labels, fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.3, linewidth=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    # extra Platz oben/unten, damit Legende und Notiz nicht an den Rand stossen
    ymin, ymax = ax.get_ylim()
    pad = (ymax - ymin) * 0.12
    ax.set_ylim(ymin - pad, ymax + pad)
    if note:
        ax.text(0.98, 0.03, note, transform=ax.transAxes,
                fontsize=8.5, color="gray", ha="right", va="bottom",
                style="italic", zorder=5,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor="none", alpha=0.75))

def _legend_inside(ax, loc="best", ncol=1):
    """Legende innerhalb der Plot-Flaeche, mit Hintergrund gegen Ueberlappung."""
    ax.legend(loc=loc, fontsize=9, framealpha=0.85, ncol=ncol,
              facecolor="white", edgecolor="none")

def _pair_single(col, title, ylabel, note, filename, out_dir):
    """Einzelplot: Popularity vs. CF fuer eine Metrik."""
    fig, ax = plt.subplots(figsize=(8, 5))
    y_pop = pop_result[col].values
    y_cf  = cf_result[col].values
    ax.plot(x_pos, y_pop, color=C_POP, linewidth=2.2, marker="o", markersize=5,
            label="Popularity", zorder=3)
    ax.plot(x_pos, y_cf, color=C_CF, linewidth=2.2, marker="s", markersize=5,
            label="CF (SVD)", zorder=3)
    ax.fill_between(x_pos, y_pop, alpha=0.07, color=C_POP)
    ax.fill_between(x_pos, y_cf, alpha=0.07, color=C_CF)
    _style_single(ax, title, ylabel, note)
    _legend_inside(ax)
    plt.tight_layout()
    path = os.path.join(out_dir, filename)
    plt.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"  Plot -> {path}")
    return path

def _multi_single(col, title, ylabel, note, filename, out_dir,
                   skip_ni=False, skip_rs=None):
    """Einzelplot: alle RS fuer eine Metrik."""
    skip_rs = skip_rs or []
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for rs_name, rs_df in ALL_RS:
        if skip_ni and rs_name.startswith("NextItem"): continue
        if rs_name in skip_rs: continue
        if col not in rs_df.columns: continue
        y  = rs_df[col].values
        st = RS_STYLES[rs_name]
        ax.plot(x_pos, y, color=st["color"], linewidth=st["lw"],
                marker=st["marker"], markersize=5, linestyle=st["ls"],
                label=rs_name, zorder=3, alpha=0.9)
    _style_single(ax, title, ylabel, note)
    _legend_inside(ax)
    plt.tight_layout()
    path = os.path.join(out_dir, filename)
    plt.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"  Plot -> {path}")
    return path

def _individual_pair(col_gender, col_age, title, ylabel, filename, out_dir, source_df):
    """Gepaarter Plot: Gender (links) | Age max (rechts), CF only."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, col, sub in zip(axes, [col_gender, col_age], ["Gender", "Age (max pairwise)"]):
        y_cf = source_df[col].values
        ax.plot(x_pos, y_cf, color=C_CF, linewidth=2.2, marker="s", markersize=5,
                label="CF (SVD)", zorder=3)
        ax.fill_between(x_pos, y_cf, alpha=0.10, color=C_CF)
        _style_single(ax, f"{title} ({sub})", ylabel, "0 = fair")
        _legend_inside(ax)
    plt.tight_layout()
    path = os.path.join(out_dir, filename)
    plt.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"  Plot -> {path}")
    return path


# =============================================================================
# 6a. BESTEHENDE PLOTS (Popularity + CF)
# =============================================================================
print("\nGenerating plots (Popularity + CF) ...")

# --- Gini, Entropy, APLT, Jain (4 einzelne Dateien) -------------------------
_pair_single("gini",    "Gini",                  "Gini [0,1]",
              "0 = fair, 1 = unfair", "1a_gini.png", OUTPUT_DIR)
_pair_single("entropy", "Entropy (Entour)",      "Entropy [0,1]",
              "0 = unfair, 1 = fair", "1b_entropy.png", OUTPUT_DIR)
_pair_single("aplt",    "APLT: Long-Tail Share", "APLT [0,1]",
              "higher = fairer", "1c_aplt.png", OUTPUT_DIR)
_pair_single("jain",    "Jain's Index",          "Jain [0,1]",
              "0 = unfair, 1 = fair", "1d_jain.png", OUTPUT_DIR)

# --- FSat, Coverage, MinSkew, MaxSkew (4 einzelne Dateien) ------------------
_pair_single("fsat",     "FSat: Satisfied Items",        "FSat [0,1]",
              "0 = unfair, 1 = fair", "2a_fsat.png", OUTPUT_DIR)
_pair_single("coverage", "Coverage (QF)",                "Coverage [0,1]",
              "diversity metric", "2b_coverage.png", OUTPUT_DIR)
_pair_single("minskew",  "MinSkew (most favoured film)", "MinSkew (log)",
              "< 0 = more than fair share", "2c_minskew.png", OUTPUT_DIR)
_pair_single("maxskew",  "MaxSkew (most disadvantaged)", "MaxSkew (log)",
              "> 0 = less than fair share", "2d_maxskew.png", OUTPUT_DIR)

# --- Hit Rate + Group Fairness (3 einzelne Dateien) -------------------------
def _hitrate_single(filename, out_dir):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x_pos, pop_result["hit_rate_pop"].values, color=C_POP, linewidth=2.2,
            marker="o", markersize=5, label="Popularity", zorder=3)
    ax.plot(x_pos, cf_result["hit_rate_cf"].values, color=C_CF, linewidth=2.2,
            marker="s", markersize=5, label="CF (SVD)", zorder=3)
    ax.fill_between(x_pos, pop_result["hit_rate_pop"].values, alpha=0.07, color=C_POP)
    ax.fill_between(x_pos, cf_result["hit_rate_cf"].values, alpha=0.07, color=C_CF)
    _style_single(ax, "Hit Rate", "Hit Rate [0,1]", "higher = better")
    _legend_inside(ax)
    plt.tight_layout()
    path = os.path.join(out_dir, filename)
    plt.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"  Plot -> {path}")
    return path

_hitrate_single("4a_hit_rate.png", OUTPUT_DIR)
_pair_single("hr_gap_gender", "Hit Rate Gap (Gender)",  "Gap [0,1]",
              "0 = fair", "4b_hr_gap_gender.png", OUTPUT_DIR)
_pair_single("hr_gap_age",    "Hit Rate Gap (Age Max)", "Gap [0,1]",
              "0 = fair", "4c_hr_gap_age.png", OUTPUT_DIR)

# --- nDCG Gap ----------------------------------------------------------------
_pair_single("ndcg_gap_gender", "nDCG Gap (Gender)", "nDCG Gap [0,1]",
              f"0 = fair | relevance threshold = {NDCG_THRESHOLD} stars",
              "5_ndcg_gap_gender.png", OUTPUT_DIR)

# --- Individual Fairness (CF only), gepaart Gender | Age --------------------
_individual_pair("u_val_gender",   "u_val_age",   "Value Unfairness",
                 "U_val [>=0]",  "6a_value_unfairness.png", OUTPUT_DIR, cf_result)
_individual_pair("u_over_gender",  "u_over_age",  "Overestimation Unfairness",
                 "U_over [>=0]", "6b_overestimation_unfairness.png", OUTPUT_DIR, cf_result)
_individual_pair("u_under_gender", "u_under_age", "Underestimation Unfairness",
                 "U_under [>=0]","6c_underestimation_unfairness.png", OUTPUT_DIR, cf_result)

# --- Summary Table (letzter Split) -------------------------------------------
last_pop = pop_result.iloc[-1]
last_cf  = cf_result.iloc[-1]
summary = [
    ("Gini",              "gini",           "down lower=fairer"),
    ("Gini_our",          "gini_our",       "down lower=fairer"),
    ("Entropy (Entour)",  "entropy",        "up higher=fairer"),
    ("APLT",              "aplt",           "up higher=fairer"),
    ("Jain's Index",      "jain",           "up higher=fairer"),
    ("FSat",              "fsat",           "up higher=fairer"),
    ("MinSkew",           "minskew",        "0=fairest"),
    ("MaxSkew",           "maxskew",        "0=fairest"),
    ("Coverage (QF)",     "coverage",       "diversity metric"),
    ("HR Gap Gender",     "hr_gap_gender",  "down 0=fair"),
    ("HR Gap Age",        "hr_gap_age",     "down 0=fair"),
    ("nDCG Gap Gender",   "ndcg_gap_gender","down 0=fair"),
    ("U_val Gender",      "u_val_gender",   "down 0=fair  CF only"),
    ("U_over Gender",     "u_over_gender",  "down 0=fair  CF only"),
    ("U_under Gender",    "u_under_gender", "down 0=fair  CF only"),
    ("U_val Age",         "u_val_age",      "down 0=fair  CF only"),
    ("U_over Age",        "u_over_age",     "down 0=fair  CF only"),
    ("U_under Age",       "u_under_age",    "down 0=fair  CF only"),
]
table_data = [[m,
               f"{last_pop[col]:.4f}" if not pd.isna(last_pop[col]) else "N/A",
               f"{last_cf[col]:.4f}"  if not pd.isna(last_cf[col])  else "N/A",
               note]
              for m, col, note in summary]

fig, ax = plt.subplots(figsize=(11, len(summary)*0.46+2))
ax.axis("off")
table = ax.table(cellText=table_data,
                 colLabels=["Metric","Popularity","CF (SVD)","Interpretation"],
                 cellLoc="center", loc="center",
                 colWidths=[0.32,0.16,0.16,0.36])
table.auto_set_font_size(False); table.set_fontsize(9); table.scale(1, 1.4)
for j in range(4):
    table[(0,j)].set_facecolor("#2166AC")
    table[(0,j)].set_text_props(color="white", fontweight="bold")
for i in range(1, len(summary)+1):
    bg = "#f0f4f8" if i%2==0 else "white"
    for j in range(4): table[(i,j)].set_facecolor(bg)
ax.set_title("Fairness Summary — Split 10 (100% of data)  |  MovieLens 1M",
             fontsize=11, fontweight="bold", pad=20)
plt.tight_layout()
p7 = os.path.join(OUTPUT_DIR, "7_summary_table_split10.png")
plt.savefig(p7, dpi=DPI, bbox_inches="tight"); plt.close(); print(f"  Plot -> {p7}")

# --- Korrelationsmatrizen -----------------------------------------------------
def _draw_corr_matrix(ax, corr_full, cols, title):
    """Zeichnet eine Korrelationsmatrix als Heatmap (unteres Dreieck)."""
    vals   = corr_full.values.copy()
    mask   = np.triu(np.ones_like(vals, dtype=bool), k=1)
    disp   = np.where(mask, np.nan, vals)
    im     = ax.imshow(disp, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    labels = [c.replace("_", " ") for c in cols]
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=7.5)
    ax.set_yticks(range(len(cols)))
    ax.set_yticklabels(labels, fontsize=7.5)
    for i in range(len(cols)):
        for j in range(len(cols)):
            if j > i:
                ax.add_patch(plt.Rectangle((j-0.5, i-0.5), 1, 1,
                                           color="white", zorder=2))
                continue
            val   = corr_full.values[i, j]
            color = "white" if abs(val) > 0.6 else "#222"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=7, color=color, fontweight="bold")
    plt.colorbar(im, ax=ax, label="Pearson r", shrink=0.8, pad=0.02)
    ax.set_title(title, fontsize=10, fontweight="bold", loc="left", pad=8)

print("\nGenerating correlation plots ...")

fig, ax = plt.subplots(figsize=(13, 11))
_draw_corr_matrix(ax, cf_corr_full, cf_metric_cols,
                  "Correlation Matrix — CF (SVD)  |  MovieLens 1M\n"
                  "Provider-Side + Group Fairness + Individual Fairness\n"
                  "Red = positive  |  Blue = negative  |  Lower triangle only")
plt.tight_layout()
p8 = os.path.join(OUTPUT_DIR, "8_correlation_matrix_cf.png")
plt.savefig(p8, dpi=DPI, bbox_inches="tight"); plt.close(); print(f"  Plot -> {p8}")

fig, ax = plt.subplots(figsize=(11, 9))
_draw_corr_matrix(ax, pop_corr_full, pop_metric_cols,
                  "Correlation Matrix — Popularity  |  MovieLens 1M\n"
                  "Provider-Side + Group Fairness\n"
                  "Red = positive  |  Blue = negative  |  Lower triangle only")
plt.tight_layout()
p9 = os.path.join(OUTPUT_DIR, "9_correlation_matrix_popularity.png")
plt.savefig(p9, dpi=DPI, bbox_inches="tight"); plt.close(); print(f"  Plot -> {p9}")


# =============================================================================
# 7. METRIKEN BERECHNEN — ERWEITERT (5 RS, ohne NextItem_k1)
# =============================================================================
print(f"\nComputing extended metrics (5 RS) ...")

rows_ext = []

for r in simulation_results:
    t         = r["t"]
    pct_label = r["split"]
    n_users   = r["n_users"]
    km_std    = n_users * TOP_N

    active_movie_ids = set(df[df["Bucket"] <= t]["MovieID"].unique())
    active_mask      = np.array([mid in active_movie_ids for mid in all_movie_ids])
    n_active         = int(active_mask.sum())

    def _get_arr_act(key):
        return r[key].values.astype(float)[active_mask]

    pop_arr  = _get_arr_act("pop_counts")
    cf_arr   = _get_arr_act("cf_counts")
    rand_arr = _get_arr_act("rand_counts") if "rand_counts" in r else np.zeros(n_active)
    fair_arr = _get_arr_act("fair_counts") if "fair_counts" in r else np.zeros(n_active)
    ni10_arr = _get_arr_act("ni10_counts") if "ni10_counts" in r else np.zeros(n_active)

    def _prov_km(arr, km):
        g    = gini(arr)
        g_ou = gini_our(arr, n_active, n_users)
        ent  = entropy_rs(arr)
        ap   = aplt(arr)
        ja   = jain(arr, n_active, km)
        fs   = fsat(arr, n_active, km)
        mn, mx = skews(arr, n_active)
        cov  = coverage(arr, n_active)
        return g, g_ou, ent, ap, ja, fs, mn, mx, cov

    prov_pop  = _prov_km(pop_arr,  km_std)
    prov_cf   = _prov_km(cf_arr,   km_std)
    prov_rand = _prov_km(rand_arr, km_std)

    # FairnessExp: APLT = NaN (near-uniform distribution macht APLT instabil)
    prov_fair = list(_prov_km(fair_arr, km_std))
    prov_fair[3] = np.nan  # index 3 = aplt
    prov_fair = tuple(prov_fair)

    # NextItem_k10: wenn sum=0 (t=10, kein Testset) → alle Metriken NaN
    ni10_sum = int(ni10_arr.sum())
    if ni10_sum == 0:
        prov_ni10 = tuple([np.nan] * 9)
    else:
        prov_ni10 = _prov_km(ni10_arr, ni10_sum)

    if t < N_SPLITS:
        future_df = df[df["Bucket"] > t]

        pop_hrg  = hr_gap_gender(r["pop_recs"],  future_df)
        cf_hrg   = hr_gap_gender(r["cf_recs"],   future_df)
        rand_hrg = hr_gap_gender(r.get("rand_recs", {}), future_df)
        fair_hrg = hr_gap_gender(r.get("fair_recs", {}), future_df)
        ni10_hrg = hr_gap_gender(r.get("ni10_recs", {}), future_df, denom="actual")

        pop_hra  = hr_gap_age_max(r["pop_recs"],  future_df)
        cf_hra   = hr_gap_age_max(r["cf_recs"],   future_df)
        rand_hra = hr_gap_age_max(r.get("rand_recs", {}), future_df)
        fair_hra = hr_gap_age_max(r.get("fair_recs", {}), future_df)
        ni10_hra = hr_gap_age_max(r.get("ni10_recs", {}), future_df, denom="actual")

        pop_ndcg  = ndcg_gap_gender(r["pop_recs"],  future_df)
        cf_ndcg   = ndcg_gap_gender(r["cf_recs"],   future_df)
        rand_ndcg = ndcg_gap_gender(r.get("rand_recs", {}), future_df)
        fair_ndcg = ndcg_gap_gender(r.get("fair_recs", {}), future_df)
        ni10_ndcg = ndcg_gap_gender(r.get("ni10_recs", {}), future_df)

        # HR@1: Hit Rate wenn nur die erste Empfehlung pro User gezählt wird
        hr1_pop  = compute_hit_rate_at_1(r["pop_recs"],  future_df)
        hr1_cf   = compute_hit_rate_at_1(r["cf_recs"],   future_df)
        hr1_rand = compute_hit_rate_at_1(r.get("rand_recs", {}), future_df)
        hr1_fair = compute_hit_rate_at_1(r.get("fair_recs", {}), future_df)
        hr1_ni10 = compute_hit_rate_at_1(r.get("ni10_recs", {}), future_df)
    else:
        pop_hrg = cf_hrg = rand_hrg = fair_hrg = ni10_hrg  = np.nan
        pop_hra = cf_hra = rand_hra = fair_hra = ni10_hra  = np.nan
        pop_ndcg= cf_ndcg=rand_ndcg=fair_ndcg=ni10_ndcg = np.nan
        hr1_pop = hr1_cf = hr1_rand = hr1_fair = hr1_ni10 = np.nan

    def _row(rs_name, prov, hit_rate, hrg, hra, ndcg_g,
             hit_rate_at_1=np.nan, cf_u=None):
        g, g_ou, ent, ap, ja, fs, mn, mx, cov = prov
        row = dict(
            split=pct_label, t=t, n_users=n_users, n_active_films=n_active,
            rs=rs_name, hit_rate=hit_rate, hit_rate_at_1=hit_rate_at_1,
            gini=g, gini_our=g_ou, entropy=ent, aplt=ap,
            jain=ja, fsat=fs, minskew=mn, maxskew=mx, coverage=cov,
            hr_gap_gender=hrg, hr_gap_age=hra, ndcg_gap_gender=ndcg_g,
            u_val_gender=np.nan, u_over_gender=np.nan, u_under_gender=np.nan,
            u_val_age=np.nan,    u_over_age=np.nan,    u_under_age=np.nan,
        )
        if cf_u: row.update(cf_u)
        return row

    cf_u = {
        "u_val_gender":  r.get("u_val_gender",   np.nan),
        "u_over_gender": r.get("u_over_gender",  np.nan),
        "u_under_gender":r.get("u_under_gender", np.nan),
        "u_val_age":     r.get("u_val_age",      np.nan),
        "u_over_age":    r.get("u_over_age",     np.nan),
        "u_under_age":   r.get("u_under_age",    np.nan),
    }

    rows_ext.extend([
        _row("Popularity",   prov_pop,  r["pop_hit_rate"],
             pop_hrg,  pop_hra,  pop_ndcg,  hr1_pop),
        _row("CF",           prov_cf,   r["cf_hit_rate"],
             cf_hrg,   cf_hra,   cf_ndcg,   hr1_cf,   cf_u),
        _row("Random",       prov_rand, r.get("rand_hit_rate", np.nan),
             rand_hrg, rand_hra, rand_ndcg, hr1_rand),
        _row("FairnessExp",  prov_fair, r.get("fair_hit_rate", np.nan),
             fair_hrg, fair_hra, fair_ndcg, hr1_fair),
        _row("NextItem_k10", prov_ni10, r.get("ni10_hit_rate", np.nan),
             ni10_hrg, ni10_hra, ni10_ndcg, hr1_ni10),
    ])

ext_df = pd.DataFrame(rows_ext)

def _rs(name): return ext_df[ext_df["rs"]==name].reset_index(drop=True)
pop_ext  = _rs("Popularity")
cf_ext   = _rs("CF")
rand_ext = _rs("Random")
fair_ext = _rs("FairnessExp")
ni10_ext = _rs("NextItem_k10")

ALL_RS = [
    ("Popularity",   pop_ext),
    ("CF",           cf_ext),
    ("Random",       rand_ext),
    ("FairnessExp",  fair_ext),
    ("NextItem_k10", ni10_ext),
]

# =============================================================================
# 8. CSV EXPORT — ERWEITERT
# =============================================================================
ext_df.to_csv(os.path.join(CSV_EXT, "fairness_rs_extended_all.csv"), index=False)
for rs_name, rs_df in ALL_RS:
    rs_df.to_csv(os.path.join(CSV_EXT, f"fairness_rs_{rs_name.lower()}.csv"), index=False)

print(f"CSVs -> {CSV_EXT}/")
print(f"  fairness_rs_extended_all.csv + 5 einzelne RS CSVs")

# =============================================================================
# 9. ERWEITERTE PLOTS (5 RS) -> plots/fairness_rs_extended/final/
# =============================================================================
print(f"\nGenerating extended plots (5 RS) -> {OUTPUT_EXT_FIN}/")

# --- Gini, Entropy, APLT, Jain (4 einzelne Dateien) -------------------------
_multi_single("gini",    "Gini",                  "Gini [0,1]",
               "0 = fair, 1 = unfair", "1a_gini.png", OUTPUT_EXT_FIN)
_multi_single("entropy", "Entropy (Entour)",      "Entropy [0,1]",
               "0 = unfair, 1 = fair", "1b_entropy.png", OUTPUT_EXT_FIN)
_multi_single("aplt",    "APLT: Long-Tail Share", "APLT [0,1]",
               "higher = fairer | FairnessExp excluded (near-uniform -> unstable)",
               "1c_aplt.png", OUTPUT_EXT_FIN, skip_rs=["FairnessExp"])
_multi_single("jain",    "Jain's Index",          "Jain [0,1]",
               "0 = unfair, 1 = fair", "1d_jain.png", OUTPUT_EXT_FIN)

# --- FSat, Coverage, MinSkew, MaxSkew (4 einzelne Dateien) ------------------
_multi_single("fsat",     "FSat: Satisfied Items",     "FSat [0,1]",
               "0 = unfair, 1 = fair | NextItem_k10 excluded (variable km)",
               "2a_fsat.png", OUTPUT_EXT_FIN, skip_rs=["NextItem_k10"])
_multi_single("coverage", "Coverage (QF)",             "Coverage [0,1]",
               "diversity metric", "2b_coverage.png", OUTPUT_EXT_FIN)
_multi_single("minskew",  "MinSkew (best-off film)",   "MinSkew (log)",
               "< 0 = more than fair share", "2c_minskew.png", OUTPUT_EXT_FIN)
_multi_single("maxskew",  "MaxSkew (worst-off film)",  "MaxSkew (log)",
               "> 0 = less than fair share | Random excluded (non-monotonic due to random seed)",
               "2d_maxskew.png", OUTPUT_EXT_FIN, skip_rs=["Random"])

# --- Hit Rate @TOP_N ----------------------------------------------------------
def _hitrate_all_single(filename, out_dir):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for rs_name, rs_df in [("Popularity", pop_result), ("CF", cf_result)]:
        hr_col = "hit_rate_pop" if rs_name == "Popularity" else "hit_rate_cf"
        y  = rs_df[hr_col].values
        st = RS_STYLES[rs_name]
        ax.plot(x_pos, y, color=st["color"], linewidth=st["lw"],
                marker=st["marker"], markersize=5, linestyle=st["ls"],
                label=rs_name, zorder=3, alpha=0.9)
    for rs_name, rs_df in [("Random", rand_ext), ("FairnessExp", fair_ext), ("NextItem_k10", ni10_ext)]:
        if "hit_rate" not in rs_df.columns: continue
        y  = rs_df["hit_rate"].values
        st = RS_STYLES[rs_name]
        ax.plot(x_pos, y, color=st["color"], linewidth=st["lw"],
                marker=st["marker"], markersize=5, linestyle=st["ls"],
                label=rs_name, zorder=3, alpha=0.9)
    _style_single(ax, "Hit Rate @TOP_N", "Hit Rate [0,1]",
                   "higher = better relevance | NextItem_k10 = Oracle (HR \u2248 1.0)")
    _legend_inside(ax)
    plt.tight_layout()
    path = os.path.join(out_dir, filename)
    plt.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"  Plot -> {path}")
    return path

_hitrate_all_single("4a_hit_rate_all_rs.png", OUTPUT_EXT_FIN)

# --- Hit Rate @1 ---------------------------------------------------------------
_multi_single("hit_rate_at_1", "Hit Rate @1 (Top-1 Recommendation per User)",
               "Hit Rate [0,1]", "higher = better top-1 precision",
               "4b_hit_rate_at_1_all_rs.png", OUTPUT_EXT_FIN)

# --- Group Fairness (3 einzelne Dateien) --------------------------------------
_multi_single("hr_gap_gender",   "HR Gap (Gender)",   "Gap [0,1]",
               "0 = fair | Next-Item excluded (variable list length)",
               "5a_hr_gap_gender.png", OUTPUT_EXT_FIN, skip_ni=True)
_multi_single("hr_gap_age",      "HR Gap (Age Max)",  "Gap [0,1]",
               "0 = fair | Next-Item excluded (variable list length)",
               "5b_hr_gap_age.png", OUTPUT_EXT_FIN, skip_ni=True)
_multi_single("ndcg_gap_gender", "nDCG Gap (Gender)", "Gap [0,1]",
               "0 = fair | Next-Item excluded (variable list length)",
               "5c_ndcg_gap_gender.png", OUTPUT_EXT_FIN, skip_ni=True)

# --- Individual Fairness (CF only), gepaart Gender | Age --------------------
_individual_pair("u_val_gender",   "u_val_age",   "Value Unfairness",
                 "U_val [>=0]",  "6a_value_unfairness_cf.png", OUTPUT_EXT_FIN, cf_ext)
_individual_pair("u_over_gender",  "u_over_age",  "Overestimation Unfairness",
                 "U_over [>=0]", "6b_overestimation_unfairness_cf.png", OUTPUT_EXT_FIN, cf_ext)
_individual_pair("u_under_gender", "u_under_age", "Underestimation Unfairness",
                 "U_under [>=0]","6c_underestimation_unfairness_cf.png", OUTPUT_EXT_FIN, cf_ext)

# --- Summary Table alle 5 RS (Split 10) ---------------------------------------
last_rows = {rs_name: rs_df.iloc[-1] for rs_name, rs_df in ALL_RS}
sum_metrics = [
    ("Gini",          "gini"),    ("Entropy",    "entropy"),
    ("APLT",          "aplt"),    ("Jain",       "jain"),
    ("FSat",          "fsat"),    ("Coverage",   "coverage"),
    ("MinSkew",       "minskew"), ("MaxSkew",    "maxskew"),
    ("HR Gap Gender", "hr_gap_gender"),
    ("HR Gap Age",    "hr_gap_age"),
    ("nDCG Gap",      "ndcg_gap_gender"),
]
col_labels = ["Metric"] + [rs for rs, _ in ALL_RS]
tdata7 = []
for m, col in sum_metrics:
    row = [m]
    for rs_name, _ in ALL_RS:
        v = last_rows[rs_name].get(col, np.nan)
        row.append(f"{v:.4f}" if not pd.isna(v) else "N/A")
    tdata7.append(row)

fig, ax = plt.subplots(figsize=(17, len(sum_metrics)*0.50+2)); ax.axis("off")
tbl = ax.table(cellText=tdata7, colLabels=col_labels,
               cellLoc="center", loc="center",
               colWidths=[0.22]+[0.155]*5)
tbl.auto_set_font_size(False); tbl.set_fontsize(8.5); tbl.scale(1, 1.5)
header_colors = ["#2166AC"] + [RS_STYLES[rs]["color"] for rs, _ in ALL_RS]
for j, hc in enumerate(header_colors):
    tbl[(0,j)].set_facecolor(hc)
    tbl[(0,j)].set_text_props(color="white", fontweight="bold")
for i in range(1, len(sum_metrics)+1):
    bg = "#f0f4f8" if i%2==0 else "white"
    for j in range(len(col_labels)): tbl[(i,j)].set_facecolor(bg)
ax.set_title("Fairness Summary — Split 10 (100% of data)  |  All RS  |  MovieLens 1M",
             fontsize=11, fontweight="bold", pad=20)
plt.tight_layout()
ep7 = os.path.join(OUTPUT_EXT_FIN, "7_summary_table_all_rs.png")
plt.savefig(ep7, dpi=DPI, bbox_inches="tight"); plt.close(); print(f"  Plot -> {ep7}")

# =============================================================================
# 10. ABSCHLUSS
# =============================================================================
print(f"""
{'='*65}
ALLE OUTPUTS
{'='*65}
  Bestehend (Popularity + CF):
    CSVs  ->  {CSV_DIR}/
      fairness_rs_all.csv
      fairness_rs_popularity.csv / fairness_rs_cf.csv
      fairness_rs_correlations_cf.csv / _popularity.csv
    Plots ->  {OUTPUT_DIR}/
      1a_gini.png
      1b_entropy.png
      1c_aplt.png
      1d_jain.png
      2a_fsat.png
      2b_coverage.png
      2c_minskew.png
      2d_maxskew.png
      4a_hit_rate.png
      4b_hr_gap_gender.png
      4c_hr_gap_age.png
      5_ndcg_gap_gender.png
      6a_value_unfairness.png            (Gender | Age, side by side)
      6b_overestimation_unfairness.png   (Gender | Age, side by side)
      6c_underestimation_unfairness.png  (Gender | Age, side by side)
      7_summary_table_split10.png
      8_correlation_matrix_cf.png
      9_correlation_matrix_popularity.png

  Neu (5 RS, ohne NextItem_k1):
    CSVs  ->  {CSV_EXT}/
      fairness_rs_extended_all.csv
      fairness_rs_popularity.csv / _cf.csv / _random.csv
      fairness_rs_fairnessexp.csv / _nextitem_k10.csv
    Plots ->  {OUTPUT_EXT_FIN}/
      1a_gini.png
      1b_entropy.png
      1c_aplt.png                        (FairnessExp excluded)
      1d_jain.png
      2a_fsat.png                        (NextItem_k10 excluded)
      2b_coverage.png
      2c_minskew.png
      2d_maxskew.png                     (Random excluded)
      4a_hit_rate_all_rs.png
      4b_hit_rate_at_1_all_rs.png
      5a_hr_gap_gender.png                (Next-Item excluded)
      5b_hr_gap_age.png                   (Next-Item excluded)
      5c_ndcg_gap_gender.png              (Next-Item excluded)
      6a_value_unfairness_cf.png          (Gender | Age, side by side)
      6b_overestimation_unfairness_cf.png (Gender | Age, side by side)
      6c_underestimation_unfairness_cf.png(Gender | Age, side by side)
      7_summary_table_all_rs.png
{'='*65}
""")