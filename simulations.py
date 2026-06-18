"""
simulation.py  -  MovieLens 1M
Bachelor Thesis  -  Simulation mit Hit Rate und Fake Ratings

RS ÜBERSICHT:
  1. Popularity      — Top-N meistbewertet, gesehene Filme gefiltert
  2. CF (SVD)        — personalisiert, Mean-Centering nur für Individual Fairness
  3. Random          — 10 zufällige ungesehene Filme pro User
  4. Fairness Exp.   — enforced equal exposure (Round-Robin, Greedy)
  5a. Next-Item k=1  — nächster tatsächlich bewerteter Film des Users
  5b. Next-Item k=10 — nächste 10 tatsächlich bewertete Filme des Users

Individual Fairness (U_val/U_over/U_under) nur für CF.

LAUFZEIT: ~90-120 Minuten für alle 10 Splits
"""

import pandas as pd
import numpy as np
import os
import pickle
from collections import defaultdict
import scipy.sparse as sp
from sklearn.decomposition import TruncatedSVD

# =============================================================================
# 0. CONFIG
# =============================================================================
DATA_DIR  = "ml-1m/ml-1m"
N_SPLITS  = 10
TOP_N     = 10
N_FACTORS = 50
FAST_TEST = False

AGE_MAP = {1: "Under 18", 18: "18-24", 25: "25-34", 35: "35-44",
           45: "45-49", 50: "50-55", 56: "56+"}

# =============================================================================
# 1. DATEN LADEN
# =============================================================================
print("Loading data ...")

users = pd.read_csv(
    os.path.join(DATA_DIR, "users.dat"), sep="::", engine="python",
    names=["UserID", "Gender", "Age", "Occupation", "Zip"], dtype={"Zip": str})
movies = pd.read_csv(
    os.path.join(DATA_DIR, "movies.dat"), sep="::", engine="python",
    names=["MovieID", "Title", "Genres"], encoding="latin-1")
ratings = pd.read_csv(
    os.path.join(DATA_DIR, "ratings.dat"), sep="::", engine="python",
    names=["UserID", "MovieID", "Rating", "Timestamp"])

users["AgeLabel"] = users["Age"].map(AGE_MAP)
ratings["Date"]   = pd.to_datetime(ratings["Timestamp"], unit="s")

df = ratings.merge(users[["UserID", "Gender", "AgeLabel"]], on="UserID", how="left")
df = df.sort_values("Timestamp").reset_index(drop=True)

total_ratings = len(df)
split_size    = total_ratings // N_SPLITS
df["Bucket"]  = np.clip(np.arange(len(df)) // split_size, 0, N_SPLITS - 1) + 1
pct_labels    = [f"{int(i * 100 / N_SPLITS)}%" for i in range(1, N_SPLITS + 1)]

title_map     = movies.set_index("MovieID")["Title"].to_dict()
all_movie_ids = df["MovieID"].unique().tolist()

all_users_global  = df["UserID"].unique()
all_movies_global = df["MovieID"].unique()
user_to_idx_g     = {uid: i for i, uid in enumerate(all_users_global)}
movie_to_idx_g    = {mid: i for i, mid in enumerate(all_movies_global)}

user_info = users.set_index("UserID")[["Gender", "AgeLabel"]].to_dict(orient="index")

print(f"  {total_ratings:,} ratings | {len(all_users_global)} users | "
      f"{len(all_movies_global)} movies | {N_SPLITS} splits")


# =============================================================================
# 2. HILFSFUNKTIONEN
# =============================================================================

def build_rec_counts(recommendations, all_movie_ids):
    """Zählt wie oft jeder Film empfohlen wird. 0 wenn nie empfohlen."""
    counts = defaultdict(int)
    for movie_list in recommendations.values():
        for mid in movie_list:
            counts[mid] += 1
    return pd.Series({mid: counts[mid] for mid in all_movie_ids})


def popularity_recommendations(train_df, all_users, n):
    """
    Top-n meistbewertete Filme. Gesehene Filme pro User ausgeschlossen.
    """
    popularity_ranking = (
        train_df.groupby("MovieID").size()
                .sort_values(ascending=False).index.tolist()
    )
    seen_per_user = train_df.groupby("UserID")["MovieID"].apply(set).to_dict()
    return {uid: [mid for mid in popularity_ranking
                  if mid not in seen_per_user.get(uid, set())][:n]
            for uid in all_users}


def cf_recommendations(train_df, n, n_factors=N_FACTORS):
    """
    CF via TruncatedSVD. Gibt auch R_pred zurück (für Individual Fairness).
    """
    user_ids  = train_df["UserID"].unique()
    movie_ids = train_df["MovieID"].unique()
    user_idx  = {uid: i for i, uid in enumerate(user_ids)}
    movie_idx = {mid: i for i, mid in enumerate(movie_ids)}

    rows = train_df["UserID"].map(user_idx).values
    cols = train_df["MovieID"].map(movie_idx).values
    vals = train_df["Rating"].values.astype(float)
    R    = sp.csr_matrix((vals, (rows, cols)),
                         shape=(len(user_ids), len(movie_ids)))

    k   = min(n_factors, len(movie_ids) - 1, len(user_ids) - 1)
    svd = TruncatedSVD(n_components=k, random_state=42)
    U   = svd.fit_transform(R)
    Vt  = svd.components_

    R_pred        = U @ Vt
    R_dense       = R.toarray()
    R_pred_masked = R_pred.copy()
    R_pred_masked[R_dense > 0] = -np.inf

    top_n_recs = {}
    for i, uid in enumerate(user_ids):
        top_indices = np.argpartition(R_pred_masked[i], -n)[-n:]
        top_indices = top_indices[np.argsort(R_pred_masked[i][top_indices])[::-1]]
        top_n_recs[uid] = [movie_ids[idx] for idx in top_indices]

    return top_n_recs, R_pred, user_ids, movie_ids


def random_recommendations(train_df, all_users, n, seed=42):
    """
    Zufällige RS: pro User n zufällige ungesehene Filme.

    LOGIK:
      Für jeden User: aktive Filme minus gesehene = Kandidatenpool
      Ziehe n zufällig ohne Zurücklegen aus Kandidatenpool.

    Kein Lerneffekt — dient als faire Baseline.

    WICHTIG: Nur Filme die bis Split t mindestens 1 Rating haben (aktive Filme).
    Filme die noch nie bewertet wurden werden nicht empfohlen — konsistent mit
    Popularity und CF die ebenfalls nur train_df-Filme kennen.
    """
    rng           = np.random.default_rng(seed)
    seen_per_user = train_df.groupby("UserID")["MovieID"].apply(set).to_dict()
    # Nur aktive Filme (mindestens 1 Rating in train_df) — kein all_movie_ids
    movie_ids_arr = np.array(train_df["MovieID"].unique())

    recs = {}
    for uid in all_users:
        seen     = seen_per_user.get(uid, set())
        unseen   = [mid for mid in movie_ids_arr if mid not in seen]
        k        = min(n, len(unseen))
        chosen   = rng.choice(unseen, size=k, replace=False).tolist()
        recs[uid] = chosen
    return recs


def fairness_exposure_recommendations(train_df, all_users, n, seed=42):
    """
    Enforced Equal Exposure RS (Round-Robin, Greedy).

    LOGIK:
      Ziel: jeder Film erhält annähernd gleich viele Empfehlungen.
      Greedy-Ansatz: pro User werden die n Filme mit den wenigsten
      bisherigen Empfehlungen gewählt (unter den ungesehenen Filmen).

      Damit wird km = n_users × n Empfehlungsslots möglichst gleichmässig
      auf alle Filme verteilt → enforced equal exposure.

    User werden zufällig gemischt um systematischen Bias zu vermeiden.
    Dies ist das theoretische Ideal der Provider-Side Fairness:
    alle Filme bekommen gleiche Exposition.
    """
    rng           = np.random.default_rng(seed)
    seen_per_user = train_df.groupby("UserID")["MovieID"].apply(set).to_dict()
    # Nur aktive Filme (mindestens 1 Rating in train_df) — kein all_movie_ids
    movie_ids_arr = np.array(train_df["MovieID"].unique())
    n_films       = len(movie_ids_arr)
    counts        = np.zeros(n_films, dtype=np.int32)

    # Zufällige Userreihenfolge für faire Verteilung
    shuffled_users = list(all_users)
    rng.shuffle(shuffled_users)

    recs = {}
    for uid in shuffled_users:
        seen       = seen_per_user.get(uid, set())
        seen_mask  = np.array([mid in seen for mid in movie_ids_arr], dtype=bool)

        # Kosten = aktuelle Empfehlungszählung, gesehene Filme → inf
        cost       = counts.astype(float)
        cost[seen_mask] = np.inf

        if np.all(np.isinf(cost)):
            recs[uid] = []
            continue

        # Wähle n Filme mit niedrigstem count (ties zufällig brechen)
        noise      = rng.random(n_films) * 1e-9
        cost_noisy = cost + noise
        top_idx    = np.argpartition(cost_noisy, n)[:n]
        top_idx    = top_idx[np.argsort(cost_noisy[top_idx])]

        recs[uid]  = movie_ids_arr[top_idx].tolist()
        counts[top_idx] += 1

    return recs


def next_item_recommendations(future_df, train_df, all_users, k):
    """
    Next-Item RS: empfiehlt die k Filme die der User als nächstes
    tatsächlich bewertet hat (chronologisch aus future_df).

    LOGIK:
      Für jeden User: nimm seine k nächsten echten Ratings aus future_df
      (chronologisch sortiert nach Timestamp).
      Gesehene Filme aus train_df werden gefiltert (sollte kein Problem
      sein da future_df = Ratings nach Split t).

    Wenn User < k zukünftige Ratings hat: kürzere Liste (kein Auffüllen).

    WICHTIG:
      Dieses RS ist ein "Perfect Oracle" — es kennt die Zukunft.
      Hit Rate = immer 1.0 (per Definition).
      Dient als theoretische Obergrenze der Empfehlungsqualität.
      Fairness-Metriken zeigen wie fair das reale User-Verhalten ist.
    """
    seen_per_user   = train_df.groupby("UserID")["MovieID"].apply(set).to_dict()
    future_sorted   = future_df.sort_values("Timestamp")

    recs = {}
    for uid in all_users:
        seen         = seen_per_user.get(uid, set())
        user_future  = future_sorted[future_sorted["UserID"] == uid]["MovieID"].tolist()
        # Filter gesehene (Sicherheit — sollte leer sein)
        unseen_future = [m for m in user_future if m not in seen]
        recs[uid]     = unseen_future[:k]
    return recs


def compute_hit_rate(recommendations, future_df, top_n):
    """
    Hit Rate: Anteil empfohlener Filme die der User später wirklich bewertet hat.
    Nenner = top_n (konstant für alle RS → Vergleichbarkeit).
    """
    future_per_user = (future_df.groupby("UserID")["MovieID"]
                                .apply(set).to_dict())
    hit_rates = []
    for uid, rec_list in recommendations.items():
        future_movies = future_per_user.get(uid, set())
        hits = len(set(rec_list) & future_movies)
        hit_rates.append(hits / top_n)
    return float(np.mean(hit_rates)) if hit_rates else np.nan


def compute_hit_rate_next_item(recommendations, future_df):
    """
    Hit Rate für Next-Item RS.
    Nenner = tatsächliche Listenlänge pro User (variiert je nach k).
    Ergebnis sollte 1.0 sein (Oracle-RS).
    """
    future_per_user = (future_df.groupby("UserID")["MovieID"]
                                .apply(set).to_dict())
    hit_rates = []
    for uid, rec_list in recommendations.items():
        if not rec_list:
            continue
        future_movies = future_per_user.get(uid, set())
        hits = len(set(rec_list) & future_movies)
        hit_rates.append(hits / len(rec_list))
    return float(np.mean(hit_rates)) if hit_rates else np.nan


# =============================================================================
# 3. 100%-MODELL TRAINIEREN (NUR FÜR FAKE RATINGS)
# =============================================================================
print("\nTraining 100% oracle model (for fake ratings only) ...")

rows_g = df["UserID"].map(user_to_idx_g).values
cols_g = df["MovieID"].map(movie_to_idx_g).values
vals_g = df["Rating"].values.astype(float)

R_full = sp.csr_matrix(
    (vals_g, (rows_g, cols_g)),
    shape=(len(all_users_global), len(all_movies_global))
)
svd_full    = TruncatedSVD(n_components=N_FACTORS, random_state=42)
U_full      = svd_full.fit_transform(R_full)
Vt_full     = svd_full.components_
R_pred_full = U_full @ Vt_full

print(f"  Done. Shape: {R_pred_full.shape}  |  "
      f"Memory: ~{R_pred_full.nbytes / 1e6:.0f} MB\n")


def get_fake_rating(user_id, movie_id):
    u_idx = user_to_idx_g.get(user_id)
    m_idx = movie_to_idx_g.get(movie_id)
    if u_idx is None or m_idx is None:
        return 3.0
    return float(np.clip(R_pred_full[u_idx, m_idx], 1.0, 5.0))


# =============================================================================
# 4. SIMULATION SCHLEIFE
# =============================================================================
splits_to_run = [1, N_SPLITS - 1] if FAST_TEST else list(range(1, N_SPLITS + 1))
if FAST_TEST:
    print(f"FAST_TEST=True → nur Splits {splits_to_run}")

fake_ratings_accumulated = []
simulation_results       = []

for t in splits_to_run:
    pct_label = pct_labels[t - 1]

    print(f"\n{'='*70}")
    print(f"SPLIT t={t}/{N_SPLITS}  |  {pct_label}  |  "
          + (f"Testset = Splits {t+1}..{N_SPLITS}" if t < N_SPLITS else "kein Testset"))
    print(f"{'='*70}")

    # ── A: Trainingsdaten ────────────────────────────────────────────────────
    real_train = df[df["Bucket"] <= t].copy()

    if fake_ratings_accumulated:
        fake_so_far = pd.DataFrame(fake_ratings_accumulated)
        fake_so_far = fake_so_far[fake_so_far["Bucket"] < t]
        train_df    = pd.concat([real_train, fake_so_far], ignore_index=True)
        n_fake_used = len(fake_so_far)
    else:
        train_df    = real_train
        n_fake_used = 0

    all_users_t = train_df["UserID"].unique().tolist()

    print(f"\n  Trainingsdaten:")
    print(f"    Echte Ratings   : {len(real_train):,}")
    print(f"    Fake Ratings    : {n_fake_used:,}")
    print(f"    Total           : {len(train_df):,}")
    print(f"    User            : {len(all_users_t):,}")

    # ── B: Empfehlungen ──────────────────────────────────────────────────────
    print(f"\n  [Popularity RS] ...")
    pop_recs = popularity_recommendations(train_df, all_users_t, TOP_N)

    print(f"  [CF SVD RS] ...")
    cf_recs, R_pred_cf, cf_user_ids, cf_movie_ids = cf_recommendations(train_df, TOP_N)

    print(f"  [Random RS] ...")
    rand_recs = random_recommendations(train_df, all_users_t, TOP_N, seed=42+t)

    print(f"  [Fairness Exposure RS] ...")
    fair_recs = fairness_exposure_recommendations(train_df, all_users_t, TOP_N, seed=42+t)

    pop_counts  = build_rec_counts(pop_recs,  all_movie_ids)
    cf_counts   = build_rec_counts(cf_recs,   all_movie_ids)
    rand_counts = build_rec_counts(rand_recs, all_movie_ids)
    fair_counts = build_rec_counts(fair_recs, all_movie_ids)

    print(f"    Pop  : {(pop_counts  > 0).sum():,} einzigartige Filme")
    print(f"    CF   : {(cf_counts   > 0).sum():,} einzigartige Filme")
    print(f"    Rand : {(rand_counts > 0).sum():,} einzigartige Filme")
    print(f"    Fair : {(fair_counts > 0).sum():,} einzigartige Filme")

    # ── C: Hit Rate + Next-Item RS ───────────────────────────────────────────
    if t < N_SPLITS:
        future_df = df[df["Bucket"] > t]

        pop_hr  = compute_hit_rate(pop_recs,  future_df, TOP_N)
        cf_hr   = compute_hit_rate(cf_recs,   future_df, TOP_N)
        rand_hr = compute_hit_rate(rand_recs, future_df, TOP_N)
        fair_hr = compute_hit_rate(fair_recs, future_df, TOP_N)

        # Next-Item RS (nur wenn Testset vorhanden)
        print(f"  [Next-Item RS k=1] ...")
        ni1_recs  = next_item_recommendations(future_df, train_df, all_users_t, k=1)

        print(f"  [Next-Item RS k=10] ...")
        ni10_recs = next_item_recommendations(future_df, train_df, all_users_t, k=10)

        ni1_counts  = build_rec_counts(ni1_recs,  all_movie_ids)
        ni10_counts = build_rec_counts(ni10_recs, all_movie_ids)

        ni1_hr  = compute_hit_rate_next_item(ni1_recs,  future_df)
        ni10_hr = compute_hit_rate_next_item(ni10_recs, future_df)

        # Durchschnittliche Listenlänge Next-Item
        avg_ni1_len  = np.mean([len(v) for v in ni1_recs.values()  if v])
        avg_ni10_len = np.mean([len(v) for v in ni10_recs.values() if v])

        print(f"\n  Hit Rate (Testset = Splits {t+1}..{N_SPLITS}):")
        print(f"    Popularity     : {pop_hr:.4f}  ({pop_hr*100:.2f}%)")
        print(f"    CF             : {cf_hr:.4f}  ({cf_hr*100:.2f}%)")
        print(f"    Random         : {rand_hr:.4f}  ({rand_hr*100:.2f}%)")
        print(f"    Fairness Exp.  : {fair_hr:.4f}  ({fair_hr*100:.2f}%)")
        print(f"    Next-Item k=1  : {ni1_hr:.4f}  (avg list len={avg_ni1_len:.1f})")
        print(f"    Next-Item k=10 : {ni10_hr:.4f}  (avg list len={avg_ni10_len:.1f})")

    else:
        future_df   = pd.DataFrame()
        pop_hr = cf_hr = rand_hr = fair_hr = np.nan
        ni1_recs = ni10_recs = {}
        ni1_counts  = pd.Series({mid: 0 for mid in all_movie_ids})
        ni10_counts = pd.Series({mid: 0 for mid in all_movie_ids})
        ni1_hr = ni10_hr = np.nan
        print(f"\n  Hit Rate: N/A (Split {N_SPLITS} = 100%)")
        print(f"  Next-Item: N/A (kein Testset)")

    # ── D: Fake Ratings (basierend auf CF) ───────────────────────────────────
    if t < N_SPLITS:
        print(f"\n  Modelliere User-Verhalten (max. 1 Eintrag pro User, CF-Empfehlungen) ...")

        future_real = df[df["Bucket"] > t]
        future_ratings_per_user = (
            future_real.groupby("UserID")
                       .apply(lambda x: dict(zip(x["MovieID"], x["Rating"])))
                       .to_dict()
        )
        n_new_real = 0
        n_new_fake = 0

        for uid, rec_list in cf_recs.items():
            user_meta      = user_info.get(uid, {"Gender": "M", "AgeLabel": "25-34"})
            future_ratings = future_ratings_per_user.get(uid, {})
            real_hits      = {mid: future_ratings[mid]
                              for mid in rec_list if mid in future_ratings}

            if real_hits:
                best_mid    = max(real_hits, key=real_hits.get)
                fake_ratings_accumulated.append({
                    "UserID": uid, "MovieID": best_mid,
                    "Rating": real_hits[best_mid], "Timestamp": 0,
                    "Date": pd.NaT, "Bucket": t,
                    "Gender": user_meta["Gender"], "AgeLabel": user_meta["AgeLabel"],
                    "is_fake": False,
                })
                n_new_real += 1
            else:
                pred_scores = {mid: get_fake_rating(uid, mid) for mid in rec_list}
                best_mid    = max(pred_scores, key=pred_scores.get)
                fake_ratings_accumulated.append({
                    "UserID": uid, "MovieID": best_mid,
                    "Rating": pred_scores[best_mid], "Timestamp": 0,
                    "Date": pd.NaT, "Bucket": t,
                    "Gender": user_meta["Gender"], "AgeLabel": user_meta["AgeLabel"],
                    "is_fake": True,
                })
                n_new_fake += 1

        print(f"    Echte Reaktionen : {n_new_real:,}")
        print(f"    Fake Ratings     : {n_new_fake:,}")
        print(f"    Total akkumuliert: {len(fake_ratings_accumulated):,}")

    # ── E: Individual Fairness — Mean-Centered SVD (CF only) ─────────────────
    print(f"\n  [Individual Fairness — Mean-Centered SVD] ...")
    AGE_ORDER_SIM = ["Under 18", "18-24", "25-34", "35-44", "45-49", "50-55", "56+"]

    _user_ids_f  = train_df["UserID"].unique()
    _movie_ids_f = train_df["MovieID"].unique()
    _user_idx_f  = {uid: i for i, uid in enumerate(_user_ids_f)}
    _movie_idx_f = {mid: i for i, mid in enumerate(_movie_ids_f)}

    _rows_f        = train_df["UserID"].map(_user_idx_f).values
    _cols_f        = train_df["MovieID"].map(_movie_idx_f).values
    _vals_f        = train_df["Rating"].values.astype(float)
    _global_mean   = _vals_f.mean()
    _vals_centered = _vals_f - _global_mean

    _R_fair   = sp.csr_matrix((_vals_centered, (_rows_f, _cols_f)),
                               shape=(len(_user_ids_f), len(_movie_ids_f)))
    _k_fair   = min(N_FACTORS, len(_movie_ids_f) - 1, len(_user_ids_f) - 1)
    _svd_fair = TruncatedSVD(n_components=_k_fair, random_state=42)
    _U_fair   = _svd_fair.fit_transform(_R_fair)
    _Vt_fair  = _svd_fair.components_

    R_pred_fair = np.clip(_U_fair @ _Vt_fair + _global_mean, 1.0, 5.0)
    print(f"    global_mean={_global_mean:.3f}  "
          f"pred=[{R_pred_fair.min():.2f}, {R_pred_fair.max():.2f}]")

    def _uval_trio(Ey_g, Er_g, Ey_ng, Er_ng, valid):
        ey_g,  er_g  = Ey_g[valid],  Er_g[valid]
        ey_ng, er_ng = Ey_ng[valid], Er_ng[valid]
        return (float(np.mean(np.abs((ey_g - er_g) - (ey_ng - er_ng)))),
                float(np.mean(np.abs(np.maximum(0, ey_g-er_g)  - np.maximum(0, ey_ng-er_ng)))),
                float(np.mean(np.abs(np.maximum(0, er_g-ey_g)  - np.maximum(0, er_ng-ey_ng)))))

    female_avg = train_df[train_df["Gender"]=="F"].groupby("MovieID")["Rating"].mean()
    male_avg   = train_df[train_df["Gender"]=="M"].groupby("MovieID")["Rating"].mean()
    Er_f_all   = female_avg.reindex(_movie_ids_f).values
    Er_m_all   = male_avg.reindex(_movie_ids_f).values

    genders_local = np.array([user_info.get(uid, {"Gender": "M"})["Gender"]
                               for uid in _user_ids_f])
    f_mask = (genders_local == "F")
    m_mask = (genders_local == "M")

    if f_mask.sum() > 0 and m_mask.sum() > 0:
        Ey_f  = R_pred_fair[f_mask].mean(axis=0)
        Ey_m  = R_pred_fair[m_mask].mean(axis=0)
        valid = ~(np.isnan(Er_f_all) | np.isnan(Er_m_all))
        u_val_g, u_over_g, u_under_g = (_uval_trio(Ey_f, Er_f_all, Ey_m, Er_m_all, valid)
                                         if valid.sum() > 0 else (np.nan, np.nan, np.nan))
    else:
        u_val_g = u_over_g = u_under_g = np.nan

    ages_local   = np.array([user_info.get(uid, {"AgeLabel": ""})["AgeLabel"]
                              for uid in _user_ids_f])
    present_ages = [a for a in AGE_ORDER_SIM if (ages_local == a).sum() > 0]
    val_list, over_list, under_list = [], [], []

    if len(present_ages) >= 2:
        Ey_age = {a: R_pred_fair[(ages_local == a)].mean(axis=0) for a in present_ages}
        Er_age = {}
        for a in present_ages:
            age_avg   = train_df[train_df["AgeLabel"] == a].groupby("MovieID")["Rating"].mean()
            Er_age[a] = age_avg.reindex(_movie_ids_f).values
        for i, aa in enumerate(present_ages):
            for ab in present_ages[i+1:]:
                valid = ~(np.isnan(Er_age[aa]) | np.isnan(Er_age[ab]))
                if valid.sum() == 0: continue
                v, ov, un = _uval_trio(Ey_age[aa], Er_age[aa], Ey_age[ab], Er_age[ab], valid)
                val_list.append(v); over_list.append(ov); under_list.append(un)

    u_val_a   = float(max(val_list))   if val_list   else np.nan
    u_over_a  = float(max(over_list))  if over_list  else np.nan
    u_under_a = float(max(under_list)) if under_list else np.nan

    if not np.isnan(u_val_g):
        print(f"    U_val_gender={u_val_g:.4f}  U_over={u_over_g:.4f}  U_under={u_under_g:.4f}")
        print(f"    U_val_age={u_val_a:.4f}    U_over={u_over_a:.4f}  U_under={u_under_a:.4f}")

    # ── Ergebnisse speichern ─────────────────────────────────────────────────
    def _r(v): return round(v, 4) if not np.isnan(v) else np.nan

    simulation_results.append({
        "split"          : pct_label,
        "t"              : t,
        "n_real_train"   : len(real_train),
        "n_fake_train"   : n_fake_used,
        "n_total_train"  : len(train_df),
        "n_users"        : len(all_users_t),
        # Hit Rates
        "pop_hit_rate"   : _r(pop_hr),
        "cf_hit_rate"    : _r(cf_hr),
        "rand_hit_rate"  : _r(rand_hr),
        "fair_hit_rate"  : _r(fair_hr),
        "ni1_hit_rate"   : _r(ni1_hr),
        "ni10_hit_rate"  : _r(ni10_hr),
        # Recommendation lists
        "pop_recs"       : pop_recs,
        "cf_recs"        : cf_recs,
        "rand_recs"      : rand_recs,
        "fair_recs"      : fair_recs,
        "ni1_recs"       : ni1_recs,
        "ni10_recs"      : ni10_recs,
        # Counts
        "pop_counts"     : pop_counts,
        "cf_counts"      : cf_counts,
        "rand_counts"    : rand_counts,
        "fair_counts"    : fair_counts,
        "ni1_counts"     : ni1_counts,
        "ni10_counts"    : ni10_counts,
        # Individual Fairness (CF only)
        "u_val_gender"   : _r(u_val_g),
        "u_over_gender"  : _r(u_over_g),
        "u_under_gender" : _r(u_under_g),
        "u_val_age"      : _r(u_val_a),
        "u_over_age"     : _r(u_over_a),
        "u_under_age"    : _r(u_under_a),
    })


# =============================================================================
# 5. ZUSAMMENFASSUNG
# =============================================================================
print("\n\n" + "=" * 75)
print("SIMULATION ABGESCHLOSSEN — ZUSAMMENFASSUNG")
print("=" * 75)

header = (f"{'Split':<6}  {'Echte':>8}  {'Fake':>7}  {'Pop HR':>7}  "
          f"{'CF HR':>7}  {'Rand HR':>8}  {'Fair HR':>8}  "
          f"{'NI1 HR':>7}  {'NI10 HR':>8}")
print(header)
print("-" * 75)
for r in simulation_results:
    def _fmt(v): return f"{v:.4f}" if not pd.isna(v) else "   N/A"
    print(f"  {r['split']:<6}  {r['n_real_train']:>8,}  {r['n_fake_train']:>7,}  "
          f"{_fmt(r['pop_hit_rate']):>7}  {_fmt(r['cf_hit_rate']):>7}  "
          f"{_fmt(r['rand_hit_rate']):>8}  {_fmt(r['fair_hit_rate']):>8}  "
          f"{_fmt(r['ni1_hit_rate']):>7}  {_fmt(r['ni10_hit_rate']):>8}")

print(f"\nTotal fake ratings erstellt: {len(fake_ratings_accumulated):,}")

# =============================================================================
# 6. PICKLE SPEICHERN
# =============================================================================
os.makedirs("results", exist_ok=True)
save_path = os.path.join("results", "simulation_results.pkl")
with open(save_path, "wb") as f:
    pickle.dump({
        "simulation_results": simulation_results,
        "all_movie_ids"     : all_movie_ids,
        "pct_labels"        : pct_labels,
    }, f)
print(f"\nErgebnisse gespeichert: {save_path}")
print("→ Jetzt fairness_metrics_rs.py laufen lassen.")

print("""
HOW TO READ THE RESULTS:
─────────────────────────────────────────────────────────────
  Hit Rate:
    = Anteil der Empfehlungen die der User DANACH bewertet hat
    Typische Werte: 0.02–0.10 (2%–10%)
    Das klingt niedrig — ist normal, weil User nur wenige Filme bewerten.

    RS Vergleich:
      Next-Item  → Hit Rate ~1.0 (Oracle — kennt die Zukunft)
      CF         → personalisiert, höher als Popularity
      Popularity → empfiehlt populäre Filme die sowieso bewertet werden
      Random     → Zufallsbasis, sehr niedrig
      Fair Exp.  → ähnlich Random, kein Lerneffekt

    Genau deshalb brauchen wir BEIDES: Relevanz + Fairness.
    Ein perfekter RS (Next-Item) kann trotzdem unfair sein.

  Fake Ratings:
    = Simulierte User-Reaktionen auf CF-Empfehlungen
    → Mit jedem Split t wächst die Anzahl der fake ratings
    → Das CF-Modell bei t+1 hat damit mehr Daten

  Neue RS:
    Random         → 10 zufällige ungesehene Filme pro User
                     dient als faire untere Schranke
    Fairness Exp.  → enforced equal exposure (Round-Robin)
                     zeigt wie Fairness-Metriken bei idealer
                     Provider-Fairness aussehen
    Next-Item k=1  → nächster tatsächlich bewerteter Film
                     theoretische Obergrenze (Oracle)
    Next-Item k=10 → nächste 10 tatsächlich bewertete Filme
                     Oracle mit grösserem Fenster
─────────────────────────────────────────────────────────────
""")