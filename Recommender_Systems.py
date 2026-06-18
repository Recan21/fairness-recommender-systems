"""
recommender_systems.py  -  MovieLens 1M
Bachelor Thesis  -  Recommender System Implementations

Zwei RS-Typen für späteren Fairness-Vergleich:
  1. Popularity-Based  — empfiehlt global meistbewertete Filme
  2. Collaborative Filtering (SVD)  — Matrix-Faktorisierung via sklearn (TruncatedSVD)

Pro Zeitschnitt k (dieselben 10 kumulativen Splits wie analysis_14march.py):
  - Training auf allen Ratings bis Split k
  - Generiere Top-N Empfehlungen für alle User im Trainingsset
  - Speichere rec_counts: wie oft jeder Film empfohlen wird
    → rec_counts ist später direkt der Input für gini(), entropy(), aplt_adapted() etc.
    → entspricht item_counts in analysis_14march.py

Laufzeit:
  FAST_TEST = True  → nur letzter Split, ~3-5 min, zum Testen
  FAST_TEST = False → alle 10 Splits, ~30-50 min, für Vollanalyse
"""

import pandas as pd
import numpy as np
import os
from collections import defaultdict
import scipy.sparse as sp
from sklearn.decomposition import TruncatedSVD

# =============================================================================
# 0.  CONFIG
# =============================================================================
DATA_DIR   = "ml-1m/ml-1m"
N_SPLITS   = 10
TOP_N      = 10          # Empfehlungen pro User
FAST_TEST  = False        # True = nur Split 10 (schneller Test), False = alle Splits

AGE_MAP = {1: "Under 18", 18: "18-24", 25: "25-34", 35: "35-44",
           45: "45-49", 50: "50-55", 56: "56+"}

# =============================================================================
# 1.  DATEN LADEN  (gleiche Logik wie analysis_14march.py)
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

# Gleiche Splits wie analysis_14march.py
total_ratings = len(df)
split_size    = total_ratings // N_SPLITS
df["Bucket"]  = np.clip(np.arange(len(df)) // split_size, 0, N_SPLITS - 1) + 1
pct_labels    = [f"{int(i * 100 / N_SPLITS)}%" for i in range(1, N_SPLITS + 1)]

# Hilfreich für Verifikation: Film-Titel nachschlagen
title_map = movies.set_index("MovieID")["Title"].to_dict()

all_movie_ids = df["MovieID"].unique().tolist()

print(f"  {total_ratings:,} ratings | {df['UserID'].nunique()} users | "
      f"{df['MovieID'].nunique()} movies | {N_SPLITS} splits")


# =============================================================================
# 2.  HILFSFUNKTIONEN
# =============================================================================

def build_rec_counts(recommendations, all_movie_ids):
    """
    Zählt wie oft jeder Film über alle Empfehlungslisten vorkommt.

    Input:
      recommendations  dict { UserID: [MovieID, MovieID, ...] }
      all_movie_ids    Liste aller MovieIDs im Dataset

    Output:
      pd.Series  Index=MovieID, Value=Anzahl Empfehlungen
      Filme die nie empfohlen werden haben Value=0

    WARUM:
      Das ist das RS-Äquivalent zu item_counts in analysis_14march.py.
      Statt "wie oft wurde Film X bewertet" messen wir "wie oft wurde Film X
      empfohlen" → direkter Input für die Fairness-Metriken später.
    """
    #recommendation is output from pop/cf: 
    """    
    1: [2858, 260, 1196, 1210, 480],   # User 1 bekommt diese 5 Filme
    2: [2858, 260, 1196, 1210, 480],   # User 2 (Popularity: gleiche Liste)
    3: [356,  296, 2858, 1197, 593],   # User 3 (CF: andere Liste)
    """
    counts = defaultdict(int) # startet bei 0
    for movie_list in recommendations.values(): # all movie values for every user
        for mid in movie_list: # every movie
            counts[mid] += 1
    return pd.Series({mid: counts[mid] for mid in all_movie_ids}) # liste von movieID sortiert nach ID ascending with the # of rec.


def popularity_recommendations(train_df, all_users, n):
    """
    Popularity-Based RS: empfiehlt die n meistbewerteten Filme im Trainingsset.
    Jeder User bekommt dieselbe Liste.

    LOGIK:
      1. Zähle Ratings pro Film im Trainingsset
      2. Nimm Top-n Filme nach Anzahl
      3. Weise dieselbe Liste jedem User zu

    STÄRKEN:  sehr einfach, guter Baseline, schnell
    SCHWÄCHEN: keinerlei Personalisierung, benachteiligt Long-Tail-Filme
    """
    top_movies = (
        train_df.groupby("MovieID")
                .size() # counts how many ratings a movie received
                .sort_values(ascending=False)
                .head(n) # nur top ten
                .index.tolist()
    )
    return {uid: top_movies for uid in all_users} # gives everyone the same list

# still some randomness -> the weight of choosing it depends on its popularity

def cf_recommendations(train_df, n, n_factors=50):
    """
    Collaborative Filtering via TruncatedSVD (Matrix-Faktorisierung).
    Verwendet sklearn — keine externe Kompilierung nötig.

    LOGIK (Schritt für Schritt):
      1. Baue eine User×Film-Matrix R  (Zeile = User, Spalte = Film, Wert = Rating)
         Die meisten Einträge sind 0 (User hat Film nicht bewertet).
      2. TruncatedSVD zerlegt R in:
           R  ≈  U  ×  Vt
           U   (User × k)   — jeder User als Vektor mit k "Geschmacks-Dimensionen"
           Vt  (k × Filme)  — jeder Film als Vektor mit k "Eigenschaften"
         k = n_factors = 50  (Standard, gut für MovieLens 1M)
      3. Predicted-Rating-Matrix = U × Vt
         Für jeden User und jeden Film: Skalarprodukt → geschätztes Rating
      4. Pro User: setze bereits bewertete Filme auf -∞ (sollen nicht empfohlen werden)
         → sortiere verbleibende Filme nach predicted score → Top-n

    STÄRKEN:  personalisiert, erkennt Geschmacksmuster, kein C++ nötig
    SCHWÄCHEN: braucht mehr Speicher als surprise (ganze Matrix im RAM)
               bei 6040 User × 3706 Filme = ~22M Einträge, ca. 170 MB → kein Problem
    """
    # --- Schritt 1: Integer-Index für User und Filme erstellen ---------------
    # SVD braucht numerische Indizes (0, 1, 2, ...) statt der echten IDs
    user_ids  = train_df["UserID"].unique()
    movie_ids = train_df["MovieID"].unique()

    user_idx  = {uid: i for i, uid in enumerate(user_ids)}
    movie_idx = {mid: i for i, mid in enumerate(movie_ids)}

    """
    user_idx  = {1: 0,  2: 1,  3: 2,  ...}   # UserID 1  → Zeile 0
    movie_idx = {1: 0, 48: 1, 150: 2, ...}   # MovieID 1 → Spalte 0
    
            Film1  Film2  Film3  Film4  Film5
    User 1     5      0      3      0      0
    User 2     0      4      0      5      0
    User 3     0      0      2      0      4
    User 4     3      0      0      0      5
    
    Baue eine User×Film-Matrix R  (Zeile = User, Spalte = Film, Wert = Rating)
    Die meisten Einträge sind 0 (User hat Film nicht bewertet).

    """

    n_users  = len(user_ids)
    n_movies = len(movie_ids)

    # --- Schritt 2: Sparse User×Film-Matrix bauen ----------------------------
    # Sparse = nur Nicht-Null-Einträge gespeichert → spart Speicher
    rows   = train_df["UserID"].map(user_idx).values
    cols   = train_df["MovieID"].map(movie_idx).values
    vals   = train_df["Rating"].values.astype(float)
    R      = sp.csr_matrix((vals, (rows, cols)), shape=(n_users, n_movies))

    """
    UserID  MovieID  Rating       rows  cols  vals
    1       2858     5      →      0     394   5.0
    1       260      4      →      0     22    4.0
    2       1196     3      →      1     87    3.0
    to save ram, 1 million entries instead of 6040 users * 3706 movies = 22 million entries
    """

    # --- Schritt 3: TruncatedSVD fitten --------------------------------------
    print(f"    Fitting TruncatedSVD (k={n_factors}) ...")
    svd = TruncatedSVD(n_components=n_factors, random_state=42)
    U   = svd.fit_transform(R)   # U: (n_users × n_factors) 6040 * 50
    Vt  = svd.components_        # Vt: (n_factors × n_movies) 50 * 3706
    # factors like: prefers action, prefers old films, gives high ratings in general
    # does it work -> change of metrics 
    
    """
    U[0] = [0.12, -0.34, 0.89, ...]   ← User 1 als 50-Zahlen-Vektor
    U[1] = [0.45,  0.21, 0.33, ...]   ← User 2 als 50-Zahlen-Vektor

    Vt[:,394] = [0.67, 0.11, 0.54, ...]  ← Film 394 als 50-Zahlen-Vektor
    """

    # --- Schritt 4: Predicted-Rating-Matrix berechnen ------------------------
    # R_pred[user_i, movie_j] = geschätztes Rating von User i für Film j
    print(f"    Computing predicted ratings ...")
    R_pred = U @ Vt              # Matrix-Multiplikation: (n_users × n_movies)

    """
            Film1  Film2  Film3  Film4  Film5
    User 1    4.8    2.1    3.2    1.9    3.7   ← geschätzte Ratings
    User 2    1.3    3.9    2.1    4.7    1.2
    User 3    2.2    1.8    3.1    2.3    4.5

    User 1 mag Action (Faktor 3 = 0.89)
    Film 394 ist Action (Faktor 3 = 0.54)
    → Skalarprodukt hoch → hoher predicted Score
    """

    # --- Schritt 5: Bereits bewertete Filme maskieren ------------------------
    # Nur filme die noch nicht bewertet worden sind für die empfehlungen
    # Wandle sparse Matrix in dense um und setze bekannte Ratings auf -inf
    # → diese Filme kommen nicht in die Top-n
    R_dense = R.toarray()
    R_pred[R_dense > 0] = -np.inf

    """
    Vor Maskierung:
    R_pred[User1] = [4.8, 2.1, 5.3, 1.9, 3.7, ...]

    R_dense[User1] = [5.0, 0,   4.0, 0,   0, ...]   ← echte Ratings (0 = nicht gesehen)

    Nach Maskierung:
    R_pred[User1] = [-inf, 2.1, -inf, 1.9, 3.7, ...]
                    ↑ bereits gesehen → kann nie Top-N werden
    """

    # --- Schritt 6: Top-n pro User bestimmen ---------------------------------
    print(f"    Extracting top-{n} recommendations per user ...")
    # np.argpartition ist schneller als full sort für große Matrizen
    top_n_recs = {}
    for i, uid in enumerate(user_ids):
        # Indizes der top-n Filme für diesen User (nach predicted score)
        top_indices = np.argpartition(R_pred[i], -n)[-n:]
        # Sortiere diese n Indizes noch nach Score (argpartition gibt keine Reihenfolge)
        top_indices = top_indices[np.argsort(R_pred[i][top_indices])[::-1]]
        # Konvertiere Index zurück zu echter MovieID
        top_n_recs[uid] = [movie_ids[idx] for idx in top_indices]

    return top_n_recs

    """
    R_pred[0] = [-inf, 2.1, -inf, 1.9, 3.7, 4.2, 2.8, ...]

    argpartition(..., -10)[-10:]  →  [5, 4, 6, 11, 23, ...]   ← Indizes der Top-10
    argsort(R_pred[0][[5,4,6,...]])[::-1]  →  sortiert diese 10 nach Score

    movie_ids[[5, 4, 6, ...]]  →  [2858_idx→echteID, ...]
    →  top_n_recs[1] = [356, 593, 2858, ...]   ← echte MovieIDs
    
    train_df (Ratings bis Split k)
        ↓
    [Schritt 1] IDs → Matrix-Indizes
            ↓
    [Schritt 2] Sparse Matrix R  (6040×3706, nur 1M Einträge)
            ↓
    [Schritt 3] SVD → U (6040×50) × Vt (50×3706)
            ↓
    [Schritt 4] R_pred = U @ Vt  → volle 6040×3706 Matrix
            ↓
    [Schritt 5] Gesehene Filme → -inf
            ↓
    [Schritt 6] Top-10 pro User → top_n_recs {UserID: [MovieID, ...]}
            ↓
    build_rec_counts → cf_counts  (MovieID → wie oft empfohlen)
            ↓
    gini(cf_counts.values), entropy(...), aplt(...)
    """

# =============================================================================
# 3.  HAUPTSCHLEIFE: RS für jeden Zeitschnitt berechnen
# =============================================================================

# FAST_TEST: nur den letzten Split berechnen (für schnellen Funktionstest)
splits_to_run = [N_SPLITS] if FAST_TEST else list(range(1, N_SPLITS + 1))
if FAST_TEST:
    print(f"\nFAST_TEST=True → nur Split {N_SPLITS} ({pct_labels[-1]})")
    print("Setze FAST_TEST=False für alle 10 Splits.\n")

# Ergebnisse für alle Splits speichern
# Struktur: Liste von Dicts, ein Dict pro Split
# Später: fairness-metriken direkt auf pop_counts / cf_counts anwenden
results = []

for bucket in splits_to_run:
    pct_label = pct_labels[bucket - 1]
    train_df  = df[df["Bucket"] <= bucket].copy()
    all_users = train_df["UserID"].unique().tolist()
    n_ratings = len(train_df)

    print(f"\n{'='*60}")
    print(f"Split {bucket}/{N_SPLITS}  |  {pct_label}  |  "
          f"{n_ratings:,} ratings  |  {len(all_users):,} users")
    print(f"{'='*60}")

    # -------------------------------------------------------------------------
    # 3a. Popularity-Based RS
    # -------------------------------------------------------------------------
    print("  [Popularity RS] ...")
    pop_recs   = popularity_recommendations(train_df, all_users, TOP_N)
    pop_counts = build_rec_counts(pop_recs, all_movie_ids)
    n_pop_covered = (pop_counts > 0).sum()
    print(f"    Users mit Empfehlungen : {len(pop_recs):,}")
    print(f"    Einzigartige Filme     : {n_pop_covered}  "
          f"(von {len(all_movie_ids)} total)")
    # Popularity empfiehlt immer genau TOP_N verschiedene Filme an alle User
    # → nur TOP_N Filme haben rec_count > 0 → sehr ungleiche Verteilung

    # -------------------------------------------------------------------------
    # 3b. Collaborative Filtering RS
    # -------------------------------------------------------------------------
    print("  [CF SVD RS] ...")
    cf_recs   = cf_recommendations(train_df, TOP_N)
    cf_counts = build_rec_counts(cf_recs, all_movie_ids)
    n_cf_covered = (cf_counts > 0).sum()
    print(f"    Users mit Empfehlungen : {len(cf_recs):,}")
    print(f"    Einzigartige Filme     : {n_cf_covered}  "
          f"(von {len(all_movie_ids)} total)")

    # -------------------------------------------------------------------------
    # Ergebnisse speichern
    # TODO: hier später fairness-metriken berechnen, z.B.:
    #   gini_pop  = gini(pop_counts.values)
    #   gini_cf   = gini(cf_counts.values)
    # -------------------------------------------------------------------------
    results.append({
        "split"      : pct_label,
        "bucket"     : bucket,
        "n_ratings"  : n_ratings,
        "n_users"    : len(all_users),
        "pop_recs"   : pop_recs,    # dict {UserID: [MovieID, ...]}
        "pop_counts" : pop_counts,  # Series: MovieID → Empfehlungsanzahl
        "cf_recs"    : cf_recs,
        "cf_counts"  : cf_counts,
    })

print(f"\n\nAlle {len(results)} Splits berechnet.")


# =============================================================================
# 4.  VERIFIKATION
#
# Ziel: sicherstellen dass die Implementierung korrekt funktioniert.
# Führe diesen Block aus und prüfe die Outputs manuell.
# =============================================================================
print("\n" + "=" * 60)
print("VERIFIKATION")
print("=" * 60)

# Letzter berechneter Split für Verifikation
r = results[-1]

# ----- CHECK 1: Grundlegende Zahlen ----------------------------------------
print(f"\n--- CHECK 1: Grundzahlen (Split {r['split']}) ---")
print(f"  Trainings-Ratings  : {r['n_ratings']:,}")
print(f"  User total         : {r['n_users']:,}")
print(f"  User mit Pop-Recs  : {len(r['pop_recs']):,}  ← muss = User total sein")
print(f"  User mit CF-Recs   : {len(r['cf_recs']):,}   ← muss ≈ User total sein")
print(f"  ERWARTUNG: beide Zahlen gleich wie 'User total'")
# CF kann leicht weniger haben wenn ein User sehr wenige Ratings hat

# ----- CHECK 2: Empfehlungslänge pro User ----------------------------------
print(f"\n--- CHECK 2: Empfehlungslänge (sollen alle = {TOP_N} sein) ---")
pop_lengths = [len(v) for v in r["pop_recs"].values()]
cf_lengths  = [len(v) for v in r["cf_recs"].values()]
print(f"  Popularity  Min/Max/Mean: "
      f"{min(pop_lengths)} / {max(pop_lengths)} / {np.mean(pop_lengths):.1f}")
print(f"  CF          Min/Max/Mean: "
      f"{min(cf_lengths)} / {max(cf_lengths)} / {np.mean(cf_lengths):.1f}")
print(f"  ERWARTUNG: alle Werte = {TOP_N}")

# ----- CHECK 3: Coverage (wie viele einzigartige Filme werden empfohlen) ---
print(f"\n--- CHECK 3: Coverage (einzigartige Filme in Empfehlungslisten) ---")
pop_unique = (r["pop_counts"] > 0).sum()
cf_unique  = (r["cf_counts"] > 0).sum()
print(f"  Popularity  : {pop_unique} einzigartige Filme")
print(f"  CF          : {cf_unique} einzigartige Filme")
print(f"  ERWARTUNG:")
print(f"    Popularity = exakt {TOP_N} (alle User bekommen dieselbe Liste)")
print(f"    CF         = viel mehr (jeder User bekommt andere Filme)")

# ----- CHECK 4: Konkrete Empfehlungen für User 1 ---------------------------
print(f"\n--- CHECK 4: Empfehlungen für User 1 (Titel-Vergleich) ---")
sample_user = 1
if sample_user in r["pop_recs"]:
    pop_list = r["pop_recs"][sample_user]
    cf_list  = r["cf_recs"].get(sample_user, [])
    print(f"\n  POPULARITY (gleich für alle User):")
    for i, mid in enumerate(pop_list, 1):
        print(f"    {i:2}. [{mid}] {title_map.get(mid, 'Unbekannt')}")
    print(f"\n  COLLABORATIVE FILTERING (personalisiert für User 1):")
    for i, mid in enumerate(cf_list, 1):
        print(f"    {i:2}. [{mid}] {title_map.get(mid, 'Unbekannt')}")
    overlap = set(pop_list) & set(cf_list)
    print(f"\n  Überschneidung Pop ∩ CF: {len(overlap)} Filme  {overlap if overlap else '(keine)'}")
    print(f"  ERWARTUNG: wenig bis keine Überschneidung (CF ist personalisiert)")

# ----- CHECK 5: Top-10 meistempfohlene Filme (CF vs Popularity) ------------
print(f"\n--- CHECK 5: Top-10 meistempfohlene Filme ---")
pop_top10 = r["pop_counts"].sort_values(ascending=False).head(10)
cf_top10  = r["cf_counts"].sort_values(ascending=False).head(10)

print(f"\n  POPULARITY — Top 10 (alle User × gleiche Liste):")
for mid, cnt in pop_top10.items():
    pct = cnt / r["n_users"] * 100
    print(f"    [{mid}] {title_map.get(mid,'?'):<45} "
          f"empfohlen: {cnt:,} ({pct:.0f}% aller User)")

print(f"\n  CF — Top 10 (nach Empfehlungshäufigkeit):")
for mid, cnt in cf_top10.items():
    pct = cnt / r["n_users"] * 100
    print(f"    [{mid}] {title_map.get(mid,'?'):<45} "
          f"empfohlen: {cnt:,} ({pct:.1f}% aller User)")
print(f"  ERWARTUNG CF: populäre Filme oben, aber viel weniger dominant als Popularity")

# ----- CHECK 6: Keine Doppelempfehlungen pro User --------------------------
print(f"\n--- CHECK 6: Keine Duplikate in Empfehlungslisten ---")
pop_has_dups = any(len(v) != len(set(v)) for v in r["pop_recs"].values())
cf_has_dups  = any(len(v) != len(set(v)) for v in r["cf_recs"].values())
print(f"  Popularity hat Duplikate: {pop_has_dups}  ← muss False sein")
print(f"  CF hat Duplikate        : {cf_has_dups}   ← muss False sein")

# ----- CHECK 7: Rec_counts Summe -------------------------------------------
print(f"\n--- CHECK 7: rec_counts Summe ---")
expected_total = r["n_users"] * TOP_N
pop_total = r["pop_counts"].sum()
cf_total  = r["cf_counts"].sum()
print(f"  Erwartet (n_users × TOP_N): {expected_total:,}")
print(f"  Popularity rec_counts Summe: {pop_total:,}  ← muss = erwartet")
print(f"  CF         rec_counts Summe: {cf_total:,}   ← muss = erwartet")

# ----- ZUSAMMENFASSUNG -------------------------------------------------------
print(f"\n{'='*60}")
print("VERIFIKATION ABGESCHLOSSEN")
print(f"{'='*60}")
print("""
Was du manuell prüfen solltest:
  CHECK 1: Pop und CF decken gleich viele User ab? ✓/✗
  CHECK 2: Alle Listen genau TOP_N lang? ✓/✗
  CHECK 3: Pop = 10 einzigartige Filme, CF = viel mehr? ✓/✗
  CHECK 4: CF-Empfehlungen für User 1 anders als Pop? ✓/✗
  CHECK 5: CF-Top-Filme realistisch (bekannte Filme oben)? ✓/✗
  CHECK 6: Keine Duplikate? ✓/✗
  CHECK 7: Summen stimmen? ✓/✗

Wenn alle 7 Checks passen → Implementierung korrekt.
""")

# =============================================================================
# 5.  NÄCHSTE SCHRITTE (TODO für Fairness-Metriken)
# =============================================================================
print("""
TODO — Fairness-Metriken einfügen:
  from analysis_14march import gini, entropy, l1_norm, aplt_adapted

  Für jeden Split r in results:
    # Provider-Side (wie im Baseline-File, aber auf Empfehlungen)
    gini_pop     = gini(r["pop_counts"].values)
    gini_cf      = gini(r["cf_counts"].values)
    entropy_pop  = entropy(r["pop_counts"].values)
    ...

    # Consumer-Side (braucht User-Infos aus df)
    train_df = df[df["Bucket"] <= r["bucket"]]
    # → merge User-Gender/Age zu Empfehlungen, dann AD_gender etc.
""")