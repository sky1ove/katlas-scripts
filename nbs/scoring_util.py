"""Shared helpers for the scoring benchmark (`scoring_*.py`; formerly `05_scoring/eval16.py`).

Everything the scripts need to build a split, score a branch, and turn raw per-pair rows into
metrics. The invariant of the whole module: **each script persists only the raw per-pair table**
(`score_pairs`), and every metric — any cut, any grouping — is derived from it with `summarize`.
A new metric therefore never requires re-scoring.

pairs schema: [method, seed, branch, site_seq, kinase, kinase_group, num_kin, rank, ap, n_pool]
(+ whatever the script tags on, e.g. `window`, `split`, `train_numkin`)

Artifacts keep their historical `16_` / `18_` prefixes — the script files were renumbered in the
port, the files on disk were not. See SCRIPTS.md for the mapping.
"""
import copy
from pathlib import Path

import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from kplot.ranking import get_AUCDF  # noqa: F401  (kept for parity / optional use)

NBS = Path(__file__).resolve().parent
OUT = NBS / 'out'
RES = OUT / 'scoring_pairs'        # raw per-pair tables
PCT = OUT / 'pct_ref'         # percentile references (human-background scores)

SEEDS = [0, 1, 2]   # legacy overlapping 20% draws — kept for the window (02*) and num_kin (03a) analyses
N_FOLDS = 5         # HEADLINE (04*): RepeatedStratifiedGroupKFold K
N_REPEATS = 3       # HEADLINE (04*): RepeatedStratifiedGroupKFold repeats — the CI unit (SD across these)
BASE_NUMKIN = 40    # site-promiscuity cutoff for training / CDDM
MIN_SITES = 40      # kinases with at least this many sites define the fixed pool
POOL_COUNT = 20     # min TRAIN sites to build a pool kinase's PSSM (< MIN_SITES so the pool is covered)
NK_MAIN = 10        # MAIN RESULT: evaluate on the test sites with num_kin <= this


def repeated_group_folds(df, n_folds=N_FOLDS, n_repeats=N_REPEATS,
                         label='kinase_protein', group='site_seq_upper'):
    """HEADLINE split: RepeatedStratifiedGroupKFold fold assignment. Returns a DataFrame with columns
    `fold_0..fold_{R-1}`, each row's held-out fold index (0..K-1) within that repeat.

    Within a repeat the K folds tile the data (every row tested exactly once, folds non-overlapping);
    the R repeats are independent re-shuffles, so `fold_r` is one full leak-free K-fold CV and the R
    of them give a reproducibility spread (SD across repeats). Leak-free: whole `group` values (uppercase
    backbones) stay in one fold. sklearn has no RepeatedStratifiedGroupKFold, so we loop the grouped
    splitter with a per-repeat seed. This supersedes the three OVERLAPPING 20% draws (`test_0/1/2`) for
    the main result — those draws are not independent, so a CI over them is not robust."""
    from sklearn.model_selection import StratifiedGroupKFold
    d = df.reset_index(drop=True)
    out = pd.DataFrame(index=d.index)
    for r in range(n_repeats):
        assign = np.full(len(d), -1, dtype=np.int8)
        for k, (_, te) in enumerate(StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=r)
                                    .split(d, d[label], groups=d[group])):
            assign[te] = k
        assert (assign >= 0).all(), f'repeat {r}: {int((assign < 0).sum())} rows unassigned'
        out[f'fold_{r}'] = assign
    return out


# ---------- artifacts ----------
def split_path():
    return OUT / 'scoring_split.parquet'


def load_split():
    "The full site table with one boolean `test_<seed>` column per seed (built by scoring_01)."
    return pd.read_parquet(split_path())


def cddm_path(seed):
    return OUT / f'scoring_cddm_seed{seed}.parquet'


def onehot_cols(window=None):
    "The one-hot `{pos}{aa}` key space, optionally narrowed to positions -window..+window."
    cols = pd.read_parquet(cddm_path(0)).columns
    return list(cols) if window is None else [c for c in cols if abs(int(c[:-1])) <= window]


def win(ref, window):
    "Restrict a flat PSSM's columns to positions -window..+window."
    return ref[[c for c in ref.columns if abs(int(c[:-1])) <= window]]


# ---------- data ----------
def branch_split(df):
    "Split a site table into (ST, Tyr) by kinase class + central acceptor residue."
    st  = df[(df.kinase_group != 'TK') & (df.site_seq.str[20] != 'y')]
    tyr = df[(df.kinase_group == 'TK') & (df.site_seq.str[20] == 'y')]
    return st, tyr


def site_kinase_map(split):
    "site_seq -> set of all true kinases in the whole dataset (for multi-label MAP)."
    return split.groupby('site_seq')['kinase_protein'].agg(set).to_dict()


def iter_folds(split):
    """HEADLINE evaluation loop: yield (repeat, fold, train_df, test_df) over the RepeatedSGKFold
    columns `fold_<repeat>`. The pairs stamp `seed = repeat`, so within a repeat the K folds pool to
    cover every site once (one full-CV estimate) and the R repeats give the reproducibility SD — the
    unit `scoring_04d` aggregates over. Train for a fold is the other K-1 folds (leak-free by
    construction)."""
    for r in range(N_REPEATS):
        col = split[f'fold_{r}']
        for k in range(N_FOLDS):
            te = col == k
            yield r, k, split[~te], split[te]


def load_pool(path=None):
    """The FIXED candidate pool, defined once in 16a and CONSTANT across seeds (avoids the per-seed
    `pspa ∩ cddm` drift where a kinase's train split randomly has <40 sites). Returns
    (pools, group_map): pools = {'ST': [...], 'Tyr': [...]} by kinase class (TK -> Tyr, else ST),
    group_map = {kinase -> kinase_group}. Branch assignment is from kinase_info, NOT per-seed test
    rows, so a kinase with zero test sites in a seed is never misfiled into the wrong branch."""
    df = pd.read_parquet(path or OUT / 'scoring_pool.parquet')
    pools = {'ST': df.loc[df.branch == 'ST', 'kinase'].tolist(),
             'Tyr': df.loc[df.branch == 'Tyr', 'kinase'].tolist()}
    return pools, dict(zip(df.kinase, df.kinase_group))


def strat_group_split(df, frac=0.2, seed=0, label='kinase_protein', group='site_seq_upper'):
    """Boolean hold-out mask (positional, len == len(df)) for a STRATIFIED GROUP split: whole `group` values
    (uppercase backbones) go entirely to one side → no sequence shared train/test (leak-free), while each
    `label` (kinase) keeps ~`frac` of its rows held out (StratifiedGroupKFold). Used for the 16a train/test
    split and the 16g/16h/16i inner train/validation carve, so both are leak-free AND representative."""
    from sklearn.model_selection import StratifiedGroupKFold
    d = df.reset_index(drop=True)
    _, idx = next(StratifiedGroupKFold(n_splits=int(round(1 / frac)), shuffle=True, random_state=seed)
                  .split(d, d[label], groups=d[group]))
    m = np.zeros(len(d), bool); m[idx] = True
    return m


# ---------- percentile references / PSSM building ----------
_human = None


def human_bg():
    "Human phosphoproteome background, deduplicated on the uppercase sequence (lazy, cached)."
    global _human
    if _human is None:
        import kdata
        h = kdata.load('human_site')
        h['site_seq_upper'] = h.site_seq.str.upper()
        _human = h.drop_duplicates('site_seq_upper').reset_index(drop=True)
        print('human background:', _human.shape)
    return _human


def get_ref(df, pssm, seq_col, func, n_chunk=5):
    "Score a background df against `pssm` in chunks -> percentile reference."
    from katlas.scoring import predict_kinase_df
    chunk = (len(df) + n_chunk - 1) // n_chunk
    out = []
    for i in range(n_chunk):
        a, b = i * chunk, min((i + 1) * chunk, len(df))
        if a >= b:
            break
        out.append(predict_kinase_df(df.iloc[a:b], seq_col=seq_col, ref=pssm, func=func))
    return pd.concat(out, ignore_index=True)


def cached_ref(label, pssm, seq_col, func, force=True):
    "Percentile reference for `pssm`, cached at PCT/<label>.parquet. force=True always recomputes."
    PCT.mkdir(parents=True, exist_ok=True)
    path = PCT / f'{label}.parquet'
    if force or not path.exists():
        print('  compute pct ref', label)
        get_ref(human_bg(), pssm, seq_col, func).to_parquet(path)
    return pd.read_parquet(path)


def build_pssm(df, pool_names, seq_col='site_seq', count_thr=POOL_COUNT):
    """CDDM PSSM over `df`, reindexed to the FIXED pool so the kinase set is constant across seeds.

    `count_thr` is deliberately below MIN_SITES: a pool kinase whose train split happens to have
    32-40 sites still gets a PSSM instead of dropping out that seed.
    """
    from katlas.pssm import get_cluster_pssms
    p = get_cluster_pssms(df, cluster_col='kinase_id', seq_col=seq_col,
                          count_thr=count_thr, valid_thr=None)
    p.index = p.index.str.split('_').str[1]
    p = p[~p.index.duplicated()]
    return p.reindex(pool_names)


def class_balance(kinases, pool):
    "Class-balance CE weights proportional to 1/sqrt(n_k), normalised to mean 1."
    nk = pd.Series(kinases).value_counts().reindex(pool).fillna(0).to_numpy() + 1.0
    w = 1 / np.sqrt(nk)
    return (w / w.mean()).astype(np.float32)


# ---------- features / scoring ----------
def onehot(seqs, cols, eps_center=20):
    "One-hot of 41-mers in a (pos+aa) key space -> [n, len(cols)]."
    cidx = {c: i for i, c in enumerate(cols)}; pos = sorted({int(c[:-1]) for c in cols})
    arr = np.array([list(s) for s in seqs]); X = np.zeros((len(seqs), len(cols)), np.float32)
    for p in pos:
        j = np.array([cidx.get(f'{p}{a}', -1) for a in arr[:, p + eps_center]]); v = j >= 0
        X[np.arange(len(seqs))[v], j[v]] = 1.0
    return X


def gen_scores(df_b, ref, func, pool, EPS=1e-4):
    "Generative PSSM score matrix [n_sites x len(pool)] for a branch (equal-weight multiply)."
    from katlas.scoring import predict_kinase_df
    r = predict_kinase_df(df_b, seq_col='site_seq', ref=ref[ref.index.isin(pool)], func=func)
    return r.reindex(columns=pool).to_numpy(np.float32)


def make_mlp(D, K, hidden=512, depth=2, dropout=0.3):
    if hidden is None: return nn.Linear(D, K)
    L = [nn.Linear(D, hidden), nn.ReLU(), nn.Dropout(dropout)]
    for _ in range(depth - 1): L += [nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout)]
    return nn.Sequential(*L, nn.Linear(hidden, K))


def otk_mask(seqs, y, pool, site_kin):
    "Bool [N, len(pool)]: True at each row's site's OTHER true kinases -> excluded from the softmax denominator "
    "(multi-label masked-CE: don't penalise a promiscuous site's co-true kinases as negatives)."
    pidx = {k: i for i, k in enumerate(pool)}
    m = np.zeros((len(seqs), len(pool)), bool)
    for i, (s, yy) in enumerate(zip(seqs, y)):
        idx = [pidx[k] for k in site_kin.get(s, ()) if k in pidx and pidx[k] != yy]
        if idx: m[i, idx] = True
    return m


def val_carve(n, y, groups, val=0.2, seed=0):
    """(train_idx, val_idx) for the internal early-stop split, with NO backbone shared across the
    two — the leak-free counterpart of a plain random carve. StratifiedGroupKFold (stratified +
    grouped, the same splitter as the main train/test split via strat_group_split); a
    GroupShuffleSplit fallback if a rare class trips it. Both are leak-free, so early stopping
    never scores itself on a backbone the fit already saw."""
    import warnings
    from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
    ns = max(2, int(round(1 / val)))
    with warnings.catch_warnings():                            # rare class < n_splits -> warning, not error
        warnings.simplefilter('ignore')
        try:
            return next(StratifiedGroupKFold(n_splits=ns, shuffle=True, random_state=seed)
                        .split(np.zeros(n), y, groups))
        except ValueError:
            return next(GroupShuffleSplit(n_splits=1, test_size=val, random_state=seed)
                        .split(np.zeros(n), y, groups))


def fit_mlp(X, y, K, hidden=512, depth=2, pspa=None, epochs=120, bs=1024, lr=1e-2, wd=1e-3, val=0.2, seed=0,
            mask=None, class_weight=None, logit_prior=None, groups=None):
    "Train a classifier (optionally + γ·PSPA channel). mask = bool [N,K] of a site's OTHER true kinases to drop "
    "from the softmax (masked-CE, via otk_mask); early-stop on a val slice. `groups` = per-row backbone "
    "(uppercase site_seq): when given, the early-stop val is a leak-free grouped carve (no backbone shared "
    "with the fit) instead of a plain random slice. Imbalance handling (both optional): "
    "class_weight = [K] per-class CE weights; logit_prior = [K] added to logits at TRAIN time only (balanced "
    "softmax — pass log(n_k); inference via mlp_scores is unadjusted)."
    torch.manual_seed(seed); n = len(y)
    if groups is None:
        rng = np.random.default_rng(seed); perm = rng.permutation(n); nv = int(n * val); ti, vi = perm[nv:], perm[:nv]
    else:
        ti, vi = val_carve(n, y, np.asarray(groups), val=val, seed=seed)
    Xt, yt = torch.tensor(X), torch.tensor(y); PS = torch.tensor(pspa) if pspa is not None else None
    Mt = torch.tensor(mask) if mask is not None else None
    CW = torch.tensor(class_weight, dtype=torch.float32) if class_weight is not None else None
    LP = torch.tensor(logit_prior, dtype=torch.float32) if logit_prior is not None else None
    model = make_mlp(X.shape[1], K, hidden, depth)
    gamma = torch.zeros(1, requires_grad=True) if pspa is not None else None
    params = list(model.parameters()) + ([gamma] if gamma is not None else [])
    opt = torch.optim.Adam(params, lr=lr, weight_decay=wd)
    def logit(idx, tr):
        model.train(tr); z = model(Xt[idx])
        if PS is not None: z = z + gamma * PS[idx]
        if Mt is not None: z = z.masked_fill(Mt[idx], -1e9)     # masked-CE: co-true kinases out of the denominator
        if LP is not None: z = z + LP                          # balanced-softmax prior (TRAIN+val loss only; not at inference)
        return z
    best = (1e9, None)
    for ep in range(epochs):
        order = ti[np.random.default_rng(seed + ep).permutation(len(ti))]
        for s in range(0, len(order), bs):
            b = order[s:s + bs]; opt.zero_grad(); F.cross_entropy(logit(b, True), yt[b], weight=CW).backward(); opt.step()
        with torch.no_grad():
            vl = F.cross_entropy(logit(vi, False), yt[vi], weight=CW).item()
            if vl < best[0]: best = (vl, (copy.deepcopy(model.state_dict()), None if gamma is None else gamma.detach().clone()))
    model.load_state_dict(best[1][0]); model.eval()
    return model, best[1][1]


def mlp_scores(model, X, gamma=None, pspa=None):
    with torch.no_grad():
        z = model(torch.tensor(X))
        if gamma is not None: z = z + gamma * torch.tensor(pspa)
    return z.numpy()


# ---------- metrics ----------
def _ap(rs, rel):
    "Average precision for one site. rs=score vector over pool; rel=relevant kinase indices."
    rr = np.sort((rs[None, :] > rs[rel][:, None]).sum(1) + 1)
    return float(np.mean([(j + 1) / r for j, r in enumerate(rr)]))


def score_pairs(df_b, scores, pool, site_kin, method, seed, branch):
    """RAW layer: one row per (test site, its true kinase) — rank in the pool, per-site AP, and
    num_kin / kinase_group / pool-size tags. This is the only thing the experiment notebooks persist;
    every metric (any cut, any grouping) is derived from it via `summarize`, so a new metric never needs re-scoring.

    rank = position of the true kinase among `pool` for that site (1 = best). AP uses the site's FULL
    true-kinase set (multi-label); it is identical across a site's pairs, so MAP dedups on site_seq.
    """
    pidx = {k: i for i, k in enumerate(pool)}; K = len(pool)
    df = df_b.reset_index(drop=True)
    # A NaN score means "no prediction" (e.g. a pool kinase absent from a ref -> reindex fills NaN, or a
    # site with no scorable positions). NaN must rank WORST, not best: `NaN > x` is False, so an un-sanitized
    # NaN true-kinase score would otherwise get rank 1. Map NaN -> -inf so it ranks last and never outranks a real score.
    scores = np.where(np.isnan(scores), -np.inf, np.asarray(scores, np.float64))
    y = np.array([pidx[k] for k in df.kinase_protein])
    ranks = (scores > scores[np.arange(len(y)), y][:, None]).sum(1) + 1
    ap = np.full(len(df), np.nan)
    for i in range(len(df)):
        rel = [pidx[k] for k in site_kin.get(df.site_seq.iat[i], ()) if k in pidx]
        if rel: ap[i] = _ap(scores[i], np.array(rel))
    return pd.DataFrame({'method': method, 'seed': seed, 'branch': branch,
                         'site_seq': df.site_seq.values, 'kinase': df.kinase_protein.values,
                         'kinase_group': df.kinase_group.values, 'num_kin': df.num_kin.values,
                         'rank': ranks.astype(np.int32), 'ap': ap, 'n_pool': np.int32(K)})


# ---------- aggregation (used across scoring_02–12) ----------
METRICS = ['AUCDF', 'top1', 'top5', 'top10', 'MRR', 'MAP']

def _block(g):
    "Metrics for one group of pairs: top-k/MRR/AUCDF over PAIRS, MAP over UNIQUE sites."
    r = g['rank'].to_numpy(); K = int(g['n_pool'].iloc[0])
    return pd.Series({'n_pairs': len(g), 'n_sites': g['site_seq'].nunique(),
                      'AUCDF': (K - r.mean()) / (K - 1), 'top1': (r == 1).mean(), 'top5': (r <= 5).mean(),
                      'top10': (r <= 10).mean(), 'MRR': (1.0 / r).mean(),
                      'MAP': g.drop_duplicates('site_seq')['ap'].mean()})


def summarize(pairs, keys):
    """Aggregate the per-pair table to `keys`. Micro by construction (each pair equal; MAP each unique site
    equal). Works for any grouping — overall ['method','seed','branch'], per group (+['kinase_group']),
    per kinase (+['kinase']) — and any pre-filtered subset, e.g. summarize(pairs[pairs.num_kin<=10], keys)."""
    return pairs.groupby(keys, observed=True).apply(_block, include_groups=False).reset_index()


# ---------- confidence intervals (bootstrap; the biological replicate is the KINASE) ----------
# The benchmark's independent unit is the kinase, NOT the test site (pseudoreplicated – one kinase
# owns many sites) and NOT the CV fold (folds of one partition are not independent experiments). So
# every CI resamples KINASES with replacement (a cluster bootstrap), which is also the unit the
# per-kinase paired Wilcoxon already uses. MAP is the exception: it is a per-site metric, so it is
# resampled over unique sites. All of this derives from the persisted per-pair table – never a re-score.
# The generic 1-D `boot_ci` / `fmt_ci` live in stats_util (shared across modules); re-exported here so
# `su.boot_ci` keeps working. `boot_micro_ci` is scoring-specific (it knows the per-pair schema).
from stats_util import DEFAULT_B as N_BOOT  # noqa: E402
from stats_util import boot_ci, fdr_bh, fmt_ci, mwu_report, wilcoxon_report  # noqa: E402,F401  (re-exported)

def boot_micro_ci(pairs_mb, B=N_BOOT, seed=0):
    """95% percentile CIs for one (method, branch) leaderboard row, each metric bootstrapped over its
    natural unit. Rank metrics (recall@k / AUCDF / MRR) are a CLUSTER BOOTSTRAP OVER KINASES via
    per-kinase sufficient statistics, pooled over the 3 repeats – exact, because within a repeat the
    folds tile the data so every site is tested once, giving each repeat equal weight (pooling == the
    leaderboard's average-over-repeats). MAP is a per-site metric, so it is resampled over unique sites
    (a site's AP averaged over repeats first). Pure-numpy resampling on the sufficient statistics, so
    B=2000 draws are cheap and it never re-scores. Returns {metric: (lo, hi)} for the metrics present."""
    out = {}
    d = pairs_mb
    if len(d):
        r = d['rank'].to_numpy(np.float64)
        K = int(d['n_pool'].iloc[0])
        # per-kinase sums of the sufficient statistics (cnt, hits@1/5/10, Σrank, Σ1/rank)
        stat = (pd.DataFrame({'kinase': d['kinase'].to_numpy(), 'cnt': 1.0,
                              'h1': r == 1, 'h5': r <= 5, 'h10': r <= 10, 'sr': r, 'rr': 1.0 / r})
                .groupby('kinase', observed=True)[['cnt', 'h1', 'h5', 'h10', 'sr', 'rr']].sum().to_numpy())
        nk = len(stat)
        if nk >= 3:
            rng = np.random.default_rng(seed)
            S = stat[rng.integers(0, nk, (B, nk))].sum(1)          # [B, 6]: sums over resampled kinases
            cnt = S[:, 0]
            draws = {'top1': S[:, 1] / cnt, 'top5': S[:, 2] / cnt, 'top10': S[:, 3] / cnt,
                     'MRR': S[:, 5] / cnt, 'AUCDF': (K - S[:, 4] / cnt) / (K - 1)}
            out.update({m: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))
                        for m, v in draws.items()})
        ap = d.dropna(subset=['ap']).groupby('site_seq')['ap'].mean().to_numpy()   # per-site, over repeats
        if len(ap) >= 3:
            rng = np.random.default_rng(seed + 1)
            bs = np.array([ap[rng.integers(0, len(ap), len(ap))].mean() for _ in range(B)])
            out['MAP'] = (float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)))
    return out


def boot_macro_ci(vals, B=N_BOOT, seed=0):
    "95% percentile CI for a macro metric (mean over kinases): resample KINASES with replacement, re-average."
    v = np.asarray(vals, np.float64)
    if len(v) < 3:
        return (float('nan'), float('nan'))
    rng = np.random.default_rng(seed)
    bs = v[rng.integers(0, len(v), (B, len(v)))].mean(1)
    return float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def window_summary(pairs):
    "Per (split, window, seed, branch): AUCDF, micro recall@10, and macro recall@10 over groups."
    ov = summarize(pairs, ['split', 'window', 'seed', 'branch'])
    grp = summarize(pairs, ['split', 'window', 'seed', 'branch', 'kinase_group'])
    macro = (grp.groupby(['split', 'window', 'seed', 'branch'], as_index=False)['top10']
                .mean().rename(columns={'top10': 'top10_macro'}))
    return ov.merge(macro, on=['split', 'window', 'seed', 'branch'])


def select_window(pairs, what):
    """Window with the best micro recall@10 on the VALIDATION split, reported for the record.

    Selection is on validation, never on test: picking a hyperparameter on the set you then
    report is circular. The test curve is printed alongside purely as a generalization check.
    The chosen window is printed, not written to a file: the scorers (04b/04c) fix +/-5 directly.
    """
    val = summarize(pairs[pairs.split == 'val'], ['window', 'branch', 'seed']).groupby('window').top10.mean()
    test = summarize(pairs[pairs.split == 'test'], ['window', 'branch', 'seed']).groupby('window').top10.mean()
    best = val.idxmax()
    print(f'selected {what} window = {best} (VALIDATION argmax micro recall@10)')
    print('  val :', {k: round(v, 3) for k, v in val.items()})
    print('  test:', {k: round(v, 3) for k, v in test.items()})
    return best


PANELS = [('AUCDF', 'AUCDF'), ('top10', 'micro recall@10'),
          ('top10_macro', 'macro recall@10 (over group)')]


def plot_window_sweep(pairs, out_svg, suptitle, xlabel, order=None, xticks=None):
    """Two rows of three panels: VALIDATION on top (what selects the window), TEST below
    (generalization). `order` gives a categorical window order; otherwise windows are numeric."""
    from matplotlib import pyplot as plt
    from kplot.utils import save_svg, set_sns
    set_sns()

    d = window_summary(pairs)
    cols = [c for c, _ in PANELS]
    fig, axes = plt.subplots(2, 3, figsize=(16, 8.5), sharex=True)
    for ri, sp in enumerate(['val', 'test']):
        g = d[d.split == sp].groupby(['branch', 'window'])
        mean, std = g[cols].mean(), g[cols].std()
        for ax, (col, title) in zip(axes[ri], PANELS):
            for br, mk in [('ST', '-o'), ('Tyr', '--s')]:
                mm = mean.loc[br][col] if order is None else mean.loc[br].reindex(order)[col]
                ss = std.loc[br][col] if order is None else std.loc[br].reindex(order)[col]
                x = mm.index if order is None else range(len(order))
                ax.errorbar(x, mm.values, yerr=ss.values, fmt=mk, capsize=3, label=br)
            ax.set_title(f'{sp.upper()} — {title}')
            if order is None:
                ax.set_xticks(xticks if xticks is not None else sorted(d.window.unique()))
            else:
                ax.set_xticks(range(len(order)))
                ax.set_xticklabels(order, rotation=20)
            ax.grid(alpha=.3)
            if col == 'AUCDF':
                ax.set_ylabel(sp.upper())
                ax.legend(fontsize=8)
    axes[1][1].set_xlabel(xlabel)
    fig.suptitle(suptitle)
    plt.tight_layout()
    save_svg(out_svg)
    plt.close('all')
    print('  wrote', out_svg)

    look = d[(d.branch == 'ST') & (d.split == 'val')].groupby('window')[cols].mean()
    print(' ST validation:\n', (look if order is None else look.reindex(order)).round(3).to_string())


def per_kinase_metrics(df_b, scores, pool, site_kin, method, seed, branch):
    "Back-compat shim (used by 16e): per-kinase table + per-unique-site AP table, derived from score_pairs."
    pairs = score_pairs(df_b, scores, pool, site_kin, method, seed, branch)
    kin_df = summarize(pairs, ['method', 'seed', 'branch', 'kinase', 'kinase_group'])
    site_df = (pairs.dropna(subset=['ap']).drop_duplicates('site_seq')[['site_seq', 'ap', 'method', 'seed', 'branch']])
    return kin_df, site_df
