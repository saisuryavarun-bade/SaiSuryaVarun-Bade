"""Exploratory pilot, not a confirmatory publication experiment.

All selectors share each classifier's predictions. Test labels never fit selectors.
Run from any directory: paths resolve relative to this script's project root.
"""
from pathlib import Path
import hashlib
import io
import json
import platform
import sys
import zipfile
from datetime import datetime, timezone

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs' / 'pilot_seed42'
SEED = 42
COVERAGES = [0.5, 0.7, 0.8, 0.9, 0.95, 1.0]
RATES = [0.0, 0.1, 0.3, 0.5]


def load_data():
    path = ROOT / 'data' / 'raw' / 'bank_marketing.zip'
    with zipfile.ZipFile(path) as outer:
        with zipfile.ZipFile(io.BytesIO(outer.read('bank.zip'))) as inner:
            frame = pd.read_csv(inner.open('bank-full.csv'), sep=';')
    original_n = len(frame)
    frame = frame.drop_duplicates().copy()
    removed = original_n - len(frame)
    frame = frame.drop(columns=['duration'])
    y = frame.pop('y').map({'no': 0, 'yes': 1}).to_numpy()
    assert not pd.isna(y).any()
    # Work with object categorical columns so masking never silently changes dtype.
    for col in frame.select_dtypes(exclude='number'):
        frame[col] = frame[col].astype(object).replace('unknown', np.nan)
    numeric = frame.select_dtypes(include='number').columns.tolist()
    frame[numeric] = frame[numeric].astype(float)
    return frame, y, removed, hashlib.sha256(path.read_bytes()).hexdigest()


def corrupt(frame, kind, rate, seed):
    rng = np.random.default_rng(seed)
    out = frame.copy()
    if kind == 'independent':
        mask = rng.random(frame.shape) < rate
    elif kind == 'block':
        mask = np.zeros(frame.shape, dtype=bool)
        mask[:, :len(frame.columns) // 2] = (rng.random(len(frame)) < rate)[:, None]
    else:
        raise ValueError(kind)
    for j, col in enumerate(frame.columns):
        out.loc[out.index[mask[:, j]], col] = np.nan
    return out, float(mask.mean())


def make_classifier(X, name):
    num = X.select_dtypes(include='number').columns.tolist()
    cat = [col for col in X if col not in num]
    pre = ColumnTransformer([
        ('num', Pipeline([('impute', SimpleImputer(strategy='median')),
                          ('scale', StandardScaler())]), num),
        ('cat', Pipeline([('impute', SimpleImputer(strategy='most_frequent')),
                          ('encode', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]), cat)
    ])
    estimator = (LogisticRegression(max_iter=2000, C=1.0) if name == 'logistic'
                 else RandomForestClassifier(n_estimators=150, min_samples_leaf=5,
                                             random_state=SEED, n_jobs=2))
    return Pipeline([('pre', pre), ('model', estimator)])


def meta_features(model, X):
    prob = model.predict_proba(X)[:, 1]
    bounded = np.clip(prob, 1e-12, 1-1e-12)
    entropy = -bounded*np.log(bounded) - (1-bounded)*np.log(1-bounded)
    conf = np.column_stack([np.minimum(prob, 1-prob), prob, entropy])
    masks = X.isna().to_numpy(dtype=float)
    missing = np.column_stack([masks, masks.mean(axis=1)])
    return prob, {'learned_confidence': conf, 'learned_missingness': missing,
                  'combined': np.column_stack([conf, missing])}


def selector_scores(model, selectors, X, seed):
    prob, meta = meta_features(model, X)
    scores = {'raw_confidence': np.minimum(prob, 1-prob),
              'missing_fraction': X.isna().mean(axis=1).to_numpy(),
              'random': np.random.default_rng(seed).random(len(X))}
    for name, selector in selectors.items():
        scores[name] = selector.predict_proba(meta[name])[:, 1]
    return prob, scores


def order_with_ties(score, seed):
    # Random tie order, identical across methods. No class labels used.
    return np.lexsort((np.random.default_rng(seed).random(len(score)), score))


def invariant_checks():
    # Independent checks of the evaluation math, including ties and zero masking.
    errors = np.array([0, 1, 0, 1])
    order = order_with_ties(np.array([0.1, 0.3, 0.2, 0.4]), 1)
    assert errors[order[:2]].mean() == 0
    assert errors[order].mean() == 0.5
    assert np.isclose(np.mean(np.cumsum(errors[order])/np.arange(1, 5)), (0+0+1/3+1/2)/4)
    assert sorted(order_with_ties(np.ones(4), 1)) == [0, 1, 2, 3]
    tiny = pd.DataFrame({'a': [1., 2.], 'b': ['x', 'y']})
    same, rate = corrupt(tiny, 'independent', 0, 1)
    pd.testing.assert_frame_equal(tiny, same)
    assert rate == 0


def main():
    invariant_checks()
    OUT.mkdir(parents=True, exist_ok=True)
    X, y, removed, source_hash = load_data()
    indices = np.arange(len(X))
    train, rest = train_test_split(indices, test_size=0.5, stratify=y, random_state=SEED)
    sel, rest = train_test_split(rest, train_size=0.4, stratify=y[rest], random_state=SEED+1)
    val, test = train_test_split(rest, train_size=1/3, stratify=y[rest], random_state=SEED+2)
    groups = [set(z) for z in [train, sel, val, test]]
    assert sum(map(len, groups)) == len(X)
    assert all(not groups[i] & groups[j] for i in range(4) for j in range(i+1, 4))
    splits = {'classifier_train': train.tolist(), 'selector_train': sel.tolist(),
              'threshold_validation': val.tolist(), 'exploratory_test': test.tolist()}
    (OUT / 'splits.json').write_text(json.dumps(splits), encoding='utf-8')
    manifest = {'utc_started': datetime.now(timezone.utc).isoformat(), 'seed': SEED,
                'python': sys.version, 'platform': platform.platform(),
                'versions': {'numpy': np.__version__, 'pandas': pd.__version__,
                             'sklearn': sklearn.__version__, 'matplotlib': matplotlib.__version__},
                'source_url': 'https://archive.ics.uci.edu/static/public/222/bank+marketing.zip',
                'source_sha256': source_hash, 'rows': len(X), 'exact_duplicates_removed': removed,
                'excluded_features': ['duration'], 'features': X.columns.tolist(),
                'block_features': X.columns[:len(X.columns)//2].tolist(),
                'partition_sizes': {k: len(v) for k, v in splits.items()},
                'split_sha256': hashlib.sha256(json.dumps(splits).encode()).hexdigest(),
                'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'status': 'exploratory; not confirmatory; no human-review study'}
    (OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    rows, fixed_rows, full_rows, predictions = [], [], [], []
    plot_curves = []
    for model_index, name in enumerate(['logistic', 'random_forest']):
        print(f'Fitting {name}', flush=True)
        model = make_classifier(X.iloc[train], name).fit(X.iloc[train], y[train])
        features_by_method = {key: [] for key in ['learned_confidence', 'learned_missingness', 'combined']}
        labels = []
        for i, rate in enumerate(RATES):
            damaged, _ = corrupt(X.iloc[sel], 'independent', rate, 1000+i)
            p, meta = meta_features(model, damaged)
            labels.append(((p >= 0.5) != y[sel]).astype(int))
            for key in features_by_method:
                features_by_method[key].append(meta[key])
        err_labels = np.concatenate(labels)
        selectors = {}
        for key, pieces in features_by_method.items():
            selector = Pipeline([('scale', StandardScaler()),
                                 ('model', LogisticRegression(max_iter=2000, C=1.0))])
            selectors[key] = selector.fit(np.vstack(pieces), err_labels)
        # Validation-derived thresholds are frozen across every test mechanism.
        val_scores = {}
        for i, rate in enumerate(RATES):
            damaged, _ = corrupt(X.iloc[val], 'independent', rate, 2000+i)
            _, score = selector_scores(model, selectors, damaged, 3000+i)
            for key, values in score.items():
                val_scores.setdefault(key, []).append(values)
        thresholds = {key: float(np.quantile(np.concatenate(pieces), 0.8))
                      for key, pieces in val_scores.items()}
        conditions = [('independent', rate) for rate in RATES] + [('block', 0.3), ('block', 0.5)]
        for condition_index, (kind, rate) in enumerate(conditions):
            # Corruptions and tie seeds are shared across classifiers and selectors.
            damaged, added = corrupt(X.iloc[test], kind, rate, 4000+condition_index)
            prob, scores = selector_scores(model, selectors, damaged, 5000+condition_index)
            errors = ((prob >= 0.5) != y[test]).astype(int)
            condition = f'{kind}_{rate:.1f}'
            base = {'model': name, 'condition': condition, 'added_mask_fraction': added,
                    'total_missing_fraction': float(damaged.isna().to_numpy().mean())}
            full_rows.append({**base, 'error': float(errors.mean()),
                              'roc_auc': roc_auc_score(y[test], prob),
                              'average_precision': average_precision_score(y[test], prob),
                              'brier': brier_score_loss(y[test], prob),
                              'log_loss': log_loss(y[test], np.column_stack([1-prob, prob]))})
            pred = pd.DataFrame({'source_index': X.index[test], 'label': y[test],
                                 'probability': prob, 'error': errors,
                                 'missing_fraction': damaged.isna().mean(axis=1).to_numpy()})
            pred['model'], pred['condition'] = name, condition
            for key, score in scores.items():
                pred[f'score_{key}'] = score
                order = order_with_ties(score, 6000+condition_index)
                rc_risk = np.cumsum(errors[order])/np.arange(1, len(order)+1)
                aurc = float(rc_risk.mean())
                for coverage in COVERAGES:
                    count = max(1, int(np.ceil(coverage*len(test))))
                    keep = order[:count]
                    positives = y[test] == 1
                    negatives = ~positives
                    rows.append({**base, 'selector': key, 'target_coverage': coverage,
                                 'actual_coverage': count/len(test), 'retained_n': count,
                                 'selective_risk': float(errors[keep].mean()), 'aurc': aurc,
                                 'positive_retention': float(positives[keep].sum()/positives.sum()),
                                 'negative_retention': float(negatives[keep].sum()/negatives.sum())})
                keep = score <= thresholds[key]
                fixed_rows.append({**base, 'selector': key, 'threshold': thresholds[key],
                                   'achieved_coverage': float(keep.mean()),
                                   'selective_risk': float(errors[keep].mean()) if keep.any() else None})
                if kind == 'block' and rate == 0.5:
                    plot_curves.append((name, key, np.arange(1, len(order)+1)/len(order), rc_risk))
            predictions.append(pred)
        print(f'Finished {name}', flush=True)
    result = pd.DataFrame(rows)
    result.to_csv(OUT / 'selective_metrics.csv', index=False)
    pd.DataFrame(fixed_rows).to_csv(OUT / 'fixed_threshold_metrics.csv', index=False)
    pd.DataFrame(full_rows).to_csv(OUT / 'classifier_metrics.csv', index=False)
    pd.concat(predictions, ignore_index=True).to_csv(OUT / 'predictions.csv', index=False)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for ax, model_name in zip(axes, ['logistic', 'random_forest']):
        for fitted_name, key, cover, risk in plot_curves:
            if fitted_name == model_name:
                ax.plot(cover, risk, label=key, linewidth=1)
        ax.set(xlim=(0.1, 1), xlabel='Fraction retained', ylabel='Error among retained cases',
               title=f'{model_name}: exploratory block outage')
        ax.grid(alpha=0.2)
    axes[1].legend(fontsize=7)
    fig.suptitle('One dataset, one seed; no uncertainty estimates or publication claims')
    fig.tight_layout()
    fig.savefig(OUT / 'risk_coverage.png', dpi=180)
    plt.close(fig)
    primary = result[np.isclose(result.target_coverage, 0.8)].pivot(
        index=['model', 'condition'], columns='selector', values='selective_risk')
    primary.to_csv(OUT / 'risk_at_80_percent.csv')
    print(primary.round(4).to_string(), flush=True)
    assert result[result.target_coverage == 1].groupby(['model', 'condition']).selective_risk.nunique().eq(1).all()
    print('Invariants passed. Pilot outputs:', OUT, flush=True)


if __name__ == '__main__':
    main()
