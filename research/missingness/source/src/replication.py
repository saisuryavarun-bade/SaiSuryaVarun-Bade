"""Exploratory replication v0.2. See replication_plan.md before interpreting."""
import hashlib
import json
import platform
import sys
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import sklearn
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, log_loss
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler, FunctionTransformer
from threadpoolctl import threadpool_limits

from pilot import ROOT, load_data, meta_features, order_with_ties, invariant_checks

OUT = ROOT / 'outputs' / 'replication_v02'
SEEDS = [17, 42, 83]
MODELS = ['logistic', 'logistic_mask_augmented', 'native_boost_augmented', 'native_boost_calibrated']


def inject(X, kind, rate, seed):
    rng = np.random.default_rng(seed)
    mask = np.zeros(X.shape, dtype=bool)
    if kind == 'independent':
        mask = rng.random(X.shape) < rate
    else:
        middle = len(X.columns)//2
        cols = np.arange(middle) if kind == 'block_a' else np.arange(middle, len(X.columns))
        row_probability = rate/(len(cols)/len(X.columns))
        assert 0 <= row_probability <= 1
        mask[:, cols] = (rng.random(len(X)) < row_probability)[:, None]
    out = X.copy()
    for j, col in enumerate(X.columns):
        out.loc[out.index[mask[:, j]], col] = np.nan
    return out, float(mask.mean()), float((mask & ~X.isna().to_numpy()).mean())


def to_mask(X):
    return pd.isna(X).astype(float)


class ResearchClassifier(ClassifierMixin, BaseEstimator):
    """Augment only inside fit, keeping copied rows within calibration folds."""
    def __init__(self, kind='logistic', seed=42):
        self.kind = kind
        self.seed = seed

    def fit(self, X, y):
        num = X.select_dtypes(include='number').columns.tolist()
        cat = [c for c in X if c not in num]
        if self.kind.startswith('native'):
            transforms = [('num', 'passthrough', num)]
            if cat:
                transforms.append(('cat', OrdinalEncoder(handle_unknown='use_encoded_value',
                    unknown_value=-1, encoded_missing_value=np.nan), cat))
            self.pre_ = ColumnTransformer(transforms)
            estimator = HistGradientBoostingClassifier(max_iter=100, max_leaf_nodes=15,
                l2_regularization=1, random_state=self.seed, early_stopping=False)
        else:
            transforms = [('num', Pipeline([('impute', SimpleImputer(strategy='median')),
                                            ('scale', StandardScaler())]), num)]
            if cat:
                transforms.append(('cat', Pipeline([('impute', SimpleImputer(strategy='most_frequent')),
                    ('encode', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]), cat))
            if self.kind == 'logistic_mask_augmented':
                transforms.append(('mask', FunctionTransformer(to_mask), X.columns.tolist()))
            self.pre_ = ColumnTransformer(transforms)
            estimator = LogisticRegression(C=1, max_iter=2000)
        self.pre_.fit(X, y)
        self.classes_ = np.unique(y)
        self.n_features_in_ = X.shape[1]
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        if self.kind != 'logistic':
            damaged, _, _ = inject(X, 'independent', .2, self.seed+100)
            train_X, train_y = pd.concat([X, damaged]), np.tile(np.asarray(y), 2)
        else:
            train_X, train_y = X, y
        self.model_ = estimator.fit(self.pre_.transform(train_X), train_y)
        return self

    def predict_proba(self, X):
        return self.model_.predict_proba(self.pre_.transform(X))

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


def load_named(name):
    if name == 'bank':
        X, y, duplicates, source_hash = load_data()
        return X, y, {'source': 'https://doi.org/10.24432/C5K306', 'sha256': source_hash,
                      'duplicates_removed': duplicates, 'positive_label': 'term deposit subscription'}
    bundle = load_breast_cancer(as_frame=True)
    X = bundle.data.astype(float)
    y = (bundle.target.to_numpy() == 0).astype(int)  # malignant positive
    full = X.assign(_target=y)
    keep = ~full.duplicated()
    X, y = X.loc[keep].copy(), y[keep]
    source_hash = hashlib.sha256(full.to_csv(index=False).encode()).hexdigest()
    return X, y, {'source': 'https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic',
                  'bundled_in': 'scikit-learn', 'export_sha256': source_hash,
                  'duplicates_removed': int((~keep).sum()), 'positive_label': 'malignant',
                  'limitation': 'Small historical diagnostic benchmark; not clinical validation'}


def split_rows(y, seed):
    train, rest = train_test_split(np.arange(len(y)), test_size=.5, stratify=y, random_state=seed)
    sel, rest = train_test_split(rest, train_size=.4, stratify=y[rest], random_state=seed+1)
    val, test = train_test_split(rest, train_size=1/3, stratify=y[rest], random_state=seed+2)
    splits = [train, sel, val, test]
    assert len(set(np.concatenate(splits))) == len(y)
    assert sum(map(len, splits)) == len(y)
    return splits


def train_selectors(model, X, y, seed):
    metas, labels, classes = [], [], []
    for i, rate in enumerate([0., .1, .3]):
        damaged, _, _ = inject(X, 'independent', rate, seed*1000+i)
        prob, meta = meta_features(model, damaged)
        metas.append(meta)
        labels.append(((prob >= .5) != y).astype(int))
        classes.append(y)
    labels, classes = np.concatenate(labels), np.concatenate(classes)
    if len(np.unique(labels)) < 2:
        raise ValueError('Selector development has no error-label variation; do not fabricate scores.')
    training = {key: np.vstack([m[key] for m in metas]) for key in metas[0]}
    learned = {}
    for family in ['linear', 'nonlinear', 'classweighted']:
        for variant in ['confidence', 'combined'] + (['missingness'] if family == 'linear' else []):
            key = 'combined' if variant == 'combined' else 'learned_'+variant
            if family == 'nonlinear':
                estimator = HistGradientBoostingClassifier(max_iter=75, max_leaf_nodes=7,
                    min_samples_leaf=20, l2_regularization=2, early_stopping=False, random_state=seed)
                estimator.fit(training[key], labels)
            else:
                estimator = Pipeline([('scale', StandardScaler()),
                                      ('model', LogisticRegression(C=1, max_iter=2000))])
                fit_params = {}
                if family == 'classweighted':
                    counts = np.bincount(classes, minlength=2)
                    weights = len(classes)/(2*counts[classes])
                    fit_params['model__sample_weight'] = weights
                estimator.fit(training[key], labels, **fit_params)
            learned[f'{family}_{variant}'] = (estimator, key)
    return learned


def get_scores(model, learned, X):
    prob, meta = meta_features(model, X)
    scores = {'raw_confidence': np.minimum(prob, 1-prob),
              'missing_fraction': X.isna().mean(axis=1).to_numpy()}
    for name, (estimator, key) in learned.items():
        scores[name] = estimator.predict_proba(meta[key])[:, 1]
    return prob, scores


def retain(score, pred, seed, mode, fraction=.8):
    keep = np.zeros(len(score), dtype=bool)
    order = order_with_ties(score, seed)
    if mode == 'global':
        keep[order[:int(np.ceil(fraction*len(score)))]] = True
    elif mode == 'predicted_class':
        for cls in [0, 1]:
            subset = order[pred[order] == cls]
            keep[subset[:int(np.ceil(fraction*len(subset)))]] = True
    else:
        raise ValueError(mode)
    return keep


def metrics(y, pred, keep):
    err = pred != y
    def mean_or_nan(arr):
        return float(np.mean(arr)) if len(arr) else np.nan
    per_class = [mean_or_nan(err[keep & (y == c)]) for c in [0, 1]]
    return {'risk': mean_or_nan(err[keep]),
            'balanced_risk': float(np.mean(per_class)),
            'retained_fnr': per_class[1], 'retained_fpr': per_class[0],
            'positive_retention': mean_or_nan(keep[y == 1]),
            'negative_retention': mean_or_nan(keep[y == 0]),
            'unreviewed_fn_fraction_of_all_positives': float((keep & err & (y == 1)).sum()/(y == 1).sum()),
            'retained_positive_prevalence': mean_or_nan(y[keep]),
            'coverage': float(keep.mean()), 'retained_n': int(keep.sum())}


def run():
    invariant_checks()
    # Selection must not use true labels; known-size capacity checks cover both modes.
    for mode in ['global', 'predicted_class']:
        k = retain(np.arange(10.), np.array([0]*5+[1]*5), 1, mode)
        assert k.sum() == 8
    m = metrics(np.array([0, 0, 1, 1]), np.array([0, 1, 0, 1]), np.ones(4, bool))
    assert m['risk'] == .5 and m['balanced_risk'] == .5 and m['retained_fnr'] == .5
    OUT.mkdir(parents=True, exist_ok=True)
    rows, fixed, full = [], [], []
    manifest = {'started_utc': datetime.now(timezone.utc).isoformat(), 'seeds': SEEDS,
                'models': MODELS, 'python': sys.version, 'platform': platform.platform(),
                'versions': {'numpy': np.__version__, 'pandas': pd.__version__, 'sklearn': sklearn.__version__},
                'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'helper_sha256': hashlib.sha256((ROOT/'src/pilot.py').read_bytes()).hexdigest(),
                'plan_sha256': hashlib.sha256((ROOT/'replication_plan.md').read_bytes()).hexdigest(),
                'datasets': {}, 'status': 'exploratory running'}
    (OUT/'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    for dataset in ['bank', 'wdbc']:
        X, y, source = load_named(dataset)
        manifest['datasets'][dataset] = {**source, 'rows': len(X), 'features': list(X.columns),
            'block_a': list(X.columns[:len(X.columns)//2]), 'block_b': list(X.columns[len(X.columns)//2:])}
        for seed in SEEDS:
            train, sel, val, test = split_rows(y, seed)
            subdir = OUT/f'{dataset}_seed{seed}'
            subdir.mkdir(exist_ok=True)
            (subdir/'splits.json').write_text(json.dumps({k: v.tolist() for k, v in
                zip(['train','selector','validation','evaluation'], [train, sel, val, test])}), encoding='utf-8')
            for name in MODELS:
                print(f'{dataset} seed {seed}: {name}', flush=True)
                if name == 'native_boost_calibrated':
                    model = CalibratedClassifierCV(ResearchClassifier('native_boost_augmented', seed),
                        method='sigmoid', cv=StratifiedKFold(3, shuffle=True, random_state=seed), n_jobs=1)
                else:
                    model = ResearchClassifier(name, seed)
                model.fit(X.iloc[train], y[train])
                learned = train_selectors(model, X.iloc[sel], y[sel], seed)
                validation_scores = {}
                for i, rate in enumerate([0., .1, .3]):
                    damaged, _, _ = inject(X.iloc[val], 'independent', rate, seed*2000+i)
                    _, score_map = get_scores(model, learned, damaged)
                    for key, value in score_map.items():
                        validation_scores.setdefault(key, []).append(value)
                thresholds = {key: float(np.quantile(np.concatenate(value), .8))
                              for key, value in validation_scores.items()}
                conditions = [('independent', 0.), ('independent', .1), ('independent', .3)]
                conditions += [(kind, rate) for kind in ['block_a','block_b'] for rate in [.1,.3]]
                pred_frames = []
                for i, (kind, rate) in enumerate(conditions):
                    damaged, added, newly_lost = inject(X.iloc[test], kind, rate, seed*3000+i)
                    prob, score_map = get_scores(model, learned, damaged)
                    pred = (prob >= .5).astype(int)
                    base = {'dataset': dataset, 'seed': seed, 'model': name,
                            'condition': f'{kind}_{rate:.1f}', 'nominal_cell_loss': rate,
                            'added_mask_fraction': added, 'newly_lost_fraction': newly_lost}
                    full.append({**base, **metrics(y[test], pred, np.ones(len(test), bool)),
                        'roc_auc': roc_auc_score(y[test], prob),
                        'average_precision': average_precision_score(y[test], prob),
                        'brier': brier_score_loss(y[test], prob),
                        'log_loss': log_loss(y[test], np.column_stack([1-prob, prob]))})
                    saved = pd.DataFrame({'source_index': X.index[test], 'label': y[test],
                        'prediction': pred, 'probability': prob, 'condition': base['condition']})
                    for key, score in score_map.items():
                        saved['score_'+key] = score
                        for mode in ['global', 'predicted_class']:
                            keep = retain(score, pred, seed*4000+i, mode)
                            rows.append({**base, 'selector': key, 'selection_mode': mode,
                                **metrics(y[test], pred, keep)})
                        fixed.append({**base, 'selector': key, 'threshold': thresholds[key],
                            **metrics(y[test], pred, score <= thresholds[key])})
                    pred_frames.append(saved)
                pd.concat(pred_frames).to_csv(subdir/f'{name}_predictions.csv', index=False)
                # Save incremental progress so an interrupted run remains auditable.
                pd.DataFrame(rows).to_csv(OUT/'selective_metrics.csv', index=False)
                pd.DataFrame(fixed).to_csv(OUT/'fixed_threshold_metrics.csv', index=False)
                pd.DataFrame(full).to_csv(OUT/'classifier_metrics.csv', index=False)
    manifest['status'] = 'completed exploratory; no confirmatory claims'
    manifest['completed_utc'] = datetime.now(timezone.utc).isoformat()
    (OUT/'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print('Completed all 24 dataset/seed/classifier fits.', flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=2):
        run()
