"""평가 지표.

- 1차년도 비교 평가 지표 설계: 재현율(Recall), 정밀도(Precision), F1 (조화 평균)
- 2차년도 API 엔드포인트 알고리즘: 데이터 비율 p_i = n_i / N, 정보 엔트로피 H = -sum p log p
- 3차년도 연합 학습 특화 지표 일부: 수렴 시간, 통신 효율성, 공정성(클라이언트 성능 분산)
"""
from typing import Dict, List, Optional, Sequence

import numpy as np


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(cm, (np.asarray(y_true, dtype=np.int64), np.asarray(y_pred, dtype=np.int64)), 1)
    return cm


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int,
                           average: str = "macro") -> Dict[str, float]:
    """정확도, 정밀도, 재현율, F1 (macro 또는 weighted 평균)."""
    cm = confusion_matrix(y_true, y_pred, n_classes)
    tp = np.diag(cm).astype(np.float64)
    pred_pos = cm.sum(0).astype(np.float64)
    true_pos = cm.sum(1).astype(np.float64)
    precision = np.divide(tp, pred_pos, out=np.zeros_like(tp), where=pred_pos > 0)
    recall = np.divide(tp, true_pos, out=np.zeros_like(tp), where=true_pos > 0)
    denom = precision + recall
    f1 = np.divide(2 * precision * recall, denom, out=np.zeros_like(tp), where=denom > 0)
    present = true_pos > 0
    if average == "weighted":
        w = true_pos / max(true_pos.sum(), 1)
        agg = lambda v: float((v * w).sum())
    else:
        agg = lambda v: float(v[present].mean()) if present.any() else 0.0
    return {
        "accuracy": float(tp.sum() / max(cm.sum(), 1)),
        "precision": agg(precision),
        "recall": agg(recall),
        "f1": agg(f1),
    }


def data_ratio(counts: Sequence[float]) -> np.ndarray:
    """p_i = n_i / N (0과 1 사이)."""
    c = np.asarray(counts, dtype=np.float64)
    return c / max(c.sum(), 1e-12)


def entropy(dist: Sequence[float], base: float = 2.0) -> float:
    """정보 엔트로피 H = -sum_{p>0} p log p (2차년도 알고리즘과 동일하게 p>0 항만 합산)."""
    p = np.asarray(dist, dtype=np.float64)
    s = p.sum()
    if s <= 0:
        return 0.0
    p = p / s
    p = p[p > 0]
    return float(-(p * np.log(p)).sum() / np.log(base))


def normalized_entropy(dist: Sequence[float]) -> float:
    k = len(dist)
    return entropy(dist) / np.log2(k) if k > 1 else 0.0


def kl_divergence(p: Sequence[float], q: Sequence[float], eps: float = 1e-12) -> float:
    p = np.asarray(p, dtype=np.float64) + eps
    q = np.asarray(q, dtype=np.float64) + eps
    p, q = p / p.sum(), q / q.sum()
    return float((p * np.log(p / q)).sum())


def fairness(values: Sequence[float]) -> Dict[str, float]:
    """클라이언트 성능 분포 요약 (q-FedAvg 평가 시 공정성 비교용)."""
    v = np.asarray(values, dtype=np.float64)
    if v.size == 0:
        return {"mean": 0.0, "std": 0.0, "worst10": 0.0, "best10": 0.0}
    k = max(1, int(np.ceil(0.1 * v.size)))
    s = np.sort(v)
    return {"mean": float(v.mean()), "std": float(v.std()),
            "worst10": float(s[:k].mean()), "best10": float(s[-k:].mean())}


def convergence_round(series: Sequence[float], target: Optional[float] = None,
                      frac: float = 0.95) -> Optional[int]:
    """연합 모델 수렴 시점: 최종 성능의 frac 배(또는 target)에 처음 도달한 라운드."""
    s = np.asarray(series, dtype=np.float64)
    if s.size == 0:
        return None
    thr = target if target is not None else frac * float(np.max(s))
    hit = np.where(s >= thr)[0]
    return int(hit[0]) + 1 if hit.size else None


def bootstrap_diff_ci(y_true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray, n_classes: int,
                      metric: str = "f1", n_boot: int = 300, seed: int = 0) -> Dict[str, float]:
    """동일 테스트셋에서 두 모델 지표 차이(a - b)의 부트스트랩 95% 신뢰구간."""
    rng = np.random.RandomState(seed)
    n = len(y_true)
    diffs = []
    for _ in range(n_boot):
        i = rng.randint(0, n, n)
        a = classification_metrics(y_true[i], pred_a[i], n_classes)[metric]
        b = classification_metrics(y_true[i], pred_b[i], n_classes)[metric]
        diffs.append(a - b)
    d = np.asarray(diffs)
    return {"mean": float(d.mean()), "lo": float(np.percentile(d, 2.5)),
            "hi": float(np.percentile(d, 97.5))}


def summarize(list_of_dicts: List[Dict[str, float]]) -> Dict[str, float]:
    keys = list_of_dicts[0].keys() if list_of_dicts else []
    return {k: float(np.mean([d[k] for d in list_of_dicts])) for k in keys}
