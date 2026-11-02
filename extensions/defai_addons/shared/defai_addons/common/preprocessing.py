"""데이터 전처리 파이프라인: ➊데이터 정제 ➋정규화 ➌특징 추출 (1·3차년도).

지역 분산 컴퓨팅과 연합 컴퓨팅을 같은 기준으로 비교하기 위해, 각 단계를 세 가지
통계 공유 방식으로 맞출 수 있다.

  mode='local'      각 사이트가 자기 데이터로만 통계를 적합 (공유 없음)
  mode='federated'  사이트별 충분통계(개수·합·제곱합·외적합)만 서버로 보내 집계 → 원 데이터 미전송
  mode='global'     데이터를 한 곳에 모아 적합 (지역 분산 환경의 데이터 이동 시나리오)

federated 와 global 은 수학적으로 같은 결과를 내므로, 두 환경의 차이는 '데이터 이동 여부'와
통신량으로만 나타나게 된다.
"""
from typing import Dict, List, Optional

import numpy as np


class Cleaner:
    """결측치 대치와 이상치 클리핑.

    - 결측 비율이 row_nan_thresh 이상인 행 제거
    - 결측치는 열 중앙값(근사: 사이트 중앙값의 표본 수 가중 평균)으로 대치
    - 각 열을 [Q1 - k*IQR, Q3 + k*IQR] 로 클리핑
    """

    def __init__(self, row_nan_thresh: float = 0.5, iqr_k: float = 3.0):
        self.row_nan_thresh = row_nan_thresh
        self.iqr_k = iqr_k
        self.median_ = None
        self.lo_ = None
        self.hi_ = None

    @staticmethod
    def _site_stats(X: np.ndarray) -> Dict[str, np.ndarray]:
        q1, med, q3 = np.nanpercentile(X, [25, 50, 75], axis=0)
        return {"n": np.sum(~np.isnan(X), axis=0).astype(np.float64), "q1": q1, "med": med, "q3": q3}

    def fit(self, sites: List[np.ndarray]) -> "Cleaner":
        stats = [self._site_stats(X) for X in sites]
        n = np.stack([s["n"] for s in stats])
        w = n / np.maximum(n.sum(0, keepdims=True), 1)
        agg = {k: np.nansum(np.stack([s[k] for s in stats]) * w, axis=0) for k in ("q1", "med", "q3")}
        iqr = agg["q3"] - agg["q1"]
        self.median_ = agg["med"]
        self.lo_ = agg["q1"] - self.iqr_k * iqr
        self.hi_ = agg["q3"] + self.iqr_k * iqr
        return self

    def row_mask(self, X: np.ndarray) -> np.ndarray:
        return np.isnan(X).mean(1) < self.row_nan_thresh

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = X.copy()
        nan = np.isnan(X)
        X[nan] = np.broadcast_to(self.median_, X.shape)[nan]
        # 상수 열(iqr=0)은 클리핑하지 않는다
        flat = self.hi_ <= self.lo_
        lo = np.where(flat, -np.inf, self.lo_)
        hi = np.where(flat, np.inf, self.hi_)
        return np.clip(X, lo, hi)


class Normalizer:
    """z-score 정규화. 충분통계 (n, sum, sumsq) 로 적합하므로 연합 집계가 가능하다."""

    def __init__(self):
        self.mean_ = None
        self.std_ = None

    @staticmethod
    def sufficient_stats(X: np.ndarray) -> Dict[str, np.ndarray]:
        X = X.astype(np.float64)
        return {"n": np.array(len(X), dtype=np.float64), "s": X.sum(0), "ss": (X * X).sum(0)}

    def fit_stats(self, stats: List[Dict[str, np.ndarray]]) -> "Normalizer":
        n = sum(s["n"] for s in stats)
        s = sum(st["s"] for st in stats)
        ss = sum(st["ss"] for st in stats)
        self.mean_ = s / n
        var = np.maximum(ss / n - self.mean_ ** 2, 0.0)
        self.std_ = np.sqrt(var) + 1e-6
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return ((X - self.mean_) / self.std_).astype(np.float32)


class PCAExtractor:
    """PCA 특징 추출. 공분산을 (n, sum x, sum x x^T) 충분통계로 계산한다."""

    def __init__(self, n_components: Optional[int] = None, var_ratio: float = 0.95):
        self.n_components = n_components
        self.var_ratio = var_ratio
        self.mean_ = None
        self.components_ = None
        self.explained_ = None

    @staticmethod
    def sufficient_stats(X: np.ndarray) -> Dict[str, np.ndarray]:
        X = X.astype(np.float64)
        return {"n": np.array(len(X), dtype=np.float64), "s": X.sum(0), "xx": X.T @ X}

    def fit_stats(self, stats: List[Dict[str, np.ndarray]]) -> "PCAExtractor":
        n = sum(s["n"] for s in stats)
        mean = sum(s["s"] for s in stats) / n
        cov = sum(s["xx"] for s in stats) / n - np.outer(mean, mean)
        evals, evecs = np.linalg.eigh(cov)
        order = np.argsort(evals)[::-1]
        evals, evecs = np.maximum(evals[order], 0), evecs[:, order]
        ratio = np.cumsum(evals) / max(evals.sum(), 1e-12)
        k = self.n_components or int(np.searchsorted(ratio, self.var_ratio) + 1)
        self.mean_, self.components_ = mean, evecs[:, :k]
        self.explained_ = ratio[:k]
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return ((X - self.mean_) @ self.components_).astype(np.float32)


def stats_bytes(stats: List[Dict[str, np.ndarray]]) -> int:
    """연합 모드에서 서버로 전송되는 충분통계 바이트 (float32 기준)."""
    return int(sum(sum(np.asarray(v).size for v in s.values()) for s in stats) * 4)


class PreprocessingPipeline:
    """정제 → 정규화 → 특징추출 파이프라인.

    fit(sites) 가 반환하는 dict 에는 각 단계에서 사이트 밖으로 나간 바이트 수가 기록된다.
    """

    def __init__(self, mode: str = "federated", use_pca: bool = True, pca_var: float = 0.95,
                 iqr_k: float = 3.0):
        assert mode in ("local", "federated", "global")
        self.mode = mode
        self.use_pca = use_pca
        self.pca_var = pca_var
        self.iqr_k = iqr_k
        self.site_pipes: List[Dict] = []
        self.shared: Optional[Dict] = None

    def _fit_group(self, sites: List[np.ndarray]) -> Dict:
        cleaner = Cleaner(iqr_k=self.iqr_k).fit(sites)
        cleaned = [cleaner.transform(X[cleaner.row_mask(X)]) for X in sites]
        norm = Normalizer().fit_stats([Normalizer.sufficient_stats(X) for X in cleaned])
        normed = [norm.transform(X) for X in cleaned]
        pca = None
        if self.use_pca:
            pca = PCAExtractor(var_ratio=self.pca_var).fit_stats(
                [PCAExtractor.sufficient_stats(X) for X in normed])
        n_feat = sites[0].shape[1]
        sent = 0
        if len(sites) > 1:
            sent = len(sites) * 4 * (n_feat * 4 + 1 + n_feat + n_feat + (n_feat * n_feat if self.use_pca else 0))
        return {"cleaner": cleaner, "norm": norm, "pca": pca, "bytes": sent}

    def fit(self, sites: List[np.ndarray]) -> Dict[str, float]:
        if self.mode == "local":
            self.site_pipes = [self._fit_group([X]) for X in sites]
            return {"stat_bytes": 0, "raw_bytes": 0}
        if self.mode == "global":
            pooled = np.concatenate(sites, 0)
            self.shared = self._fit_group([pooled])
            # 원 데이터를 한 곳으로 이동 (가장 큰 사이트는 이미 그 위치에 있다고 가정)
            sizes = sorted([X.nbytes for X in sites])
            return {"stat_bytes": 0, "raw_bytes": int(sum(sizes[:-1]))}
        self.shared = self._fit_group(sites)
        return {"stat_bytes": int(self.shared["bytes"]), "raw_bytes": 0}

    def _pipe(self, site: Optional[int]) -> Dict:
        if self.mode == "local":
            if site is None:
                raise ValueError("local mode requires a site index")
            return self.site_pipes[site]
        return self.shared

    def transform(self, X: np.ndarray, site: Optional[int] = None, drop_rows: bool = False,
                  y: Optional[np.ndarray] = None):
        p = self._pipe(site)
        if drop_rows:
            m = p["cleaner"].row_mask(X)
            X = X[m]
            y = y[m] if y is not None else None
        Z = p["norm"].transform(p["cleaner"].transform(X))
        if p["pca"] is not None:
            Z = p["pca"].transform(Z)
        return (Z, y) if drop_rows else Z

    def out_dim(self, site: Optional[int] = None) -> int:
        p = self._pipe(site if self.mode == "local" else None)
        return p["pca"].components_.shape[1] if p["pca"] is not None else p["norm"].mean_.shape[0]
