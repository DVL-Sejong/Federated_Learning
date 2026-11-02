"""데이터 적재 및 클라이언트/지역 분할 유틸리티.

- 1차년도 '지역 분산 분석 성능 평가 시뮬레이션 환경 설계'의 데이터 준비 단계
  (도메인 데이터 수집 → 무작위 분할로 학습/테스트 구성)
- 3차년도 '데이터 불균형 시나리오' (클라이언트별 할당량/분포 조절)

연합 컴퓨팅 기본 시험 데이터셋은 MNIST 이다. 'label_groups' 분할로 클라이언트마다
숫자 {0,1} / {2,3} / … 처럼 일부 숫자만 보유하게 하고, 중앙(전역) 모델은 0~9 전체 숫자로 평가한다.
"""
import gzip
import os
import urllib.request
from typing import Dict, List, Optional, Tuple

import numpy as np

MNIST_URL = "https://ossci-datasets.s3.amazonaws.com/mnist/"  # torchvision 이 사용하는 공개 미러
MNIST_FILES = {"train_x": "train-images-idx3-ubyte.gz", "train_y": "train-labels-idx1-ubyte.gz",
               "test_x": "t10k-images-idx3-ubyte.gz", "test_y": "t10k-labels-idx1-ubyte.gz"}
DATA_DIR = os.environ.get("DEFAI_DATA", os.path.join(os.path.expanduser("~"), ".cache", "defai_addons"))


def _read_idx(path: str) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        raw = f.read()
    ndim = raw[3]
    shape = tuple(int.from_bytes(raw[4 + 4 * i:8 + 4 * i], "big") for i in range(ndim))
    return np.frombuffer(raw, dtype=np.uint8, offset=4 + 4 * ndim).reshape(shape)


def load_mnist(root: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """MNIST (공식 학습 60,000 / 테스트 10,000). 최초 1회 내려받아 root 에 캐시한다.
    반환 X 는 [0,1] 로 정규화한 784 차원 float32."""
    root = os.path.abspath(os.path.join(root or DATA_DIR, "mnist"))
    os.makedirs(root, exist_ok=True)
    arr = {}
    for k, fn in MNIST_FILES.items():
        path = os.path.join(root, fn)
        if not os.path.exists(path):
            tmp = path + ".part"
            urllib.request.urlretrieve(MNIST_URL + fn, tmp)
            os.replace(tmp, path)
        arr[k] = _read_idx(path)
    f = lambda x: (x.reshape(len(x), -1).astype(np.float32) / 255.0)
    return f(arr["train_x"]), arr["train_y"].astype(np.int64), f(arr["test_x"]), arr["test_y"].astype(np.int64)


def load_split(name: str = "mnist", seed: int = 0, max_train: Optional[int] = None
               ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """학습/테스트 분할을 반환한다. MNIST 는 공식 분할, 나머지는 무작위 80/20."""
    if name == "mnist":
        Xtr, ytr, Xte, yte = load_mnist()
    else:
        X, y = load_tabular(name, seed=seed)
        Xtr, ytr, Xte, yte = train_test_split(X, y, 0.2, seed=seed)
    if max_train and max_train < len(ytr):
        idx = np.random.RandomState(seed).permutation(len(ytr))[:max_train]
        Xtr, ytr = Xtr[idx], ytr[idx]
    return Xtr, ytr, Xte, yte


def load_tabular(name: str = "digits", seed: int = 0, n_samples: int = 6000,
                 n_features: int = 40, n_classes: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    """오프라인으로 사용 가능한 표 형식 분류 데이터셋을 반환한다.

    name: 'mnist' | 'digits' (sklearn 내장 손글씨 8x8, 실데이터) | 'synthetic' (make_classification)
    """
    if name == "mnist":
        Xtr, ytr, Xte, yte = load_mnist()
        return np.concatenate([Xtr, Xte]), np.concatenate([ytr, yte])
    if name == "digits":
        from sklearn.datasets import load_digits
        d = load_digits()
        return d.data.astype(np.float32), d.target.astype(np.int64)
    if name == "synthetic":
        from sklearn.datasets import make_classification
        X, y = make_classification(
            n_samples=n_samples, n_features=n_features, n_informative=n_features // 2,
            n_redundant=n_features // 4, n_classes=n_classes, n_clusters_per_class=2,
            class_sep=1.2, random_state=seed)
        return X.astype(np.float32), y.astype(np.int64)
    raise ValueError("unknown dataset: %s" % name)


def inject_corruption(X: np.ndarray, missing_rate: float = 0.0, outlier_rate: float = 0.0,
                      seed: int = 0) -> np.ndarray:
    """결측치와 이상치를 주입해 데이터 정제 단계를 시험할 수 있게 한다."""
    rng = np.random.RandomState(seed)
    X = X.astype(np.float32).copy()
    if missing_rate > 0:
        X[rng.rand(*X.shape) < missing_rate] = np.nan
    if outlier_rate > 0:
        mask = rng.rand(*X.shape) < outlier_rate
        scale = np.nanstd(X, axis=0, keepdims=True) + 1e-6
        X[mask] = (X + 15.0 * scale * rng.choice([-1, 1], size=X.shape))[mask]
    return X


def train_test_split(X: np.ndarray, y: np.ndarray, test_ratio: float = 0.2,
                     seed: int = 0) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(y))
    n_test = int(len(y) * test_ratio)
    te, tr = idx[:n_test], idx[n_test:]
    return X[tr], y[tr], X[te], y[te]


def label_groups(n_classes: int, classes_per_client: int) -> List[List[int]]:
    """연속한 숫자 묶음: classes_per_client=2 → [[0,1],[2,3],...,[8,9]]."""
    return [list(range(s, min(s + classes_per_client, n_classes))) for s in range(0, n_classes, classes_per_client)]


def partition(y: np.ndarray, n_clients: int, scheme: str = "dirichlet", alpha: float = 0.5,
              quantity_sigma: float = 0.0, seed: int = 0, min_size: int = 10,
              classes_per_client: int = 2) -> List[np.ndarray]:
    """레이블 배열을 n_clients 개로 분할해 각 클라이언트의 인덱스를 반환한다.

    scheme
      'iid'        : 균등 무작위 분할
      'dirichlet'  : Dir(alpha) 레이블 편향 분할 (alpha가 작을수록 불균형)
      'shard'      : 각 클라이언트가 소수 클래스만 보유 (무작위 샤드)
      'label_groups': 클라이언트 k 는 숫자 묶음 k mod G 만 보유 (예: {0,1} / {2,3} / …).
                     같은 묶음을 가진 클라이언트끼리 해당 숫자의 표본을 나눠 갖는다.
    quantity_sigma > 0 이면 log-normal 할당량 편차(데이터 양 불균형)를 추가로 적용한다.
    """
    rng = np.random.RandomState(seed)
    n = len(y)
    classes = np.unique(y)
    if scheme == "iid":
        props = np.ones(n_clients)
        if quantity_sigma > 0:
            props = rng.lognormal(0.0, quantity_sigma, n_clients)
        props = props / props.sum()
        idx = rng.permutation(n)
        cuts = (np.cumsum(props) * n).astype(int)[:-1]
        parts = np.split(idx, cuts)
    elif scheme == "dirichlet":
        for _ in range(100):
            parts_l = [[] for _ in range(n_clients)]
            qty = rng.lognormal(0.0, quantity_sigma, n_clients) if quantity_sigma > 0 else np.ones(n_clients)
            for c in classes:
                idx_c = rng.permutation(np.where(y == c)[0])
                p = rng.dirichlet(np.full(n_clients, alpha)) * qty
                p = p / p.sum()
                cuts = (np.cumsum(p) * len(idx_c)).astype(int)[:-1]
                for k, chunk in enumerate(np.split(idx_c, cuts)):
                    parts_l[k].extend(chunk.tolist())
            if min(len(p) for p in parts_l) >= min_size:
                break
        parts = [np.array(p, dtype=np.int64) for p in parts_l]
    elif scheme == "label_groups":
        groups = label_groups(int(y.max()) + 1, classes_per_client)
        owners = [[k for k in range(n_clients) if k % len(groups) == g] for g in range(len(groups))]
        parts_l = [[] for _ in range(n_clients)]
        for g, cls in enumerate(groups):
            if not owners[g]:
                continue
            idx = rng.permutation(np.where(np.isin(y, cls))[0])
            if quantity_sigma > 0:
                p = rng.lognormal(0.0, quantity_sigma, len(owners[g]))
                cuts = (np.cumsum(p / p.sum()) * len(idx)).astype(int)[:-1]
                chunks = np.split(idx, cuts)
            else:
                chunks = np.array_split(idx, len(owners[g]))
            for k, ch in zip(owners[g], chunks):
                parts_l[k].extend(ch.tolist())
        parts = [np.array(p, dtype=np.int64) for p in parts_l]
    elif scheme == "shard":
        shards_per = 2
        order = np.argsort(y, kind="stable")
        shards = np.array_split(order, n_clients * shards_per)
        perm = rng.permutation(len(shards))
        parts = [np.concatenate([shards[j] for j in perm[k * shards_per:(k + 1) * shards_per]])
                 for k in range(n_clients)]
    else:
        raise ValueError("unknown scheme: %s" % scheme)
    return [rng.permutation(p) for p in parts]


def split_clients(X: np.ndarray, y: np.ndarray, parts: List[np.ndarray], val_ratio: float = 0.2,
                  seed: int = 0) -> List[Dict[str, np.ndarray]]:
    """각 클라이언트 데이터를 로컬 학습/검증 세트로 나눈다."""
    rng = np.random.RandomState(seed)
    out = []
    for p in parts:
        p = rng.permutation(p)
        n_val = max(1, int(len(p) * val_ratio))
        out.append({"X": X[p[n_val:]], "y": y[p[n_val:]], "Xv": X[p[:n_val]], "yv": y[p[:n_val]]})
    return out


def class_distribution(y: np.ndarray, n_classes: int) -> np.ndarray:
    return np.bincount(y, minlength=n_classes).astype(np.float64)


def describe_partition(parts: List[np.ndarray], y: np.ndarray, n_classes: Optional[int] = None) -> np.ndarray:
    """클라이언트 x 클래스 개수 행렬."""
    n_classes = n_classes or int(y.max()) + 1
    return np.stack([class_distribution(y[p], n_classes) for p in parts])
