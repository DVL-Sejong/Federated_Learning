"""Flower(flwr) 기반 연합 학습 공통 계층.

모든 연합 학습은 Flower 의 표준 구성요소로 수행한다.
  - 클라이언트 : flwr.client.NumPyClient (fit / evaluate / get_parameters)
  - 집계       : flwr.server.strategy.* (FedAvg, FedProx, QFedAvg 및 그 하위 클래스)
  - 라운드 진행 : Strategy.configure_fit → ClientProxy.fit → Strategy.aggregate_fit
                 (flwr.server.Server.fit_round 와 같은 절차)

계층적 연합 학습(서버-엣지-클라이언트)은 Flower 가 기본 제공하지 않으므로, 엣지 서버마다 Flower Strategy 와
ClientManager 를 두고, 중앙 서버 Strategy 가 엣지 결과(FitRes)를 다시 집계하는 방식으로 구성한다.
같은 프로세스 안에서는 LocalClientProxy 로 클라이언트를 직접 호출하고, 실제 분산 배포는
flwr.server.start_server / flwr.client.start_client (scripts/flower_server.py, flower_client.py) 를 쓴다.
"""
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from flwr.client import Client, NumPyClient
from flwr.common import (Code, EvaluateIns, EvaluateRes, FitIns, FitRes, GetParametersIns, GetParametersRes,
                         GetPropertiesIns, GetPropertiesRes, Parameters, Status, ndarrays_to_parameters,
                         parameters_to_ndarrays)
from flwr.server.client_manager import SimpleClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg, FedProx, QFedAvg, Strategy
from flwr.server.strategy.aggregate import aggregate_qffl

NDArrays = List[np.ndarray]


# ----------------------------------------------------------------------------- 모델 <-> NDArrays
def state_keys(model: nn.Module, include_buffers: bool = True) -> List[str]:
    """Flower 로 주고받을 텐서 키 (기본: 파라미터 + BatchNorm running 통계 등 부동소수 버퍼)."""
    if not include_buffers:
        return [k for k, _ in model.named_parameters()]
    return [k for k, v in model.state_dict().items() if v.dtype.is_floating_point]


def get_ndarrays(model: nn.Module, keys: Sequence[str]) -> NDArrays:
    sd = model.state_dict()
    return [sd[k].detach().cpu().numpy().copy() for k in keys]


def set_ndarrays(model: nn.Module, keys: Sequence[str], arrays: NDArrays) -> None:
    sd = model.state_dict()
    with torch.no_grad():
        for k, a in zip(keys, arrays):
            sd[k].copy_(torch.as_tensor(np.asarray(a), dtype=sd[k].dtype, device=sd[k].device))


def flatten(arrays: NDArrays) -> np.ndarray:
    return np.concatenate([np.asarray(a, dtype=np.float64).ravel() for a in arrays])


def unflatten(vec: np.ndarray, like: NDArrays) -> NDArrays:
    out, i = [], 0
    for a in like:
        n = a.size
        out.append(vec[i:i + n].reshape(a.shape).astype(a.dtype))
        i += n
    return out


def nbytes(arrays: NDArrays) -> int:
    return int(sum(np.asarray(a).nbytes for a in arrays))


# ----------------------------------------------------------------------------- in-process ClientProxy
class LocalClientProxy(ClientProxy):
    """같은 프로세스의 Flower Client 를 ClientProxy 로 감싼다 (Flower 시뮬레이션의 RayActorClientProxy 대응)."""

    def __init__(self, cid: str, client: Client, props: Optional[Dict] = None):
        super().__init__(cid)
        self.client = client
        self.props = props or {}

    def get_properties(self, ins: GetPropertiesIns, timeout=None, group_id=None) -> GetPropertiesRes:
        return GetPropertiesRes(Status(Code.OK, ""), self.props)

    def get_parameters(self, ins: GetParametersIns, timeout=None, group_id=None) -> GetParametersRes:
        return self.client.get_parameters(ins)

    def fit(self, ins: FitIns, timeout=None, group_id=None) -> FitRes:
        return self.client.fit(ins)

    def evaluate(self, ins: EvaluateIns, timeout=None, group_id=None) -> EvaluateRes:
        return self.client.evaluate(ins)

    def reconnect(self, ins, timeout=None, group_id=None):
        return None


def as_proxy(cid, numpy_client: NumPyClient, props: Optional[Dict] = None) -> LocalClientProxy:
    return LocalClientProxy(str(cid), numpy_client.to_client(), props)


def client_manager(proxies: Sequence[ClientProxy]) -> SimpleClientManager:
    cm = SimpleClientManager()
    for p in proxies:
        cm.register(p)
    return cm


def run_fit(strategy: Strategy, server_round: int, parameters: Parameters, proxies: Sequence[ClientProxy]
            ) -> Tuple[List[Tuple[ClientProxy, FitRes]], List[BaseException]]:
    """Strategy.configure_fit 으로 선택·설정된 클라이언트들의 fit 을 실행한다 (Server.fit_round 와 동일 절차)."""
    if not proxies:
        return [], []
    instructions = strategy.configure_fit(server_round, parameters, client_manager(proxies))
    results, failures = [], []
    for proxy, ins in instructions:
        try:
            res = proxy.fit(ins, None, server_round)
            (results if res.status.code == Code.OK else failures).append((proxy, res))
        except Exception as e:  # pragma: no cover
            failures.append(e)
    return results, failures


def fit_res(arrays: NDArrays, num_examples: int, metrics: Optional[Dict] = None) -> FitRes:
    return FitRes(Status(Code.OK, ""), ndarrays_to_parameters(arrays), int(num_examples), metrics or {})


# ----------------------------------------------------------------------------- strategies
def _all(n: int = 0) -> Dict:
    """configure_fit 이 등록된 모든 클라이언트를 고르도록 하는 설정 (참여 선택은 시뮬레이터가 담당)."""
    return dict(fraction_fit=1.0, fraction_evaluate=0.0, min_fit_clients=1, min_evaluate_clients=0,
                min_available_clients=1, accept_failures=True)


class ClientLossQFedAvg(QFedAvg):
    """q-FedAvg (Li et al., ICLR 2020).

    Flower 의 QFedAvg.aggregate_fit 는 서버 evaluate_fn 의 단일 손실을 모든 클라이언트에 쓰므로,
    논문 식대로 각 클라이언트가 fit 지표로 보고한 전역 모델 손실 F_k (metrics['loss_before']) 를 사용하도록 재정의한다.
    집계 자체는 flwr 의 aggregate_qffl 을 그대로 사용한다.
    """

    def aggregate_fit(self, server_round, results, failures):
        if not results:
            return None, {}
        if self.pre_weights is None:
            raise AttributeError("pre_weights are None (configure_fit 을 먼저 호출해야 함)")
        w0 = self.pre_weights
        deltas, hs = [], []
        for _, r in results:
            Fk = float(r.metrics.get("loss_before", 1.0)) + 1e-10
            new = parameters_to_ndarrays(r.parameters)
            grads = [np.multiply(u - v, 1.0 / self.learning_rate) for u, v in zip(w0, new)]
            sq = float(sum(np.square(g).sum() for g in grads))
            deltas.append([np.float_power(Fk, self.q_param) * g for g in grads])
            hs.append(self.q_param * np.float_power(Fk, self.q_param - 1) * sq
                      + (1.0 / self.learning_rate) * np.float_power(Fk, self.q_param))
        return ndarrays_to_parameters(aggregate_qffl(w0, deltas, hs)), {}


class WeightedFedAvg(FedAvg):
    """FedAvg 의 표본수 가중 대신 클라이언트별 가중치 함수를 쓰는 Strategy.
    weight_fn(results) → 정규화 전 가중치 목록. (FedATC 의 데이터 크기·정보 엔트로피 가중 평균 등)"""

    def __init__(self, weight_fn: Callable[[List[Tuple[ClientProxy, FitRes]]], Sequence[float]], **kw):
        super().__init__(**kw)
        self.weight_fn = weight_fn
        self.last_weights: Dict[str, float] = {}

    def aggregate_fit(self, server_round, results, failures):
        if not results:
            return None, {}
        w = np.asarray(self.weight_fn(results), dtype=np.float64)
        w = w / w.sum()
        arrays = [parameters_to_ndarrays(r.parameters) for _, r in results]
        agg = [sum(wi * a[j] for wi, a in zip(w, arrays)).astype(arrays[0][j].dtype) for j in range(len(arrays[0]))]
        self.last_weights = {p.cid: float(wi) for (p, _), wi in zip(results, w)}
        return ndarrays_to_parameters(agg), {}


def make_strategy(algorithm: str, initial: NDArrays, mu: float = 0.1, q: float = 0.2, lr: float = 0.1,
                  on_fit_config: Optional[Callable[[int], Dict]] = None) -> Strategy:
    """FedAvg / FedProx / q-FedAvg Flower Strategy 생성.
    FedProx 는 configure_fit 에서 config['proximal_mu'] 를 클라이언트로 보내며, 클라이언트 fit 이 근접항을 적용한다."""
    kw = dict(_all(0), initial_parameters=ndarrays_to_parameters(initial), on_fit_config_fn=on_fit_config)
    if algorithm == "fedavg":
        return FedAvg(**kw)
    if algorithm == "fedprox":
        return FedProx(proximal_mu=mu, **kw)
    if algorithm == "qfedavg":
        return ClientLossQFedAvg(q_param=q, qffl_learning_rate=lr, **kw)
    raise ValueError(algorithm)


# Flower 통신 메시지는 bytes 를 지표로 담을 수 있어, 배열을 실어 보낼 때 사용한다
def array_to_bytes(a: np.ndarray) -> bytes:
    import io
    buf = io.BytesIO()
    np.save(buf, np.asarray(a), allow_pickle=False)
    return buf.getvalue()


def bytes_to_array(b: bytes) -> np.ndarray:
    import io
    return np.load(io.BytesIO(b), allow_pickle=False)
