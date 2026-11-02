"""계층적 연합 학습의 게임 이론 기반 클라이언트-엣지 서버 매칭 (3차년도).

흐름 (3차년도 계획서 '매칭 알고리즘')
  (클라이언트) 엣지 서버의 보상 R_j 와 데이터 비용 c_i 를 고려한 이익 함수 설정
  (엣지 서버) 지수 평활(EMA)로 망각 효과를 구현한 모델 품질 추정량 Q_i 도입
  우선순위 목록을 기반으로 클라이언트와 엣지 서버 매칭 (Deferred Acceptance, 안정 매칭)
"""
from typing import Dict, List, Optional

import numpy as np


class QualityEstimator:
    """Q_i ← β Q_i + (1 - β) q_i^t  (β: 망각 계수, 최근 관측일수록 영향이 큼)."""

    def __init__(self, n_clients: int, beta: float = 0.7, init: float = 0.5):
        self.beta = beta
        self.Q = np.full(n_clients, init, dtype=np.float64)
        self.n_obs = np.zeros(n_clients, dtype=np.int64)

    def update(self, i: int, q: float) -> float:
        self.Q[i] = self.beta * self.Q[i] + (1 - self.beta) * float(q)
        self.n_obs[i] += 1
        return float(self.Q[i])


def client_profit(R: np.ndarray, n: np.ndarray, cost: np.ndarray, dist: np.ndarray,
                  capacity: np.ndarray, lam: float = 0.02) -> np.ndarray:
    """이익 함수 π_ij = R_j · n_i / (n_i + E[N_j]) − c_i · n_i − λ · d_ij.

    E[N_j] = (capacity_j - 1) · mean(n) : 엣지 j 에 함께 매칭될 것으로 기대되는 다른 클라이언트 데이터량
    """
    expected_load = (capacity - 1).clip(min=0) * n.mean()
    share = n[:, None] / (n[:, None] + expected_load[None, :])
    return R[None, :] * share - (cost * n)[:, None] - lam * dist


def edge_score(Q: np.ndarray, n: np.ndarray, dist: np.ndarray, gamma: float = 0.01) -> np.ndarray:
    """엣지 서버의 클라이언트 선호 점수 s_ij = Q_i · log(1 + n_i) − γ · d_ij."""
    return Q[:, None] * np.log1p(n)[:, None] - gamma * dist


def deferred_acceptance(client_pref: np.ndarray, edge_pref: np.ndarray, capacity: np.ndarray,
                        acceptable: Optional[np.ndarray] = None) -> np.ndarray:
    """다대일 Gale–Shapley (클라이언트 제안). 반환: assign[i] = 엣지 j (미매칭은 -1).

    client_pref[i, j], edge_pref[i, j]: 값이 클수록 선호
    acceptable[i, j]: 클라이언트가 이익이 음수인 엣지에는 제안하지 않음 (개인 합리성)
    결과는 안정 매칭: 서로 현재 짝보다 상대를 더 선호하는 (i, j) 쌍이 존재하지 않는다.
    """
    nc, ne = client_pref.shape
    if acceptable is None:
        acceptable = np.ones_like(client_pref, dtype=bool)
    order = [[j for j in np.argsort(-client_pref[i]) if acceptable[i, j]] for i in range(nc)]
    nxt = np.zeros(nc, dtype=np.int64)
    held: List[List[int]] = [[] for _ in range(ne)]
    free = list(range(nc))
    while free:
        i = free.pop()
        if nxt[i] >= len(order[i]):
            continue
        j = order[i][nxt[i]]
        nxt[i] += 1
        held[j].append(i)
        if len(held[j]) > capacity[j]:
            held[j].sort(key=lambda c: -edge_pref[c, j])
            rejected = held[j].pop()
            free.append(rejected)
    assign = -np.ones(nc, dtype=np.int64)
    for j, cs in enumerate(held):
        for i in cs:
            assign[i] = j
    return assign


def blocking_pairs(assign: np.ndarray, client_pref: np.ndarray, edge_pref: np.ndarray,
                   capacity: np.ndarray, acceptable: np.ndarray) -> int:
    """안정성 검증용: blocking pair 개수 (0 이면 안정 매칭)."""
    nc, ne = client_pref.shape
    cnt = 0
    for i in range(nc):
        for j in range(ne):
            if not acceptable[i, j] or assign[i] == j:
                continue
            if assign[i] >= 0 and client_pref[i, j] <= client_pref[i, assign[i]]:
                continue
            members = np.where(assign == j)[0]
            if len(members) < capacity[j] or any(edge_pref[i, j] > edge_pref[m, j] for m in members):
                cnt += 1
    return cnt


class GameMatcher:
    """보상·비용·품질을 반영한 매칭 엔진. 매 호출마다 우선순위 목록과 매칭 행렬을 반환한다."""

    def __init__(self, n_clients: int, n_edges: int, positions_c: np.ndarray, positions_e: np.ndarray,
                 n_samples: np.ndarray, capacity: np.ndarray, rewards: np.ndarray, costs: np.ndarray,
                 beta: float = 0.7, lam: float = 0.02, gamma: float = 0.01):
        self.nc, self.ne = n_clients, n_edges
        self.dist = np.linalg.norm(positions_c[:, None, :] - positions_e[None, :, :], axis=-1)
        self.n = n_samples.astype(np.float64)
        self.capacity = capacity
        self.R, self.cost = rewards, costs
        self.lam, self.gamma = lam, gamma
        self.quality = QualityEstimator(n_clients, beta=beta)

    def match(self) -> Dict[str, np.ndarray]:
        profit = client_profit(self.R, self.n, self.cost, self.dist, self.capacity, self.lam)
        score = edge_score(self.quality.Q, self.n, self.dist, self.gamma)
        acceptable = profit > 0
        assign = deferred_acceptance(profit, score, self.capacity, acceptable)
        # 용량 초과/비합리로 남은 클라이언트는 여유 용량이 있는 가장 가까운 엣지에 배치(참여 보장)
        load = np.bincount(assign[assign >= 0], minlength=self.ne)
        for i in np.where(assign < 0)[0]:
            for j in np.argsort(self.dist[i]):
                if load[j] < self.capacity[j]:
                    assign[i] = j
                    load[j] += 1
                    break
        # 전역 우선순위: 품질 추정량 기준 순위 (1 = 최상위)
        rank = np.empty(self.nc, dtype=np.int64)
        rank[np.argsort(-self.quality.Q)] = np.arange(1, self.nc + 1)
        # 각 엣지 내부 우선순위 (엣지 선호 점수 기준)
        edge_rank = np.zeros(self.nc, dtype=np.int64)
        for j in range(self.ne):
            m = np.where(assign == j)[0]
            if m.size:
                edge_rank[m[np.argsort(-score[m, j])]] = np.arange(1, m.size + 1)
        return {"assign": assign, "profit": profit, "score": score, "rank": rank,
                "edge_rank": edge_rank, "Q": self.quality.Q.copy(),
                "blocking": blocking_pairs(assign, profit, score, self.capacity, acceptable)}
