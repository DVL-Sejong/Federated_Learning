"""연합 컴퓨팅 환경의 보안·프라이버시 요소 (3차년도: 데이터 암호화 및 차분 프라이버시 적용).

- SecureAggregator : 쌍별 가산 마스크 기반 보안 집계 (Bonawitz et al., 2017 의 단순화 버전).
  서버는 마스킹된 업데이트만 받으며 합산 시 마스크가 상쇄되어 평문 합과 동일해진다.
- GaussianDP       : 업데이트 L2 클리핑 + 가우시안 노이즈 (DP-FedAvg), RDP 기반 ε 계산
"""
import math
from typing import Dict, List, Optional

import torch


class SecureAggregator:
    def __init__(self, n_clients: int, dim: int, seed: int = 0, scale: float = 1e3):
        self.n, self.dim, self.scale = n_clients, dim, scale
        g = torch.Generator().manual_seed(seed)
        # 실제 프로토콜에서는 Diffie-Hellman 으로 합의되는 쌍별 시드
        self.pair_seed = torch.randint(0, 2 ** 31 - 1, (n_clients, n_clients), generator=g)

    def _prg(self, i: int, j: int) -> torch.Tensor:
        a, b = min(i, j), max(i, j)
        g = torch.Generator().manual_seed(int(self.pair_seed[a, b]))
        return torch.randn(self.dim, generator=g, dtype=torch.float64) * self.scale

    def mask(self, i: int, vec: torch.Tensor, active: List[int]) -> torch.Tensor:
        # float64 로 마스킹해야 합산 시 상쇄 오차가 무시할 수준(~1e-12)이 된다
        m = torch.zeros(self.dim, dtype=torch.float64)
        for j in active:
            if j == i:
                continue
            m += self._prg(i, j) if i < j else -self._prg(i, j)
        return vec.double() + m

    def aggregate(self, masked: List[torch.Tensor]) -> torch.Tensor:
        return torch.stack(masked).sum(0).float()

    def overhead_bytes(self, n_active: int) -> int:
        # 공개키 교환 (32 bytes x 쌍) 만 추가 전송, 마스크는 시드로부터 로컬 생성
        return 32 * n_active * (n_active - 1)


class GaussianDP:
    def __init__(self, noise_multiplier: float, clip: float = 1.0, delta: float = 1e-5, seed: int = 0):
        self.sigma, self.clip, self.delta = noise_multiplier, clip, delta
        self.g = torch.Generator().manual_seed(seed)
        self.steps = 0

    def clip_update(self, u: torch.Tensor) -> torch.Tensor:
        n = float(u.norm())
        return u * min(1.0, self.clip / (n + 1e-12))

    def noise(self, dim: int, n_clients: int, max_weight: Optional[float] = None) -> torch.Tensor:
        """Fixed public weights, replace-one client adjacency: sensitivity = 2 C max(w).

        This accounting covers model updates conditional on fixed preprocessing;
        it does not make the unnoised preprocessing statistics differentially private.
        Seeded noise is for reproducible experiments, not production privacy.
        """
        weight = 1.0 / max(n_clients, 1) if max_weight is None else float(max_weight)
        if not 0 < weight <= 1:
            raise ValueError("max_weight must be in (0, 1]")
        self.last_sensitivity = 2.0 * self.clip * weight
        self.steps += 1
        return torch.randn(dim, generator=self.g) * (self.sigma * self.last_sensitivity)

    def epsilon(self, steps: Optional[int] = None) -> float:
        """전체 참여(샘플링 없음) 가우시안 메커니즘의 RDP 합성 → (ε, δ)-DP 변환."""
        T = self.steps if steps is None else steps
        if self.sigma <= 0 or T == 0:
            return float("inf") if T else 0.0
        best = float("inf")
        for a in [1.25, 1.5, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 128, 256]:
            eps = T * a / (2 * self.sigma ** 2) + math.log(1 / self.delta) / (a - 1)
            best = min(best, eps)
        return best

    def describe(self) -> Dict[str, object]:
        return {"noise_multiplier": self.sigma, "clip": self.clip, "delta": self.delta,
                "epsilon": self.epsilon(), "adjacency": "replace_one_client_fixed_public_weights",
                "sensitivity": getattr(self, "last_sensitivity", None),
                "scope": "model_updates_conditional_on_fixed_preprocessing",
                "noise_source": "seeded_simulation_not_production"}
