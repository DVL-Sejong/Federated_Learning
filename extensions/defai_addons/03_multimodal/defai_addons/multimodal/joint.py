"""공동 임베딩 (Joint Embedding) 학습 (1차년도).

- JointProjection : 모달리티별 특징을 공통 임베딩 공간으로 매핑 (토큰 단위 선형 사상 + LayerNorm)
                    및 차원 축소 헤드 (d → d_joint) 로 공동 임베딩 멀티모달 특징 도출
- contrastive_alignment : 같은 샘플의 서로 다른 모달리티 임베딩을 가깝게 하는 대칭 InfoNCE.
                          결측 모달리티 쌍은 제외한다.
- joint_consistency : 1차년도 연결 기술의 손실 ||a − c|| + ||b − c|| (a, b: 모달리티 피처, c: 결합 피처)
"""
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import MODALITIES


class JointProjection(nn.Module):
    def __init__(self, d: int = 64, d_joint: int = 32, modalities=MODALITIES):
        super().__init__()
        self.proj = nn.ModuleDict({m: nn.Sequential(nn.Linear(d, d), nn.LayerNorm(d)) for m in modalities})
        self.reduce = nn.ModuleDict({m: nn.Linear(d, d_joint) for m in modalities})

    def tokens(self, m: str, x: torch.Tensor) -> torch.Tensor:
        return self.proj[m](x)

    def embed(self, m: str, pooled: torch.Tensor) -> torch.Tensor:
        """차원 축소 + L2 정규화된 공동 임베딩."""
        return F.normalize(self.reduce[m](pooled), dim=-1)


def contrastive_alignment(emb: Dict[str, torch.Tensor], mask: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    """모든 모달리티 쌍에 대한 대칭 InfoNCE 평균. mask: (B, M) 모달리티 존재 여부."""
    losses: List[torch.Tensor] = []
    names = list(emb.keys())
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            both = mask[:, MODALITIES.index(names[i])] & mask[:, MODALITIES.index(names[j])]
            if both.sum() < 2:
                continue
            a, b = emb[names[i]][both], emb[names[j]][both]
            logits = a @ b.t() / temperature
            t = torch.arange(len(a), device=a.device)
            losses.append(0.5 * (F.cross_entropy(logits, t) + F.cross_entropy(logits.t(), t)))
    return torch.stack(losses).mean() if losses else torch.zeros((), device=mask.device)


def joint_consistency(feats: torch.Tensor, fused: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Σ_m ||z_m − c||  (존재하는 모달리티만). feats: (B, M, d), fused: (B, d)."""
    dist = (feats - fused.unsqueeze(1)).norm(dim=-1)  # (B, M)
    return (dist * mask.float()).sum(1).div(mask.float().sum(1).clamp(min=1)).mean()


@torch.no_grad()
def cross_modal_retrieval(emb_a: torch.Tensor, emb_b: torch.Tensor, y: torch.Tensor) -> float:
    """a → b 최근접 검색 시 같은 클래스가 검색되는 비율 (클래스 수준 R@1)."""
    sim = emb_a @ emb_b.t()
    sim.fill_diagonal_(-1e9)  # 자기 자신 쌍 제외 → 다른 샘플 중에서 의미적으로 맞는 것을 찾는지 평가
    return float((y[sim.argmax(1)] == y).float().mean())
