"""멀티모달 융합·연결 모듈.

1차년도 '공동 임베딩 기반의 멀티모달 피처 연결 기술 설계'
  DynamicCoAttention        셀프어텐션 + 크로스어텐션으로 모달리티 중요도를 평가하고 동적으로 가중
  HierarchicalFeatureFusion 1계층: 저차원(저수준) 피처 결합, 2계층: 고차원 피처 결합

2차년도 '어텐션 기법을 적용한 멀티모달 학습 모델 / 연결 구현'
  CoAttentionFusion         모달리티 쌍(이미지-텍스트, 텍스트-오디오 …)의 코어텐션 행렬 → 쌍 중요도 → 융합
  ModalityGAT               노드 = 모달리티, 엣지 = 모달리티 간 연관성. 그래프 어텐션으로 엣지 중요도 학습
  CrossModalAligner         크로스모달 어텐션 행렬 + 정규화(Sinkhorn)로 모달리티 간 정렬·매칭

3차년도 '트랜스포머 기반 멀티모달 학습 모델'
  GatedMultimodalFusion     게이트형 멀티모달 융합 (GMU, Arevalo et al. 2017) — 결측 모달리티 마스킹 지원
"""
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

NEG = -1e4


def masked_softmax(logits: torch.Tensor, mask: torch.Tensor, dim: int = -1) -> torch.Tensor:
    return torch.softmax(logits.masked_fill(~mask, NEG), dim=dim) * mask.float()


class DynamicCoAttention(nn.Module):
    """모달리티 벡터 z (B, M, d) 에 대해
        ctx_self  = SelfAttn(z)               — 모달리티 집합 내부 상호작용
        ctx_cross = CrossAttn(q=z, kv=fused0) — 초기 결합 표현과의 상호작용
        α_m = softmax_m( w^T tanh(W [z_m; ctx_self_m; ctx_cross_m]) )  (결측 모달리티 제외)
        c   = Σ_m α_m z_m
    """

    def __init__(self, d: int, n_heads: int = 4):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d, n_heads, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d, n_heads, batch_first=True)
        self.score = nn.Sequential(nn.Linear(3 * d, d), nn.Tanh(), nn.Linear(d, 1))

    def forward(self, z: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        pad = ~mask
        s, _ = self.self_attn(z, z, z, key_padding_mask=pad)
        fused0 = (z * mask.unsqueeze(-1)).sum(1, keepdim=True) / mask.sum(1, keepdim=True).unsqueeze(-1).clamp(min=1)
        c, _ = self.cross_attn(z, fused0, fused0)
        alpha = masked_softmax(self.score(torch.cat([z, s, c], -1)).squeeze(-1), mask)
        return (alpha.unsqueeze(-1) * z).sum(1), alpha


class HierarchicalFeatureFusion(nn.Module):
    """1계층에서 저수준 피처를, 2계층에서 고수준 피처와 1계층 결과를 결합."""

    def __init__(self, d: int, n_mod: int = 3):
        super().__init__()
        self.level1 = nn.Sequential(nn.Linear(n_mod * d, d), nn.ReLU(), nn.LayerNorm(d))
        self.level2 = nn.Sequential(nn.Linear((n_mod + 1) * d, d), nn.ReLU(), nn.LayerNorm(d))

    def forward(self, low: torch.Tensor, high: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        m = mask.unsqueeze(-1).float()
        l1 = self.level1((low * m).flatten(1))
        return self.level2(torch.cat([(high * m).flatten(1), l1], -1))


class CoAttentionFusion(nn.Module):
    """모달리티 쌍 (i, j) 마다 친화도 행렬 A = H_i W H_j^T 를 계산하고
       H_i' = softmax_row(A) H_j, H_j' = softmax_col(A)^T H_i 로 상호 주의를 기울인 뒤
       쌍 특징 p_ij = MLP([pool(H_i'); pool(H_j')]) 와 쌍 중요도 β_ij 로 가중 융합한다."""

    def __init__(self, d: int, n_mod: int = 3):
        super().__init__()
        self.pairs = [(i, j) for i in range(n_mod) for j in range(i + 1, n_mod)]
        self.W = nn.ParameterList([nn.Parameter(torch.eye(d) + 0.01 * torch.randn(d, d)) for _ in self.pairs])
        self.mlp = nn.Sequential(nn.Linear(2 * d, d), nn.ReLU(), nn.LayerNorm(d))
        self.score = nn.Linear(d, 1)
        self.scale = d ** -0.5

    def forward(self, H: List[torch.Tensor], pads: List[torch.Tensor], mask: torch.Tensor):
        feats, scores, valid, mats = [], [], [], {}
        for k, (i, j) in enumerate(self.pairs):
            A = (H[i] @ self.W[k] @ H[j].transpose(1, 2)) * self.scale  # (B, Li, Lj)
            A = A.masked_fill(pads[j].unsqueeze(1), NEG).masked_fill(pads[i].unsqueeze(2), NEG)
            hi = torch.softmax(A, 2) @ H[j]
            hj = torch.softmax(A, 1).transpose(1, 2) @ H[i]
            keep_i, keep_j = (~pads[i]).unsqueeze(-1).float(), (~pads[j]).unsqueeze(-1).float()
            p = self.mlp(torch.cat([(hi * keep_i).sum(1) / keep_i.sum(1).clamp(min=1),
                                    (hj * keep_j).sum(1) / keep_j.sum(1).clamp(min=1)], -1))
            feats.append(p)
            scores.append(self.score(p).squeeze(-1))
            valid.append(mask[:, i] & mask[:, j])
            mats[(i, j)] = A
        P, S, Vd = torch.stack(feats, 1), torch.stack(scores, 1), torch.stack(valid, 1)
        # 유효한 쌍이 없는 샘플(단일 모달리티)은 모든 쌍을 균등 사용 → 호출 측에서 잔차로 보완
        Vd_eff = Vd | (~Vd.any(1, keepdim=True))
        beta = masked_softmax(S, Vd_eff)
        return (beta.unsqueeze(-1) * P).sum(1), beta, mats


class ModalityGAT(nn.Module):
    """그래프 어텐션: e_ij = LeakyReLU(a^T [W h_i ‖ W h_j]),  α_ij = softmax_j(e_ij),  h_i' = ELU(Σ_j α_ij W h_j)."""

    def __init__(self, d: int, n_heads: int = 4):
        super().__init__()
        assert d % n_heads == 0
        self.h, self.dk = n_heads, d // n_heads
        self.W = nn.Linear(d, d, bias=False)
        self.a_src = nn.Parameter(torch.randn(n_heads, self.dk) * 0.1)
        self.a_dst = nn.Parameter(torch.randn(n_heads, self.dk) * 0.1)
        self.out = nn.Linear(d, d)
        self.norm = nn.LayerNorm(d)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, M, d = x.shape
        Wh = self.W(x).view(B, M, self.h, self.dk)
        e = F.leaky_relu((Wh * self.a_src).sum(-1).unsqueeze(2) + (Wh * self.a_dst).sum(-1).unsqueeze(1), 0.2)  # (B, M, M, h)
        adj = (mask.unsqueeze(1) & mask.unsqueeze(2)).unsqueeze(-1).expand_as(e)
        alpha = masked_softmax(e, adj, dim=2)
        h = torch.einsum("bijh,bjhk->bihk", alpha, Wh).reshape(B, M, d)
        return self.norm(x + self.out(F.elu(h))), alpha.mean(-1)  # (B, M, d), 엣지 중요도 (B, M, M)


class GatedMultimodalFusion(nn.Module):
    """GMU: h_m = tanh(W_m x_m),  g = softmax_m(W_g [x_1; …; x_M])  (결측 마스킹),  h = Σ_m g_m ⊙ h_m.
    원 논문의 시그모이드 게이트를 M 개 모달리티 + 결측 처리를 위해 모달리티 축 softmax 로 일반화했다."""

    def __init__(self, d: int, n_mod: int = 3):
        super().__init__()
        self.h = nn.ModuleList([nn.Linear(d, d) for _ in range(n_mod)])
        self.gate = nn.Linear(n_mod * d, n_mod * d)
        self.n_mod = n_mod

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, M, d = x.shape
        hs = torch.stack([torch.tanh(self.h[m](x[:, m])) for m in range(M)], 1)
        g = self.gate((x * mask.unsqueeze(-1)).flatten(1)).view(B, M, d)
        g = masked_softmax(g.transpose(1, 2), mask.unsqueeze(1).expand(B, d, M)).transpose(1, 2)
        return (g * hs).sum(1), g.mean(-1)  # 융합 벡터, 모달리티별 평균 게이트 (B, M)


class CrossModalAligner(nn.Module):
    """크로스모달 어텐션 기반 정렬 (2차년도).

    배치 내 모달리티 a 의 샘플들과 모달리티 b 의 샘플들 사이 어텐션 행렬을 계산하고
    Sinkhorn 정규화로 이중 확률 행렬을 만들어 정렬 오류(한 샘플로 몰림)를 줄인다.
    정렬된 행렬로 a 의 각 샘플과 가장 연관성 높은 b 샘플을 매칭한다.
    """

    def __init__(self, d: int, temperature: float = 0.1, iters: int = 5):
        super().__init__()
        self.q = nn.Linear(d, d)
        self.k = nn.Linear(d, d)
        self.t, self.iters = temperature, iters

    def attention(self, a: torch.Tensor, b: torch.Tensor, sinkhorn: bool = True) -> torch.Tensor:
        logits = F.normalize(self.q(a), dim=-1) @ F.normalize(self.k(b), dim=-1).t() / self.t
        if not sinkhorn:
            return torch.softmax(logits, 1)
        log_p = logits
        for _ in range(self.iters):
            log_p = log_p - torch.logsumexp(log_p, 1, keepdim=True)
            log_p = log_p - torch.logsumexp(log_p, 0, keepdim=True)
        return log_p.exp()

    def loss(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        P = self.attention(a, b)
        return -torch.log(P.diagonal() + 1e-8).mean()

    @torch.no_grad()
    def match(self, a: torch.Tensor, b: torch.Tensor, sinkhorn: bool = True) -> torch.Tensor:
        return self.attention(a, b, sinkhorn).argmax(1)
