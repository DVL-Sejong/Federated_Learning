"""Year-one and year-two comparison baselines; not the default model."""
import torch
import torch.nn as nn
from ..models import _Base
from ..data import MODALITIES
from ..encoders import masked_mean
from ..fusion import DynamicCoAttention, HierarchicalFeatureFusion, CoAttentionFusion, ModalityGAT

class JointEmbeddingModel(_Base):
    """1차년도: 공동 임베딩 + 다이나믹 코어텐션(동적 가중치) + 계층적 피처 융합."""

    def __init__(self, d: int = 64, **kw):
        kw.setdefault("lambda_cons", 0.05)
        super().__init__(d, **kw)
        self.coattn = DynamicCoAttention(d)
        self.hff = HierarchicalFeatureFusion(d)
        self.head = nn.Sequential(nn.Linear(2 * d, d), nn.ReLU(), nn.Dropout(0.1), nn.Linear(d, self.n_classes))

    def forward(self, batch):
        mask = batch["mask"]
        toks, pads, low, pooled, emb = self.encode(batch)
        c, alpha = self.coattn(pooled, mask)
        h = self.hff(low, pooled, mask)
        return {"logits": self.head(torch.cat([c, h], -1)), "emb": emb,
                "aux": self.aux_loss(emb, mask, pooled, c), "attn": {"modality_weight": alpha}}


class AttentionFusionModel(_Base):
    """2차년도: 모달리티별 셀프어텐션 인코더 → 코어텐션 퓨전(모달리티 쌍) + 그래프 어텐션(모달리티 그래프)."""

    def __init__(self, d: int = 64, n_heads: int = 4, **kw):
        super().__init__(d, **kw)
        self.self_enc = nn.ModuleList([nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d, n_heads, 2 * d, 0.1, batch_first=True), 1) for _ in MODALITIES])
        self.coattn = CoAttentionFusion(d)
        self.gat = ModalityGAT(d, n_heads)
        self.head = nn.Sequential(nn.Linear(2 * d, d), nn.ReLU(), nn.Dropout(0.1), nn.Linear(d, self.n_classes))

    def forward(self, batch):
        mask = batch["mask"]
        toks, pads, low, pooled, emb = self.encode(batch)
        H = [enc(t, src_key_padding_mask=p) for enc, t, p in zip(self.self_enc, toks, pads)]
        pair, beta, _ = self.coattn(H, pads, mask)
        Hp = torch.stack([masked_mean(h, p) for h, p in zip(H, pads)], 1) * mask.unsqueeze(-1)
        g, edge = self.gat(Hp, mask)
        gpool = (g * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp(min=1)
        return {"logits": self.head(torch.cat([pair, gpool], -1)), "emb": emb, "aux": self.aux_loss(emb, mask),
                "attn": {"pair_weight": beta, "graph_edge": edge}}
