"""
SEPS 模块审阅版。

与论文（arXiv:2511.01390）和开源代码的对应关系：
- 论文模式下真正使用稠密文本注意力参与打分（s_dt 不再为 0）。
- 聚合支持稀疏/稠密联合（Eq.4 的 Ws/Wd）；默认仍保持开源代码行为。
- 可选直通式 Gumbel-topk，用于论文里可微决策矩阵（默认关闭，与代码一致）。
- 论文模式下 HRPA 只用稀疏 caption 做相似度，稠密文本仅用于筛选。
默认 use_paper_version=False 时行为与 lib/cross_net.py、lib/xttn.py 保持兼容。
"""

import math
from typing import Iterable, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

USE_PAPER_VERSION_DEFAULT = False


def l2_normalize(x: torch.Tensor, dim: int = -1, eps: float = 1e-8) -> torch.Tensor:
    # 对应论文相似度计算前的 L2 归一化
    return F.normalize(x, p=2, dim=dim, eps=eps)


def is_sqr(n: int) -> bool:
    # 判断是否含有 [CLS]，与 ViT patch 数量相关
    a = int(math.sqrt(n))
    return a * a == n


class TokenSparse(nn.Module):
    """
    Patch 选择器（可选论文打分 + 直通 Gumbel-topk）。

    Args:
        embed_dim: 特征维度。
        sparse_ratio: 保留比例。
        use_paper_version: 是否启用论文公式(1-3) 的打分。
    """

    def __init__(
        self,
        embed_dim: int = 512,
        sparse_ratio: float = 0.6,
        use_paper_version: bool = USE_PAPER_VERSION_DEFAULT,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.sparse_ratio = sparse_ratio
        self.use_paper_version = use_paper_version

        if use_paper_version:
            self.score_predictor = nn.Sequential(
                nn.Linear(embed_dim, embed_dim // 4),
                nn.GELU(),
                nn.Linear(embed_dim // 4, 1),
                nn.Sigmoid(),
            )

    def forward(
        self,
        tokens: torch.Tensor,
        attention_x: torch.Tensor,
        attention_y: torch.Tensor,
        attention_y_dense: Optional[torch.Tensor] = None,
        beta: float = 0.25,
        use_gumbel: bool = False,
        gumbel_tau: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            tokens: (B, N, C) patch 特征。
            attention_x: (B, N) 图像自注意力。
            attention_y: (B, N) 稀疏文本注意力。
            attention_y_dense: (B, N) 稠密文本注意力（论文模式）。
            beta: Eq.(3) 中的权重。
            use_gumbel: 是否启用直通式 Gumbel-topk。
            gumbel_tau: Gumbel-softmax 温度。
        """
        B_v, L_v, C = tokens.size()

        if self.use_paper_version:
            # s_i^p = σ(MLP(v_i))，对应论文 Eq.(1)
            s_pred = self.score_predictor(tokens).squeeze(-1)

            def normalize_score(s: torch.Tensor) -> torch.Tensor:
                s_min = s.min(dim=-1, keepdim=True)[0]
                s_max = s.max(dim=-1, keepdim=True)[0]
                return (s - s_min) / (s_max - s_min + 1e-8)

            # s_i^{im}, s_i^{st}, s_i^{dt}，对应论文 Eq.(2) 归一化处理
            s_im = normalize_score(attention_x)
            s_st = normalize_score(attention_y)
            s_dt = (
                normalize_score(attention_y_dense)
                if attention_y_dense is not None
                else torch.zeros_like(s_st)
            )
            # s_i = (1-2β)s_i^p + β(s_i^{st}+s_i^{dt}+2 s_i^{im})，对应论文 Eq.(3)
            score = (1 - 2 * beta) * s_pred + beta * (s_st + s_dt + 2 * s_im)
        else:
            score = attention_x + attention_y

        num_keep_token = max(1, math.ceil(L_v * self.sparse_ratio))

        score_sort, score_index = torch.sort(score, dim=1, descending=True)
        keep_policy = score_index[:, :num_keep_token]

        if use_gumbel:
            # 可微决策矩阵：Gumbel-Softmax + 直通估计，对应论文中的“differentiable decision matrix”
            gumbel_noise = -torch.log(
                -torch.log(torch.rand_like(score) + 1e-9) + 1e-9
            )
            soft_mask = F.softmax((score + gumbel_noise) / gumbel_tau, dim=1)
            hard_mask = torch.zeros_like(score).scatter(1, keep_policy, 1.0)
            score_mask = hard_mask + (soft_mask - soft_mask.detach())
        else:
            score_mask = torch.zeros_like(score).scatter(1, keep_policy, 1.0)

        select_tokens = torch.gather(
            tokens, dim=1, index=keep_policy.unsqueeze(-1).expand(-1, -1, C)
        )

        non_keep_policy = score_index[:, num_keep_token:]
        non_tokens = torch.gather(
            tokens, dim=1, index=non_keep_policy.unsqueeze(-1).expand(-1, -1, C)
        )
        non_keep_score = score_sort[:, num_keep_token:]
        non_keep_score = F.softmax(non_keep_score, dim=1).unsqueeze(-1)
        extra_token = torch.sum(non_tokens * non_keep_score, dim=1, keepdim=True)

        return select_tokens, extra_token, score_mask


class TokenAggregation(nn.Module):
    """单分支聚合（与开源代码一致）。"""

    def __init__(self, dim: int = 512, keeped_patches: int = 64, dim_ratio: float = 0.2):
        super().__init__()
        hidden_dim = int(dim * dim_ratio)
        self.weight = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, keeped_patches),
        )
        self.scale = nn.Parameter(torch.ones(1, 1, 1))

    def forward(
        self, x: torch.Tensor, keep_policy: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # 计算单分支权重矩阵 W，论文 Eq.(4) 的简化共享版
        weight = self.weight(x).transpose(2, 1) * self.scale
        if keep_policy is not None:
            weight = weight - (1 - keep_policy.unsqueeze(1)) * 1e10
        weight = F.softmax(weight, dim=2)
        return torch.bmm(weight, x)


class DualTokenAggregation(nn.Module):
    """稀疏/稠密联合聚合（论文 Eq.4）。"""

    def __init__(self, dim: int = 512, keeped_patches: int = 64, dim_ratio: float = 0.2):
        super().__init__()
        hidden_dim = int(dim * dim_ratio)
        self.weight_sparse = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, keeped_patches),
        )
        self.weight_dense = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, keeped_patches),
        )
        self.scale = nn.Parameter(torch.ones(1, 1, 1))

    def _aggregate(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor],
        weight_net: nn.Module,
    ) -> torch.Tensor:
        # Ws/Wd = Softmax(MLP(Vs/Vd))，对应 Eq.(4) 的权重学习
        weight = weight_net(x).transpose(2, 1) * self.scale
        if mask is not None:
            weight = weight - (1 - mask.unsqueeze(1)) * 1e10
        weight = F.softmax(weight, dim=2)
        return torch.bmm(weight, x)

    def forward(
        self,
        tokens_sparse: torch.Tensor,
        tokens_dense: Optional[torch.Tensor] = None,
        mask_sparse: Optional[torch.Tensor] = None,
        mask_dense: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # 传入的是已筛选的 tokens，通常不再需要 mask；如需屏蔽 padding，可提供 mask_* 与 tokens_* 同长度
        out = self._aggregate(tokens_sparse, mask_sparse, self.weight_sparse)
        if tokens_dense is not None:
            out = out + self._aggregate(tokens_dense, mask_dense, self.weight_dense)
        return out


def mask_xattn_one_text(
    img_embs: torch.Tensor,
    cap_i_expand: torch.Tensor,
    img_mask: Optional[torch.Tensor] = None,
    i2t: bool = True,
    scan: bool = True,
    use_paper_version: bool = USE_PAPER_VERSION_DEFAULT,
    top_k: int = 5,
    relevance_mlp: Optional[nn.Module] = None,
) -> torch.Tensor:
    """HRPA 相似度（可选 TopK+MLP）。"""
    cap2img_sim = torch.bmm(cap_i_expand, img_embs.transpose(1, 2))
    if scan:
        cap2img_sim = F.leaky_relu(cap2img_sim, negative_slope=0.1)

    if img_mask is None:
        row_sim = cap2img_sim.max(dim=2)[0]
    else:
        row_sim = (cap2img_sim - 1000 * (1 - img_mask).unsqueeze(1)).max(dim=2)[0]
    row_sim_mean = row_sim.mean(dim=1, keepdim=True)

    if use_paper_version and relevance_mlp is not None:
        B_v, M = row_sim.shape
        k = min(top_k, M)
        row_topk, _ = row_sim.topk(k, dim=1)
        if k < top_k:
            padding = torch.zeros(B_v, top_k - k, device=row_topk.device)
            row_topk = torch.cat([row_topk, padding], dim=1)
        row_sim_mean = row_sim_mean + relevance_mlp(row_topk)

    if i2t:
        column_sim = cap2img_sim.max(dim=1)[0]
        if img_mask is None:
            column_sim_mean = column_sim.mean(dim=1, keepdim=True)
        else:
            column_sim_mean = (column_sim * img_mask).sum(dim=-1, keepdim=True) / (
                img_mask.sum(dim=-1, keepdim=True) + 1e-8
            )

        if use_paper_version and relevance_mlp is not None:
            # TopK + MLP，对应论文 Eq.(5) “MLP(TOPK(max(A)))” 部分
            B_v, N = column_sim.shape
            k = min(top_k, N)
            col_topk, _ = column_sim.topk(k, dim=1)
            if k < top_k:
                padding = torch.zeros(B_v, top_k - k, device=col_topk.device)
                col_topk = torch.cat([col_topk, padding], dim=1)
            column_sim_mean = column_sim_mean + relevance_mlp(col_topk)

        sim_one_text = row_sim_mean + column_sim_mean
    else:
        sim_one_text = row_sim_mean

    return sim_one_text


class HRPA(nn.Module):
    def __init__(
        self,
        embed_dim: int = 512,
        top_k: int = 5,
        use_paper_version: bool = USE_PAPER_VERSION_DEFAULT,
        bidirectional: bool = True,
        scan: bool = True,
    ):
        super().__init__()
        self.use_paper_version = use_paper_version
        self.bidirectional = bidirectional
        self.top_k = top_k
        self.scan = scan

        if use_paper_version:
            self.relevance_mlp = nn.Sequential(
                nn.Linear(top_k, top_k * 2),
                nn.GELU(),
                nn.Linear(top_k * 2, 1),
            )
        else:
            self.relevance_mlp = None

    def forward(
        self,
        patch_features: torch.Tensor,
        word_features: torch.Tensor,
        patch_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # 对应论文 Eq.(5) 的双向 max+mean + TopK-MLP
        return mask_xattn_one_text(
            img_embs=patch_features,
            cap_i_expand=word_features,
            img_mask=patch_mask,
            i2t=self.bidirectional,
            scan=self.scan,
            use_paper_version=self.use_paper_version,
            top_k=self.top_k,
            relevance_mlp=self.relevance_mlp,
        )


class CrossSparseAggrNet(nn.Module):
    """完整 SDTPS + HRPA 流程，保留代码版/论文版双模式。"""

    def __init__(
        self,
        embed_size: int = 512,
        num_patches: int = 196,
        sparse_ratio: float = 0.5,
        aggr_ratio: float = 0.4,
        use_paper_version: bool = USE_PAPER_VERSION_DEFAULT,
        top_k: int = 5,
        use_gumbel_softmax: bool = False,
        gumbel_tau: float = 1.0,
        use_dual_aggr: bool = True,
        beta: float = 0.25,
    ):
        super().__init__()
        self.hidden_dim = embed_size
        self.num_patches = num_patches
        self.sparse_ratio = sparse_ratio
        self.aggr_ratio = aggr_ratio
        self.use_paper_version = use_paper_version
        self.top_k = top_k
        self.use_gumbel_softmax = use_gumbel_softmax
        self.gumbel_tau = gumbel_tau
        self.use_dual_aggr = use_dual_aggr
        self.beta = beta

        self.keeped_patches = int(num_patches * aggr_ratio * sparse_ratio)

        self.sparse_net_cap = TokenSparse(
            embed_dim=self.hidden_dim,
            sparse_ratio=self.sparse_ratio,
            use_paper_version=use_paper_version,
        )
        self.sparse_net_long = TokenSparse(
            embed_dim=self.hidden_dim,
            sparse_ratio=self.sparse_ratio,
            use_paper_version=use_paper_version,
        )

        if use_paper_version and use_dual_aggr:
            self.aggr_net = DualTokenAggregation(
                dim=self.hidden_dim, keeped_patches=self.keeped_patches
            )
        else:
            self.aggr_net = TokenAggregation(
                dim=self.hidden_dim, keeped_patches=self.keeped_patches
            )

        self.hrpa = HRPA(
            embed_dim=self.hidden_dim,
            top_k=self.top_k,
            use_paper_version=self.use_paper_version,
            bidirectional=True,
            scan=not self.use_paper_version,  # paper describes cosine, code keeps LeakyReLU
        )

    def forward(
        self,
        img_embs: torch.Tensor,
        cap_embs: torch.Tensor,
        cap_lens: torch.Tensor,
        long_cap_embs: Optional[torch.Tensor] = None,
        long_cap_lens: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Union[torch.Tensor, Tuple[torch.Tensor, ...]]]]:
        B_v, L_v, C = img_embs.shape

        # Step 0: 特征 L2 归一化（与 HRPA 余弦一致）
        img_embs_norm = F.normalize(img_embs, dim=-1)
        cap_embs_norm = F.normalize(cap_embs, dim=-1)
        long_cap_embs_norm = (
            F.normalize(long_cap_embs, dim=-1)
            if long_cap_embs is not None
            else None
        )

        self.has_cls_token = not is_sqr(img_embs.shape[1])
        if self.has_cls_token:
            img_cls_emb = img_embs[:, 0:1, :]
            img_spatial_embs = img_embs[:, 1:, :]
            img_spatial_embs_norm = img_embs_norm[:, 1:, :]
        else:
            img_cls_emb = None
            img_spatial_embs = img_embs
            img_spatial_embs_norm = img_embs_norm

        with torch.no_grad():
            # s_i^{im}: 由全局均值得到的自注意力得分，Eq.(2) 的视觉项
            img_spatial_glo_norm = F.normalize(
                img_spatial_embs.mean(dim=1, keepdim=True), dim=-1
            )
            img_spatial_self_attention = (
                img_spatial_glo_norm * img_spatial_embs_norm
            ).sum(dim=-1)

        improve_sims = []
        long_sims = []
        score_mask_all = []
        score_mask_long_all = []

        for i in range(len(cap_lens)):
            n_word = int(cap_lens[i])
            cap_i = cap_embs[i, :n_word, :]
            cap_i_expand = cap_embs_norm[i, :n_word, :].unsqueeze(0).repeat(B_v, 1, 1)

            dense_attn = None
            long_cap_i_expand = None
            select_tokens_long = extra_token_long = score_mask_long = None

            with torch.no_grad():
                cap_i_glo = F.normalize(cap_i.mean(0, keepdim=True).unsqueeze(0), dim=-1)
                attn_cap = (cap_i_glo * img_spatial_embs_norm).sum(dim=-1)

                if long_cap_embs_norm is not None and long_cap_lens is not None:
                    n_word_long = int(long_cap_lens[i])
                    long_cap_i = long_cap_embs[i, :n_word_long, :]
                    long_cap_i_expand = (
                        long_cap_embs_norm[i, :n_word_long, :]
                        .unsqueeze(0)
                        .repeat(B_v, 1, 1)
                    )
                    long_cap_i_glo = F.normalize(
                        long_cap_i.mean(0, keepdim=True).unsqueeze(0), dim=-1
                    )
                    dense_attn = (long_cap_i_glo * img_spatial_embs_norm).sum(dim=-1)

            # Stage 1: SDTPS 语义打分 + 选择，稀疏/稠密互为辅助，复现 Eq.(3)
            select_tokens_cap, extra_token_cap, score_mask_cap = self.sparse_net_cap(
                tokens=img_spatial_embs,
                attention_x=img_spatial_self_attention,
                attention_y=attn_cap,
                attention_y_dense=dense_attn if self.use_paper_version else None,
                beta=self.beta,
                use_gumbel=self.use_gumbel_softmax,
                gumbel_tau=self.gumbel_tau,
            )

            if dense_attn is not None:
                select_tokens_long, extra_token_long, score_mask_long = self.sparse_net_long(
                    tokens=img_spatial_embs,
                    attention_x=img_spatial_self_attention,
                    attention_y=dense_attn,
                    attention_y_dense=attn_cap if self.use_paper_version else None,
                    beta=self.beta,
                    use_gumbel=self.use_gumbel_softmax,
                    gumbel_tau=self.gumbel_tau,
                )

            if self.use_paper_version:
                if (
                    self.use_dual_aggr
                    and select_tokens_long is not None
                    and extra_token_long is not None
                ):
                    aggr_tokens = self.aggr_net(
                        select_tokens_cap,
                        select_tokens_long,
                        None,
                        None,
                    )
                    extra_token = torch.stack(
                        [extra_token_cap, extra_token_long], dim=0
                    ).mean(dim=0)
                    mask_pack: Union[torch.Tensor, Tuple[torch.Tensor, ...]] = (
                        score_mask_cap,
                        score_mask_long,
                    )
                else:
                    aggr_tokens = self.aggr_net(select_tokens_cap)
                    extra_token = extra_token_cap
                    mask_pack = (score_mask_cap,)

                keep_spatial_tokens = torch.cat([aggr_tokens, extra_token], dim=1)
                if self.has_cls_token:
                    select_tokens = torch.cat((img_cls_emb, keep_spatial_tokens), dim=1)
                else:
                    select_tokens = keep_spatial_tokens
                select_tokens = F.normalize(select_tokens, dim=-1)

                # Stage 2: HRPA 相似度（论文 Eq.(5)）
                sim_one_text = self.hrpa(
                    patch_features=select_tokens,
                    word_features=cap_i_expand,
                )

                improve_sims.append(sim_one_text)
                score_mask_all.append(mask_pack)
            else:
                aggr_tokens = self.aggr_net(select_tokens_cap)
                keep_spatial_tokens = torch.cat([aggr_tokens, extra_token_cap], dim=1)
                if self.has_cls_token:
                    select_tokens = torch.cat((img_cls_emb, keep_spatial_tokens), dim=1)
                else:
                    select_tokens = keep_spatial_tokens
                select_tokens = F.normalize(select_tokens, dim=-1)

                sim_one_text = self.hrpa(
                    patch_features=select_tokens,
                    word_features=cap_i_expand,
                )
                improve_sims.append(sim_one_text)
                score_mask_all.append(score_mask_cap)

                if select_tokens_long is not None and long_cap_i_expand is not None:
                    aggr_tokens_long = self.aggr_net(select_tokens_long)
                    keep_spatial_tokens = torch.cat(
                        [aggr_tokens_long, extra_token_long], dim=1
                    )
                    if self.has_cls_token:
                        select_tokens_long_final = torch.cat(
                            (img_cls_emb, keep_spatial_tokens), dim=1
                        )
                    else:
                        select_tokens_long_final = keep_spatial_tokens
                    select_tokens_long_final = F.normalize(
                        select_tokens_long_final, dim=-1
                    )

                    sim_one_text = self.hrpa(
                        patch_features=select_tokens_long_final,
                        word_features=long_cap_i_expand,
                    )
                    long_sims.append(sim_one_text)
                    score_mask_long_all.append(score_mask_long)

        improve_sims = torch.cat(improve_sims, dim=1)

        if not self.use_paper_version and long_sims:
            improve_sims = improve_sims + torch.cat(long_sims, dim=1)

        if self.use_paper_version:
            sparse_masks_list = [
                m[0] if isinstance(m, tuple) else m for m in score_mask_all
            ]
            dense_masks_list = [
                m[1]
                for m in score_mask_all
                if isinstance(m, tuple) and len(m) > 1
            ]
            sparse_stack = torch.stack(sparse_masks_list, dim=0)
            dense_stack = (
                torch.stack(dense_masks_list, dim=0) if len(dense_masks_list) > 0 else None
            )
            score_mask_out: Union[torch.Tensor, Tuple[torch.Tensor, ...]]
            if dense_stack is not None:
                score_mask_out = (sparse_stack, dense_stack)
            else:
                score_mask_out = (sparse_stack,)
        else:
            score_mask_out = torch.stack(score_mask_all, dim=0)
            if score_mask_long_all:
                score_mask_out = score_mask_out + torch.stack(score_mask_long_all, dim=0)

        if self.training:
            return improve_sims, score_mask_out
        return improve_sims


class ContrastiveLoss(nn.Module):
    def __init__(self, margin: float = 0.2, max_violation: bool = False):
        super().__init__()
        self.margin = margin
        self.max_violation = max_violation
        self.mask_repeat = True

    def max_violation_on(self):
        self.max_violation = True

    def max_violation_off(self):
        self.max_violation = False

    def forward(
        self, scores: torch.Tensor, img_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        diagonal = scores.diag().view(scores.size(0), 1)
        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)

        cost_s = (self.margin + scores - d1).clamp(min=0)
        cost_im = (self.margin + scores - d2).clamp(min=0)

        if not self.mask_repeat:
            mask = torch.eye(scores.size(0), dtype=torch.bool, device=scores.device)
        else:
            if img_ids is not None:
                mask = img_ids.unsqueeze(0) == img_ids.unsqueeze(1)
            else:
                mask = torch.eye(scores.size(0), dtype=torch.bool, device=scores.device)

        cost_s = cost_s.masked_fill_(mask, 0)
        cost_im = cost_im.masked_fill_(mask, 0)

        if self.max_violation:
            cost_s = cost_s.max(dim=1)[0]
            cost_im = cost_im.max(dim=0)[0]

        return cost_s.sum() + cost_im.sum()


class RatioLoss(nn.Module):
    """
    比例约束（论文 Eq.7，可对稀疏/稠密分别加权）。
    """

    def __init__(
        self,
        target_ratio: float = 0.5,
        lambda_sparse: float = 1.0,
        lambda_dense: float = 1.0,
    ):
        super().__init__()
        self.target_ratio = target_ratio
        self.lambda_sparse = lambda_sparse
        self.lambda_dense = lambda_dense

    def forward(
        self, score_mask: Union[torch.Tensor, Tuple[torch.Tensor, ...]]
    ) -> torch.Tensor:
        if isinstance(score_mask, tuple) or isinstance(score_mask, list):
            sparse_mask = score_mask[0]
            dense_mask = score_mask[1] if len(score_mask) > 1 else None
            sparse_loss = (sparse_mask.float().mean() - self.target_ratio) ** 2
            dense_loss = (
                (dense_mask.float().mean() - self.target_ratio) ** 2
                if dense_mask is not None
                else torch.tensor(0.0, device=sparse_mask.device)
            )
            return self.lambda_sparse * sparse_loss + self.lambda_dense * dense_loss
        return (score_mask.float().mean() - self.target_ratio) ** 2


class SEPSLoss(nn.Module):
    def __init__(
        self,
        margin: float = 0.2,
        target_ratio: float = 0.5,
        ratio_weight: float = 2.0,
        max_violation: bool = False,
        lambda_sparse: float = 1.0,
        lambda_dense: float = 1.0,
    ):
        super().__init__()
        self.contrastive_loss = ContrastiveLoss(
            margin=margin, max_violation=max_violation
        )
        self.ratio_loss = RatioLoss(
            target_ratio=target_ratio,
            lambda_sparse=lambda_sparse,
            lambda_dense=lambda_dense,
        )
        self.ratio_weight = ratio_weight

    def set_max_violation(self, max_violation: bool = True):
        if max_violation:
            self.contrastive_loss.max_violation_on()
        else:
            self.contrastive_loss.max_violation_off()

    def forward(
        self,
        similarity_matrix: torch.Tensor,
        score_mask: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
        img_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        align_loss = self.contrastive_loss(similarity_matrix, img_ids)
        r_loss = self.ratio_loss(score_mask)
        total_loss = align_loss + self.ratio_weight * r_loss
        return total_loss, align_loss, r_loss


SDTPS = CrossSparseAggrNet
SDTPS_TokenSparse = TokenSparse
SDTPS_TokenAggregation = TokenAggregation
HRPA_function = mask_xattn_one_text


def create_seps_model(
    embed_size: int = 512,
    num_patches: int = 196,
    sparse_ratio: float = 0.5,
    aggr_ratio: float = 0.4,
    use_paper_version: bool = USE_PAPER_VERSION_DEFAULT,
    use_gumbel_softmax: bool = False,
    gumbel_tau: float = 1.0,
) -> CrossSparseAggrNet:
    return CrossSparseAggrNet(
        embed_size=embed_size,
        num_patches=num_patches,
        sparse_ratio=sparse_ratio,
        aggr_ratio=aggr_ratio,
        use_paper_version=use_paper_version,
        use_gumbel_softmax=use_gumbel_softmax,
        gumbel_tau=gumbel_tau,
    )


if __name__ == "__main__":
    # 简单形状自检
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    B = 2
    num_patches = 16
    embed_size = 32
    sparse_len = 5
    dense_len = 8

    img = torch.randn(B, num_patches + 1, embed_size, device=device)
    cap = torch.randn(B, sparse_len, embed_size, device=device)
    cap_len = torch.full((B,), sparse_len, device=device)
    long_cap = torch.randn(B, dense_len, embed_size, device=device)
    long_len = torch.full((B,), dense_len, device=device)

    seps_code = create_seps_model(
        embed_size=embed_size,
        num_patches=num_patches,
        sparse_ratio=0.5,
        aggr_ratio=0.5,
        use_paper_version=False,
    ).to(device)
    seps_code.train()
    sims, mask = seps_code(img, cap, cap_len, long_cap, long_len)
    print("code version:", sims.shape, mask[0].shape if isinstance(mask, tuple) else mask.shape)

    seps_paper = create_seps_model(
        embed_size=embed_size,
        num_patches=num_patches,
        sparse_ratio=0.5,
        aggr_ratio=0.5,
        use_paper_version=True,
    ).to(device)
    seps_paper.train()
    sims, mask = seps_paper(img, cap, cap_len, long_cap, long_len)
    print(
        "paper version:",
        sims.shape,
        tuple(m.shape for m in mask) if isinstance(mask, tuple) else mask.shape,
    )
