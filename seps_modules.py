"""
SEPS: Semantic-Enhanced Patch Slimming Framework
================================================
论文核心创新模块的独立实现，方便在其他项目中调用。

提供两种实现模式（通过 use_paper_version 参数控制）：
- use_paper_version=True:  按论文描述实现（包含MLP预测网络、TopK相关性学习等）
- use_paper_version=False: 按实际代码实现（简化版本）

论文: SEPS: Semantic-Enhanced Patch Slimming Framework for Fine-Grained Cross-Modal Alignment
发表: ICLR 2026

核心创新点:
1. SDTPS (Sparse and Dense Text-Aware Patch Selection):
   - 融合稀疏文本(原始caption)和稠密文本(MLLM生成)的语义信息
   - 两阶段机制: 语义评分 + 决策聚合

2. HRPA (Highly-Relevant Patch-Word Alignment):
   - 双向对齐策略
   - patch-to-word + word-to-patch

使用示例:
    >>> from seps_modules import CrossSparseAggrNet, SEPSLoss
    >>>
    >>> # 使用实际代码版本（默认）
    >>> seps = CrossSparseAggrNet(embed_size=512, num_patches=196, use_paper_version=False)
    >>>
    >>> # 使用论文描述版本
    >>> seps_paper = CrossSparseAggrNet(embed_size=512, num_patches=196, use_paper_version=True)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Union


# =============================================================================
# 全局开关：控制使用论文版本还是实际代码版本
# =============================================================================

USE_PAPER_VERSION_DEFAULT = False  # 默认使用实际代码版本


# =============================================================================
# 辅助函数
# =============================================================================

def l2_normalize(x: torch.Tensor, dim: int = -1, eps: float = 1e-8) -> torch.Tensor:
    """L2归一化"""
    return F.normalize(x, p=2, dim=dim, eps=eps)


def is_sqr(n: int) -> bool:
    """检查n是否为完全平方数，用于判断是否有cls token"""
    a = int(math.sqrt(n))
    return a * a == n


# =============================================================================
# SDTPS模块组件1: TokenSparse (Patch稀疏选择)
# =============================================================================

class TokenSparse(nn.Module):
    """
    Token稀疏选择模块

    use_paper_version=True时:
        - 使用Score-aware Prediction Network (MLP) 预测每个patch的语义显著性
        - 综合MLP预测分数、图像自注意力、文本交叉注意力（论文公式1-3）

    use_paper_version=False时:
        - 直接使用注意力分数相加（实际代码实现）

    Args:
        embed_dim: 特征维度
        sparse_ratio: 保留patch的比例
        use_paper_version: 是否使用论文描述的复杂机制
    """

    def __init__(
        self,
        embed_dim: int = 512,
        sparse_ratio: float = 0.6,
        use_paper_version: bool = USE_PAPER_VERSION_DEFAULT
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.sparse_ratio = sparse_ratio
        self.use_paper_version = use_paper_version

        # 论文版本：Score-aware Prediction Network (公式1)
        if use_paper_version:
            self.score_predictor = nn.Sequential(
                nn.Linear(embed_dim, embed_dim // 4),
                nn.GELU(),
                nn.Linear(embed_dim // 4, 1),
                nn.Sigmoid()
            )

    def forward(
        self,
        tokens: torch.Tensor,
        attention_x: torch.Tensor,
        attention_y: torch.Tensor,
        attention_y_dense: Optional[torch.Tensor] = None,
        beta: float = 0.25
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        执行patch稀疏选择

        Args:
            tokens: (B, N, C) 视觉patch特征
            attention_x: (B, N) 图像自注意力得分 (s_im)
            attention_y: (B, N) 稀疏文本交叉注意力得分 (s_st)
            attention_y_dense: (B, N) 稠密文本交叉注意力得分 (s_dt)，仅论文版本使用
            beta: 权重参数，仅论文版本使用

        Returns:
            select_tokens: (B, N_keep, C) 选中的patch特征
            extra_token: (B, 1, C) 融合token
            score_mask: (B, N) 选择mask
        """
        B_v, L_v, C = tokens.size()

        if self.use_paper_version:
            # ===== 论文版本：公式(1)-(3) =====
            # 公式(1): s_i^p = σ(MLP(v_i))
            s_pred = self.score_predictor(tokens).squeeze(-1)  # (B, N)

            # 归一化注意力分数到[0,1]
            def normalize_score(s):
                s_min = s.min(dim=-1, keepdim=True)[0]
                s_max = s.max(dim=-1, keepdim=True)[0]
                return (s - s_min) / (s_max - s_min + 1e-8)

            s_im = normalize_score(attention_x)
            s_st = normalize_score(attention_y)
            s_dt = normalize_score(attention_y_dense) if attention_y_dense is not None else torch.zeros_like(s_st)

            # 公式(3): s_i = (1-2β)·s_i^p + β·(s_i^st + s_i^dt + 2·s_i^im)
            score = (1 - 2*beta) * s_pred + beta * (s_st + s_dt + 2*s_im)
        else:
            # ===== 实际代码版本：直接相加 =====
            score = attention_x + attention_y

        # 计算保留的patch数量
        num_keep_token = math.ceil(L_v * self.sparse_ratio)

        # 按得分降序排序
        score_sort, score_index = torch.sort(score, dim=1, descending=True)

        # 保留top-k的索引
        keep_policy = score_index[:, :num_keep_token]

        # 生成选择mask
        score_mask = torch.zeros_like(score).scatter(1, keep_policy, 1)

        # 选择保留的patches
        select_tokens = torch.gather(
            tokens, dim=1,
            index=keep_policy.unsqueeze(-1).expand(-1, -1, C)
        )

        # 获取被丢弃的patches并融合为extra token
        non_keep_policy = score_index[:, num_keep_token:]
        non_tokens = torch.gather(
            tokens, dim=1,
            index=non_keep_policy.unsqueeze(-1).expand(-1, -1, C)
        )
        non_keep_score = score_sort[:, num_keep_token:]
        non_keep_score = F.softmax(non_keep_score, dim=1).unsqueeze(-1)
        extra_token = torch.sum(non_tokens * non_keep_score, dim=1, keepdim=True)

        return select_tokens, extra_token, score_mask


# =============================================================================
# SDTPS模块组件2: TokenAggregation (Patch聚合)
# =============================================================================

class TokenAggregation(nn.Module):
    """
    Token聚合模块 - 学习权重矩阵将N个输入patches聚合为N_c个输出patches

    论文公式(4): v̂_j = Σ W_ij · v_i

    Args:
        dim: 特征维度
        keeped_patches: 输出patch数量 (N_c)
        dim_ratio: 隐藏层维度比例
    """

    def __init__(
        self,
        dim: int = 512,
        keeped_patches: int = 64,
        dim_ratio: float = 0.2
    ):
        super().__init__()

        hidden_dim = int(dim * dim_ratio)

        self.weight = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, keeped_patches)
        )

        self.scale = nn.Parameter(torch.ones(1, 1, 1))

    def forward(
        self,
        x: torch.Tensor,
        keep_policy: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        聚合patches

        Args:
            x: (B, N, C) 输入patch特征
            keep_policy: (B, N) 可选的mask

        Returns:
            aggregated: (B, N_c, C) 聚合后的patch特征
        """
        weight = self.weight(x)
        weight = weight.transpose(2, 1) * self.scale

        if keep_policy is not None:
            keep_policy = keep_policy.unsqueeze(1)
            weight = weight - (1 - keep_policy) * 1e10

        weight = F.softmax(weight, dim=2)
        x = torch.bmm(weight, x)

        return x


# =============================================================================
# HRPA模块: Highly-Relevant Patch-Word Alignment
# =============================================================================

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
    """
    HRPA: 高相关性Patch-Word对齐

    use_paper_version=True时:
        - 使用论文公式(5): mean + MLP(TopK(...))

    use_paper_version=False时:
        - 仅使用 max + mean（实际代码实现）

    Args:
        img_embs: (B_v, N, C) 已归一化的视觉patch特征
        cap_i_expand: (B_v, M, C) 已归一化的文本word特征
        img_mask: (B_v, N) 可选的patch mask
        i2t: 是否计算i2t方向
        scan: 是否使用LeakyReLU
        use_paper_version: 是否使用论文描述的复杂机制
        top_k: TopK参数，仅论文版本使用
        relevance_mlp: 相关性学习网络，仅论文版本使用

    Returns:
        sim_one_text: (B_v, 1) 图像-文本相似度分数
    """
    # 计算patch-word相似度矩阵 (B_v, M, N)
    cap2img_sim = torch.bmm(cap_i_expand, img_embs.transpose(1, 2))

    if scan:
        cap2img_sim = F.leaky_relu(cap2img_sim, negative_slope=0.1)

    # ===== t2i: Patch-to-Word 对齐 =====
    if img_mask is None:
        row_sim = cap2img_sim.max(dim=2)[0]  # (B_v, M)
    else:
        row_sim = (cap2img_sim - 1000 * (1 - img_mask).unsqueeze(1)).max(dim=2)[0]

    row_sim_mean = row_sim.mean(dim=1, keepdim=True)  # (B_v, 1)

    # 论文版本：添加TopK + MLP
    if use_paper_version and relevance_mlp is not None:
        B_v, M = row_sim.shape
        k = min(top_k, M)
        row_topk, _ = row_sim.topk(k, dim=1)
        if k < top_k:
            padding = torch.zeros(B_v, top_k - k, device=row_topk.device)
            row_topk = torch.cat([row_topk, padding], dim=1)
        row_extra = relevance_mlp(row_topk)  # (B_v, 1)
        row_sim_mean = row_sim_mean + row_extra

    if i2t:
        # ===== i2t: Word-to-Patch 对齐 =====
        column_sim = cap2img_sim.max(dim=1)[0]  # (B_v, N)

        if img_mask is None:
            column_sim_mean = column_sim.mean(dim=1, keepdim=True)
        else:
            column_sim_mean = (column_sim * img_mask).sum(dim=-1, keepdim=True) / \
                              (img_mask.sum(dim=-1, keepdim=True) + 1e-8)

        # 论文版本：添加TopK + MLP
        if use_paper_version and relevance_mlp is not None:
            B_v, N = column_sim.shape
            k = min(top_k, N)
            col_topk, _ = column_sim.topk(k, dim=1)
            if k < top_k:
                padding = torch.zeros(B_v, top_k - k, device=col_topk.device)
                col_topk = torch.cat([col_topk, padding], dim=1)
            col_extra = relevance_mlp(col_topk)
            column_sim_mean = column_sim_mean + col_extra

        sim_one_text = row_sim_mean + column_sim_mean
    else:
        sim_one_text = row_sim_mean

    return sim_one_text


class HRPA(nn.Module):
    """
    HRPA模块的类封装版本

    Args:
        embed_dim: 特征维度
        top_k: TopK参数
        use_paper_version: 是否使用论文描述的复杂机制
        bidirectional: 是否使用双向对齐
    """

    def __init__(
        self,
        embed_dim: int = 512,
        top_k: int = 5,
        use_paper_version: bool = USE_PAPER_VERSION_DEFAULT,
        bidirectional: bool = True
    ):
        super().__init__()
        self.use_paper_version = use_paper_version
        self.bidirectional = bidirectional
        self.top_k = top_k

        # 论文版本：Relevance Learning Network
        if use_paper_version:
            self.relevance_mlp = nn.Sequential(
                nn.Linear(top_k, top_k * 2),
                nn.GELU(),
                nn.Linear(top_k * 2, 1)
            )
        else:
            self.relevance_mlp = None

    def forward(
        self,
        patch_features: torch.Tensor,
        word_features: torch.Tensor,
        patch_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        return mask_xattn_one_text(
            img_embs=patch_features,
            cap_i_expand=word_features,
            img_mask=patch_mask,
            i2t=self.bidirectional,
            scan=True,
            use_paper_version=self.use_paper_version,
            top_k=self.top_k,
            relevance_mlp=self.relevance_mlp
        )


# =============================================================================
# SDTPS完整模块: CrossSparseAggrNet
# =============================================================================

class CrossSparseAggrNet(nn.Module):
    """
    完整的SDTPS模块

    Args:
        embed_size: 特征维度
        num_patches: 输入patch数量
        sparse_ratio: patch稀疏选择比例
        aggr_ratio: patch聚合比例
        use_paper_version: 是否使用论文描述的复杂机制
        top_k: HRPA的TopK参数（仅论文版本）
    """

    def __init__(
        self,
        embed_size: int = 512,
        num_patches: int = 196,
        sparse_ratio: float = 0.5,
        aggr_ratio: float = 0.4,
        use_paper_version: bool = USE_PAPER_VERSION_DEFAULT,
        top_k: int = 5,
    ):
        super().__init__()

        self.hidden_dim = embed_size
        self.num_patches = num_patches
        self.sparse_ratio = sparse_ratio
        self.aggr_ratio = aggr_ratio
        self.use_paper_version = use_paper_version
        self.top_k = top_k

        self.keeped_patches = int(num_patches * aggr_ratio * sparse_ratio)

        # sparse text 分支
        self.sparse_net_cap = TokenSparse(
            embed_dim=self.hidden_dim,
            sparse_ratio=self.sparse_ratio,
            use_paper_version=use_paper_version
        )

        # dense text 分支
        self.sparse_net_long = TokenSparse(
            embed_dim=self.hidden_dim,
            sparse_ratio=self.sparse_ratio,
            use_paper_version=use_paper_version
        )

        # 聚合网络
        self.aggr_net = TokenAggregation(
            dim=self.hidden_dim,
            keeped_patches=self.keeped_patches
        )

        # 论文版本：HRPA的Relevance Learning Network
        if use_paper_version:
            self.relevance_mlp = nn.Sequential(
                nn.Linear(top_k, top_k * 2),
                nn.GELU(),
                nn.Linear(top_k * 2, 1)
            )
        else:
            self.relevance_mlp = None

    def forward(
        self,
        img_embs: torch.Tensor,
        cap_embs: torch.Tensor,
        cap_lens: torch.Tensor,
        long_cap_embs: Optional[torch.Tensor] = None,
        long_cap_lens: Optional[torch.Tensor] = None
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        计算图像-文本相似度矩阵

        Args:
            img_embs: (B_v, N+1, C) 图像patch特征 (含cls token)
            cap_embs: (B_t, L_s, C) 稀疏文本特征
            cap_lens: (B_t,) 稀疏文本长度
            long_cap_embs: (B_t, L_d, C) 稠密文本特征
            long_cap_lens: (B_t,) 稠密文本长度

        Returns:
            训练时: (similarity_matrix, score_mask_all)
            推理时: similarity_matrix
        """
        B_v, L_v, C = img_embs.shape

        # 特征归一化
        img_embs_norm = F.normalize(img_embs, dim=-1)
        cap_embs_norm = F.normalize(cap_embs, dim=-1)
        long_cap_embs_norm = F.normalize(long_cap_embs, dim=-1) if long_cap_embs is not None else None

        # 判断是否有cls token
        self.has_cls_token = not is_sqr(img_embs.shape[1])

        # 分离cls token和空间patches
        if self.has_cls_token:
            img_cls_emb = img_embs[:, 0:1, :]
            img_spatial_embs = img_embs[:, 1:, :]
            img_spatial_embs_norm = img_embs_norm[:, 1:, :]
        else:
            img_cls_emb = None
            img_spatial_embs = img_embs
            img_spatial_embs_norm = img_embs_norm

        # 计算图像自注意力得分
        with torch.no_grad():
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

        # =================== Sparse Text 分支 ===================
        for i in range(len(cap_lens)):
            n_word = cap_lens[i]
            cap_i = cap_embs[i, :n_word, :]
            cap_i_expand = cap_embs_norm[i, :n_word, :].unsqueeze(0).repeat(B_v, 1, 1)

            with torch.no_grad():
                cap_i_glo = F.normalize(cap_i.mean(0, keepdim=True).unsqueeze(0), dim=-1)
                attn_cap = (cap_i_glo * img_spatial_embs_norm).sum(dim=-1)

                select_tokens_cap, extra_token_cap, score_mask_cap = self.sparse_net_cap(
                    tokens=img_spatial_embs,
                    attention_x=img_spatial_self_attention,
                    attention_y=attn_cap,
                )

            aggr_tokens = self.aggr_net(select_tokens_cap)
            keep_spatial_tokens = torch.cat([aggr_tokens, extra_token_cap], dim=1)

            if self.has_cls_token:
                select_tokens = torch.cat((img_cls_emb, keep_spatial_tokens), dim=1)
            else:
                select_tokens = keep_spatial_tokens

            select_tokens = F.normalize(select_tokens, dim=-1)

            sim_one_text = mask_xattn_one_text(
                img_embs=select_tokens,
                cap_i_expand=cap_i_expand,
                use_paper_version=self.use_paper_version,
                top_k=self.top_k,
                relevance_mlp=self.relevance_mlp
            )

            improve_sims.append(sim_one_text)
            score_mask_all.append(score_mask_cap)

        # =================== Dense Text 分支 ===================
        if long_cap_embs is not None and long_cap_lens is not None:
            for i in range(len(long_cap_lens)):
                n_word = long_cap_lens[i]
                long_cap_i = long_cap_embs[i, :n_word, :]
                long_cap_i_expand = long_cap_embs_norm[i, :n_word, :].unsqueeze(0).repeat(B_v, 1, 1)

                with torch.no_grad():
                    long_cap_i_glo = F.normalize(long_cap_i.mean(0, keepdim=True).unsqueeze(0), dim=-1)
                    long_attn_cap = (long_cap_i_glo * img_spatial_embs_norm).sum(dim=-1)

                    select_tokens_long, extra_token_long, score_mask_long = self.sparse_net_long(
                        tokens=img_spatial_embs,
                        attention_x=img_spatial_self_attention,
                        attention_y=long_attn_cap,
                    )

                aggr_tokens_long = self.aggr_net(select_tokens_long)
                keep_spatial_tokens = torch.cat([aggr_tokens_long, extra_token_long], dim=1)

                if self.has_cls_token:
                    select_tokens_long = torch.cat((img_cls_emb, keep_spatial_tokens), dim=1)
                else:
                    select_tokens_long = keep_spatial_tokens

                select_tokens_long = F.normalize(select_tokens_long, dim=-1)

                sim_one_text = mask_xattn_one_text(
                    img_embs=select_tokens_long,
                    cap_i_expand=long_cap_i_expand,
                    use_paper_version=self.use_paper_version,
                    top_k=self.top_k,
                    relevance_mlp=self.relevance_mlp
                )

                long_sims.append(sim_one_text)
                score_mask_long_all.append(score_mask_long)

        # 融合相似度
        improve_sims = torch.cat(improve_sims, dim=1)
        if long_sims:
            improve_sims = improve_sims + torch.cat(long_sims, dim=1)

        score_mask_all = torch.stack(score_mask_all, dim=0)
        if score_mask_long_all:
            score_mask_all = score_mask_all + torch.stack(score_mask_long_all, dim=0)

        if self.training:
            return improve_sims, score_mask_all
        else:
            return improve_sims


# =============================================================================
# 损失函数
# =============================================================================

class ContrastiveLoss(nn.Module):
    """对比损失 with Hard Negative Mining"""

    def __init__(self, margin: float = 0.2, max_violation: bool = False):
        super().__init__()
        self.margin = margin
        self.max_violation = max_violation
        self.mask_repeat = True

    def max_violation_on(self):
        self.max_violation = True

    def max_violation_off(self):
        self.max_violation = False

    def forward(self, scores: torch.Tensor, img_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        diagonal = scores.diag().view(scores.size(0), 1)
        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)

        cost_s = (self.margin + scores - d1).clamp(min=0)
        cost_im = (self.margin + scores - d2).clamp(min=0)

        if not self.mask_repeat:
            mask = torch.eye(scores.size(0), dtype=torch.bool, device=scores.device)
        else:
            if img_ids is not None:
                mask = (img_ids.unsqueeze(0) == img_ids.unsqueeze(1))
            else:
                mask = torch.eye(scores.size(0), dtype=torch.bool, device=scores.device)

        cost_s = cost_s.masked_fill_(mask, 0)
        cost_im = cost_im.masked_fill_(mask, 0)

        if self.max_violation:
            cost_s = cost_s.max(dim=1)[0]
            cost_im = cost_im.max(dim=0)[0]

        return cost_s.sum() + cost_im.sum()


class RatioLoss(nn.Module):
    """比例约束损失"""

    def __init__(self, target_ratio: float = 0.5):
        super().__init__()
        self.target_ratio = target_ratio

    def forward(self, score_mask: torch.Tensor) -> torch.Tensor:
        return (score_mask.float().mean() - self.target_ratio) ** 2


class SEPSLoss(nn.Module):
    """SEPS完整损失函数"""

    def __init__(
        self,
        margin: float = 0.2,
        target_ratio: float = 0.5,
        ratio_weight: float = 2.0,
        max_violation: bool = False
    ):
        super().__init__()
        self.contrastive_loss = ContrastiveLoss(margin=margin, max_violation=max_violation)
        self.ratio_loss = RatioLoss(target_ratio=target_ratio)
        self.ratio_weight = ratio_weight

    def set_max_violation(self, max_violation: bool = True):
        if max_violation:
            self.contrastive_loss.max_violation_on()
        else:
            self.contrastive_loss.max_violation_off()

    def forward(
        self,
        similarity_matrix: torch.Tensor,
        score_mask: torch.Tensor,
        img_ids: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        align_loss = self.contrastive_loss(similarity_matrix, img_ids)
        r_loss = self.ratio_loss(score_mask)
        total_loss = align_loss + self.ratio_weight * r_loss
        return total_loss, align_loss, r_loss


# =============================================================================
# 便捷别名
# =============================================================================

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
) -> CrossSparseAggrNet:
    """创建SEPS模型的便捷函数"""
    return CrossSparseAggrNet(
        embed_size=embed_size,
        num_patches=num_patches,
        sparse_ratio=sparse_ratio,
        aggr_ratio=aggr_ratio,
        use_paper_version=use_paper_version,
    )


# =============================================================================
# 测试代码
# =============================================================================

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 参数设置
    batch_size = 4
    num_patches = 196
    embed_size = 512
    sparse_text_len = 20
    dense_text_len = 100
    sparse_ratio = 0.5
    aggr_ratio = 0.4

    # 测试数据
    img_features = torch.randn(batch_size, num_patches + 1, embed_size).to(device)
    sparse_text = torch.randn(batch_size, sparse_text_len, embed_size).to(device)
    sparse_lens = torch.full((batch_size,), sparse_text_len).to(device)
    dense_text = torch.randn(batch_size, dense_text_len, embed_size).to(device)
    dense_lens = torch.full((batch_size,), dense_text_len).to(device)
    img_ids = torch.arange(batch_size).to(device)

    print("=" * 70)
    print("测试实际代码版本 (use_paper_version=False)")
    print("=" * 70)

    seps_actual = CrossSparseAggrNet(
        embed_size=embed_size,
        num_patches=num_patches,
        sparse_ratio=sparse_ratio,
        aggr_ratio=aggr_ratio,
        use_paper_version=False
    ).to(device)
    seps_actual.train()

    sim_actual, mask_actual = seps_actual(img_features, sparse_text, sparse_lens, dense_text, dense_lens)
    print(f"Similarity matrix shape: {sim_actual.shape}")
    print(f"Parameters: {sum(p.numel() for p in seps_actual.parameters()):,}")

    print("\n" + "=" * 70)
    print("测试论文描述版本 (use_paper_version=True)")
    print("=" * 70)

    seps_paper = CrossSparseAggrNet(
        embed_size=embed_size,
        num_patches=num_patches,
        sparse_ratio=sparse_ratio,
        aggr_ratio=aggr_ratio,
        use_paper_version=True
    ).to(device)
    seps_paper.train()

    sim_paper, mask_paper = seps_paper(img_features, sparse_text, sparse_lens, dense_text, dense_lens)
    print(f"Similarity matrix shape: {sim_paper.shape}")
    print(f"Parameters: {sum(p.numel() for p in seps_paper.parameters()):,}")

    print("\n" + "=" * 70)
    print("参数量对比")
    print("=" * 70)
    print(f"实际代码版本: {sum(p.numel() for p in seps_actual.parameters()):,} 参数")
    print(f"论文描述版本: {sum(p.numel() for p in seps_paper.parameters()):,} 参数")
    print(f"额外参数: {sum(p.numel() for p in seps_paper.parameters()) - sum(p.numel() for p in seps_actual.parameters()):,}")

    print("\n✓ 所有测试通过!")
