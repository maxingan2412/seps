"""
SEPS: Semantic-Enhanced Patch Slimming Framework (逐行注释版)
=============================================================================
基于 seps_modules_reviewed_v2.py，添加逐行注释
重点: 函数功能 + Tensor形状变化

论文: arXiv:2511.01390, ICLR 2026
"""

import math
from typing import Iterable, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

# 全局配置: 默认使用开源代码版本
USE_PAPER_VERSION_DEFAULT = False


# =============================================================================
# 辅助函数
# =============================================================================

def l2_normalize(x: torch.Tensor, dim: int = -1, eps: float = 1e-8) -> torch.Tensor:
    """
    L2归一化: 将向量归一化为单位向量

    输入: x - 任意形状的Tensor
    输出: 归一化后的Tensor (形状不变)
    作用: 使得 ||x||_2 = 1，用于计算余弦相似度
    """
    return F.normalize(x, p=2, dim=dim, eps=eps)


def is_sqr(n: int) -> bool:
    """
    判断n是否为完全平方数

    输入: n - 整数 (通常是patch数量)
    输出: bool - True表示是完全平方数
    作用: 判断ViT输出是否包含[CLS] token
    """
    a = int(math.sqrt(n))  # 计算平方根并取整
    return a * a == n      # 检查平方是否等于n


# =============================================================================
# TokenSparse: SDTPS Stage 1 (语义评分与Patch选择)
# 论文公式(1)-(3)
# =============================================================================

class TokenSparse(nn.Module):
    """
    Token稀疏选择模块

    功能: 从N个patch中选择K个显著patch (K = N * sparse_ratio)
    方法: 综合评分 = MLP预测 + 图像自注意力 + 文本交叉注意力
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

        # 论文公式(1): MLP预测器 (仅论文版本)
        if use_paper_version:
            # 输入: (*, C) → 输出: (*, 1)
            self.score_predictor = nn.Sequential(
                nn.Linear(embed_dim, embed_dim // 4),  # (*, C) → (*, C//4)
                nn.GELU(),                              # 激活函数
                nn.Linear(embed_dim // 4, 1),          # (*, C//4) → (*, 1)
                nn.Sigmoid(),                           # 输出范围[0,1]
            )

    def forward(
        self,
        tokens: torch.Tensor,                           # (B, N, C) - patch特征
        attention_x: torch.Tensor,                      # (B, N) - 图像自注意力
        attention_y: torch.Tensor,                      # (B, N) - 稀疏文本注意力
        attention_y_dense: Optional[torch.Tensor] = None,  # (B, N) - 稠密文本注意力
        beta: float = 0.25,
        use_gumbel: bool = False,
        gumbel_tau: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        前向传播

        输入:
            tokens: (B, N, C) - patch特征
            attention_x: (B, N) - 图像自注意力得分
            attention_y: (B, N) - 稀疏文本注意力得分
            attention_y_dense: (B, N) or None - 稠密文本注意力得分

        输出:
            select_tokens: (B, N_keep, C) - 选中的patch
            extra_token: (B, 1, C) - 融合的丢弃patch
            score_mask: (B, N) - 决策矩阵 (1=选中, 0=丢弃)
        """
        B_v, L_v, C = tokens.size()  # 获取形状: batch, patch数, 特征维度

        # =========================================================
        # 计算综合得分 score
        # =========================================================
        if self.use_paper_version:
            # 论文版本: 公式(1)-(3)

            # 公式(1): MLP预测得分
            s_pred = self.score_predictor(tokens)  # (B, N, C) → (B, N, 1)
            s_pred = s_pred.squeeze(-1)            # (B, N, 1) → (B, N)

            # Min-Max归一化函数
            def normalize_score(s: torch.Tensor) -> torch.Tensor:
                # 输入: (B, N) → 输出: (B, N), 范围[0,1]
                s_min = s.min(dim=-1, keepdim=True)[0]  # (B, 1)
                s_max = s.max(dim=-1, keepdim=True)[0]  # (B, 1)
                return (s - s_min) / (s_max - s_min + 1e-8)  # (B, N)

            # 公式(2): 归一化各注意力得分
            s_im = normalize_score(attention_x)     # (B, N) - 图像自注意力
            s_st = normalize_score(attention_y)     # (B, N) - 稀疏文本
            s_dt = (normalize_score(attention_y_dense) if attention_y_dense is not None
                    else torch.zeros_like(s_st))    # (B, N) - 稠密文本

            # 公式(3): 综合得分
            # score = (1-2β)·s_pred + β·(s_st + s_dt + 2·s_im)
            score = (1 - 2 * beta) * s_pred + beta * (s_st + s_dt + 2 * s_im)  # (B, N)
        else:
            # 开源代码版本: 简单相加
            score = attention_x + attention_y  # (B, N)

        # =========================================================
        # Top-K选择
        # =========================================================
        num_keep_token = max(1, math.ceil(L_v * self.sparse_ratio))  # K = ceil(N*ratio)

        score_sort, score_index = torch.sort(score, dim=1, descending=True)  # 降序排序
        # score_sort: (B, N) - 排序后的得分
        # score_index: (B, N) - 原始索引

        keep_policy = score_index[:, :num_keep_token]  # (B, K) - 保留的索引

        # =========================================================
        # 生成决策矩阵
        # =========================================================
        if use_gumbel:
            # Gumbel-Softmax可微采样
            gumbel_noise = -torch.log(-torch.log(torch.rand_like(score) + 1e-9) + 1e-9)  # (B, N)
            soft_mask = F.softmax((score + gumbel_noise) / gumbel_tau, dim=1)  # (B, N) - 软决策
            hard_mask = torch.zeros_like(score).scatter(1, keep_policy, 1.0)   # (B, N) - 硬决策
            score_mask = hard_mask + (soft_mask - soft_mask.detach())  # Straight-Through
        else:
            # 标准Top-K (不可微)
            score_mask = torch.zeros_like(score).scatter(1, keep_policy, 1.0)  # (B, N)

        # =========================================================
        # 提取选中的patch
        # =========================================================
        select_tokens = torch.gather(
            tokens, dim=1,
            index=keep_policy.unsqueeze(-1).expand(-1, -1, C)  # (B, K) → (B, K, C)
        )  # (B, K, C) - 选中的patch特征

        # =========================================================
        # 融合被丢弃的patch
        # =========================================================
        non_keep_policy = score_index[:, num_keep_token:]  # (B, N-K) - 丢弃的索引
        non_tokens = torch.gather(
            tokens, dim=1,
            index=non_keep_policy.unsqueeze(-1).expand(-1, -1, C)
        )  # (B, N-K, C) - 丢弃的patch特征

        non_keep_score = score_sort[:, num_keep_token:]  # (B, N-K) - 丢弃patch的得分
        non_keep_score = F.softmax(non_keep_score, dim=1).unsqueeze(-1)  # (B, N-K, 1) - 归一化权重
        extra_token = torch.sum(non_tokens * non_keep_score, dim=1, keepdim=True)  # (B, 1, C) - 加权融合

        return select_tokens, extra_token, score_mask


# =============================================================================
# TokenAggregation: SDTPS Stage 2a (单分支聚合)
# 论文公式(4)简化版
# =============================================================================

class TokenAggregation(nn.Module):
    """
    单分支聚合模块

    功能: 将N个patch聚合为N_c个patch (N_c < N)
    方法: 学习权重矩阵W, v̂_j = Σ_i W_ij * v_i
    """

    def __init__(
        self,
        dim: int = 512,
        keeped_patches: int = 64,
        dim_ratio: float = 0.2,
    ):
        super().__init__()
        hidden_dim = int(dim * dim_ratio)  # 隐藏层维度

        # 权重生成网络: (*, C) → (*, N_c)
        self.weight = nn.Sequential(
            nn.LayerNorm(dim),                      # 归一化
            nn.Linear(dim, hidden_dim),             # 降维
            nn.GELU(),                               # 激活
            nn.Linear(hidden_dim, keeped_patches),  # 输出权重
        )
        self.scale = nn.Parameter(torch.ones(1, 1, 1))  # 可学习缩放因子

    def forward(
        self,
        x: torch.Tensor,                              # (B, N, C) - 输入patch
        keep_policy: Optional[torch.Tensor] = None,   # (B, N) - 可选mask
    ) -> torch.Tensor:
        """
        聚合patches

        输入: x - (B, N, C)
        输出: (B, N_c, C)
        """
        weight = self.weight(x)                 # (B, N, C) → (B, N, N_c)
        weight = weight.transpose(2, 1)         # (B, N, N_c) → (B, N_c, N)
        weight = weight * self.scale            # 缩放

        if keep_policy is not None:
            # 屏蔽无效位置
            keep_policy = keep_policy.unsqueeze(1)  # (B, N) → (B, 1, N)
            weight = weight - (1 - keep_policy) * 1e10  # 无效位置设为极小值

        weight = F.softmax(weight, dim=2)       # (B, N_c, N) - 归一化权重
        return torch.bmm(weight, x)             # (B, N_c, N) @ (B, N, C) → (B, N_c, C)


# =============================================================================
# DualTokenAggregation: SDTPS Stage 2b (双分支聚合)
# 论文公式(4)完整版
# =============================================================================

class DualTokenAggregation(nn.Module):
    """
    双分支聚合模块

    功能: 联合聚合稀疏文本和稠密文本选择的patch
    公式: v̂_j = Σ_i (W_s)_ij * v_i^s + Σ_i (W_d)_ij * v_i^d
    """

    def __init__(
        self,
        dim: int = 512,
        keeped_patches: int = 64,
        dim_ratio: float = 0.2,
    ):
        super().__init__()
        hidden_dim = int(dim * dim_ratio)

        # 稀疏文本分支权重网络
        self.weight_sparse = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, keeped_patches),
        )

        # 稠密文本分支权重网络
        self.weight_dense = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, keeped_patches),
        )

        self.scale = nn.Parameter(torch.ones(1, 1, 1))

    def _aggregate(
        self,
        x: torch.Tensor,               # (B, N, C)
        mask: Optional[torch.Tensor],  # (B, N)
        weight_net: nn.Module,
    ) -> torch.Tensor:
        """
        单分支聚合辅助函数

        输入: x - (B, N, C)
        输出: (B, N_c, C)
        """
        weight = weight_net(x).transpose(2, 1) * self.scale  # (B, N_c, N)

        if mask is not None:
            weight = weight - (1 - mask.unsqueeze(1)) * 1e10  # 应用mask

        weight = F.softmax(weight, dim=2)  # 归一化
        return torch.bmm(weight, x)        # (B, N_c, C)

    def forward(
        self,
        tokens_sparse: torch.Tensor,                    # (B, N_s, C) - 稀疏文本选择的patch
        tokens_dense: Optional[torch.Tensor] = None,    # (B, N_d, C) - 稠密文本选择的patch
        mask_sparse: Optional[torch.Tensor] = None,
        mask_dense: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        双分支联合聚合

        输入:
            tokens_sparse: (B, N_s, C)
            tokens_dense: (B, N_d, C) or None
        输出: (B, N_c, C)
        """
        # 稀疏分支聚合
        out = self._aggregate(tokens_sparse, mask_sparse, self.weight_sparse)  # (B, N_c, C)

        # 稠密分支聚合 (如果有)
        if tokens_dense is not None:
            out = out + self._aggregate(tokens_dense, mask_dense, self.weight_dense)  # (B, N_c, C)

        return out  # (B, N_c, C)


# =============================================================================
# HRPA: Highly-Relevant Patch-Word Alignment
# 论文公式(5)
# =============================================================================

def mask_xattn_one_text(
    img_embs: torch.Tensor,                           # (B_v, N_c, C) - patch特征
    cap_i_expand: torch.Tensor,                       # (B_v, M, C) - word特征
    img_mask: Optional[torch.Tensor] = None,
    i2t: bool = True,
    scan: bool = True,
    use_paper_version: bool = USE_PAPER_VERSION_DEFAULT,
    top_k: int = 5,
    relevance_mlp: Optional[nn.Module] = None,
) -> torch.Tensor:
    """
    HRPA相似度计算函数

    功能: 计算图像-文本相似度 S(I, T)
    方法: 双向对齐 (patch-to-word + word-to-patch)

    输入:
        img_embs: (B_v, N_c, C) - 已归一化的patch特征
        cap_i_expand: (B_v, M, C) - 已归一化的word特征
    输出: (B_v, 1) - 相似度得分
    """
    # 计算相似度矩阵 A = patch @ word^T
    cap2img_sim = torch.bmm(cap_i_expand, img_embs.transpose(1, 2))  # (B_v, M, N_c)

    if scan:
        cap2img_sim = F.leaky_relu(cap2img_sim, negative_slope=0.1)  # SCAN技巧

    # =========================================================
    # word-to-patch对齐: 对每个word找最相关的patch
    # =========================================================
    if img_mask is None:
        row_sim = cap2img_sim.max(dim=2)[0]  # (B_v, M) - 每个word的最大相似度
    else:
        row_sim = (cap2img_sim - 1000 * (1 - img_mask).unsqueeze(1)).max(dim=2)[0]

    row_sim_mean = row_sim.mean(dim=1, keepdim=True)  # (B_v, 1) - 平均

    # 论文版本: 添加TopK+MLP
    if use_paper_version and relevance_mlp is not None:
        B_v, M = row_sim.shape
        k = min(top_k, M)
        row_topk, _ = row_sim.topk(k, dim=1)  # (B_v, k) - TopK相似度

        if k < top_k:
            padding = torch.zeros(B_v, top_k - k, device=row_topk.device)
            row_topk = torch.cat([row_topk, padding], dim=1)  # (B_v, top_k)

        row_sim_mean = row_sim_mean + relevance_mlp(row_topk)  # (B_v, 1)

    # =========================================================
    # patch-to-word对齐: 对每个patch找最相关的word
    # =========================================================
    if i2t:
        column_sim = cap2img_sim.max(dim=1)[0]  # (B_v, N_c) - 每个patch的最大相似度

        if img_mask is None:
            column_sim_mean = column_sim.mean(dim=1, keepdim=True)  # (B_v, 1)
        else:
            column_sim_mean = (column_sim * img_mask).sum(dim=-1, keepdim=True) / (
                img_mask.sum(dim=-1, keepdim=True) + 1e-8
            )

        # 论文版本: 添加TopK+MLP
        if use_paper_version and relevance_mlp is not None:
            B_v, N = column_sim.shape
            k = min(top_k, N)
            col_topk, _ = column_sim.topk(k, dim=1)  # (B_v, k)

            if k < top_k:
                padding = torch.zeros(B_v, top_k - k, device=col_topk.device)
                col_topk = torch.cat([col_topk, padding], dim=1)

            column_sim_mean = column_sim_mean + relevance_mlp(col_topk)

        # 双向对齐相加
        sim_one_text = row_sim_mean + column_sim_mean  # (B_v, 1)
    else:
        sim_one_text = row_sim_mean

    return sim_one_text  # (B_v, 1)


class HRPA(nn.Module):
    """
    HRPA模块 (类封装版本)

    功能: 计算高相关性的patch-word对齐相似度
    """

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

        # 论文版本: 相关性学习网络
        if use_paper_version:
            self.relevance_mlp = nn.Sequential(
                nn.Linear(top_k, top_k * 2),  # (*, top_k) → (*, top_k*2)
                nn.GELU(),
                nn.Linear(top_k * 2, 1),      # (*, top_k*2) → (*, 1)
            )
        else:
            self.relevance_mlp = None

    def forward(
        self,
        patch_features: torch.Tensor,                  # (B_v, N_c, C)
        word_features: torch.Tensor,                   # (B_v, M, C)
        patch_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        计算相似度

        输入:
            patch_features: (B_v, N_c, C)
            word_features: (B_v, M, C)
        输出: (B_v, 1)
        """
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


# =============================================================================
# CrossSparseAggrNet: 完整的SDTPS + HRPA流程
# =============================================================================

class CrossSparseAggrNet(nn.Module):
    """
    完整的SEPS模型

    功能: SDTPS (选择+聚合) + HRPA (对齐)
    流程:
        1. 特征归一化
        2. TokenSparse: 选择显著patch
        3. TokenAggregation: 聚合patch
        4. HRPA: 计算相似度
    """

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

        # 聚合后的patch数量
        self.keeped_patches = int(num_patches * aggr_ratio * sparse_ratio)

        # SDTPS Stage 1: TokenSparse (两个分支)
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

        # SDTPS Stage 2: Aggregation
        if use_paper_version and use_dual_aggr:
            # 论文版本: 双分支聚合
            self.aggr_net = DualTokenAggregation(
                dim=self.hidden_dim,
                keeped_patches=self.keeped_patches,
            )
        else:
            # 开源代码版本: 单分支聚合
            self.aggr_net = TokenAggregation(
                dim=self.hidden_dim,
                keeped_patches=self.keeped_patches,
            )

        # HRPA模块
        self.hrpa = HRPA(
            embed_dim=self.hidden_dim,
            top_k=self.top_k,
            use_paper_version=self.use_paper_version,
            bidirectional=True,
            scan=not self.use_paper_version,  # 论文用纯余弦, 代码用LeakyReLU
        )

    def forward(
        self,
        img_embs: torch.Tensor,                         # (B_v, N+1, C) - 图像特征
        cap_embs: torch.Tensor,                         # (B_t, L_s, C) - 稀疏文本
        cap_lens: torch.Tensor,                         # (B_t,) - 稀疏文本长度
        long_cap_embs: Optional[torch.Tensor] = None,   # (B_t, L_d, C) - 稠密文本
        long_cap_lens: Optional[torch.Tensor] = None,   # (B_t,) - 稠密文本长度
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Union[torch.Tensor, Tuple[torch.Tensor, ...]]]]:
        """
        前向传播

        输入:
            img_embs: (B_v, N+1, C) - 图像patch特征 (可能含[CLS])
            cap_embs: (B_t, L_s, C) - 稀疏文本特征
            cap_lens: (B_t,) - 稀疏文本实际长度
            long_cap_embs: (B_t, L_d, C) or None - 稠密文本特征
            long_cap_lens: (B_t,) or None - 稠密文本长度

        输出:
            训练模式: (similarity_matrix, score_mask)
                similarity_matrix: (B_v, B_t) - 相似度矩阵
                score_mask: 决策矩阵 (用于L_ratio计算)
            推理模式: similarity_matrix
        """
        B_v, L_v, C = img_embs.shape  # batch, patch数, 特征维度

        # =========================================================
        # Step 1: L2归一化
        # =========================================================
        img_embs_norm = F.normalize(img_embs, dim=-1)      # (B_v, L_v, C)
        cap_embs_norm = F.normalize(cap_embs, dim=-1)      # (B_t, L_s, C)
        long_cap_embs_norm = (
            F.normalize(long_cap_embs, dim=-1)              # (B_t, L_d, C)
            if long_cap_embs is not None
            else None
        )

        # =========================================================
        # Step 2: 分离[CLS] token
        # =========================================================
        self.has_cls_token = not is_sqr(img_embs.shape[1])  # 判断是否有[CLS]

        if self.has_cls_token:
            img_cls_emb = img_embs[:, 0:1, :]           # (B_v, 1, C) - [CLS] token
            img_spatial_embs = img_embs[:, 1:, :]       # (B_v, N, C) - 空间patch
            img_spatial_embs_norm = img_embs_norm[:, 1:, :]
        else:
            img_cls_emb = None
            img_spatial_embs = img_embs                 # (B_v, N, C)
            img_spatial_embs_norm = img_embs_norm

        # =========================================================
        # Step 3: 计算图像自注意力 s^{im}
        # =========================================================
        with torch.no_grad():
            # 图像全局表示 = 所有patch的平均
            img_spatial_glo_norm = F.normalize(
                img_spatial_embs.mean(dim=1, keepdim=True), dim=-1
            )  # (B_v, 1, C)

            # 每个patch与全局表示的相似度
            img_spatial_self_attention = (
                img_spatial_glo_norm * img_spatial_embs_norm
            ).sum(dim=-1)  # (B_v, N)

        # 结果收集
        improve_sims = []          # 相似度列表
        long_sims = []             # 稠密文本相似度 (开源代码版本)
        score_mask_all = []        # 决策矩阵列表
        score_mask_long_all = []

        # =========================================================
        # Step 4: 对每个文本进行处理
        # =========================================================
        for i in range(len(cap_lens)):
            # 获取第i个文本
            n_word = int(cap_lens[i])                       # 稀疏文本长度
            cap_i = cap_embs[i, :n_word, :]                # (M, C)
            cap_i_expand = cap_embs_norm[i, :n_word, :].unsqueeze(0).repeat(B_v, 1, 1)  # (B_v, M, C)

            dense_attn = None
            long_cap_i_expand = None
            select_tokens_long = extra_token_long = score_mask_long = None

            # ---------------------------------------------------------
            # Step 4a: 计算文本-图像交叉注意力
            # ---------------------------------------------------------
            with torch.no_grad():
                # 稀疏文本全局表示
                cap_i_glo = F.normalize(
                    cap_i.mean(0, keepdim=True).unsqueeze(0), dim=-1
                )  # (1, 1, C)

                # 稀疏文本交叉注意力 s^{st}
                attn_cap = (cap_i_glo * img_spatial_embs_norm).sum(dim=-1)  # (B_v, N)

                # 稠密文本交叉注意力 s^{dt}
                if long_cap_embs_norm is not None and long_cap_lens is not None:
                    n_word_long = int(long_cap_lens[i])
                    long_cap_i = long_cap_embs[i, :n_word_long, :]
                    long_cap_i_expand = (
                        long_cap_embs_norm[i, :n_word_long, :]
                        .unsqueeze(0).repeat(B_v, 1, 1)
                    )  # (B_v, M_d, C)

                    long_cap_i_glo = F.normalize(
                        long_cap_i.mean(0, keepdim=True).unsqueeze(0), dim=-1
                    )
                    dense_attn = (long_cap_i_glo * img_spatial_embs_norm).sum(dim=-1)  # (B_v, N)

            # ---------------------------------------------------------
            # Step 4b: SDTPS Stage 1 - TokenSparse
            # ---------------------------------------------------------
            # 稀疏文本分支
            select_tokens_cap, extra_token_cap, score_mask_cap = self.sparse_net_cap(
                tokens=img_spatial_embs,                    # (B_v, N, C)
                attention_x=img_spatial_self_attention,     # (B_v, N)
                attention_y=attn_cap,                        # (B_v, N)
                attention_y_dense=dense_attn if self.use_paper_version else None,
                beta=self.beta,
                use_gumbel=self.use_gumbel_softmax,
                gumbel_tau=self.gumbel_tau,
            )
            # select_tokens_cap: (B_v, N_keep, C)
            # extra_token_cap: (B_v, 1, C)
            # score_mask_cap: (B_v, N)

            # 稠密文本分支
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

            # ---------------------------------------------------------
            # Step 4c: SDTPS Stage 2 - Aggregation
            # ---------------------------------------------------------
            if self.use_paper_version:
                # 论文版本: 双分支联合聚合
                if (self.use_dual_aggr and select_tokens_long is not None
                    and extra_token_long is not None):
                    # 双分支聚合: v̂ = W_s·v^s + W_d·v^d
                    aggr_tokens = self.aggr_net(
                        select_tokens_cap,      # (B_v, N_keep, C)
                        select_tokens_long,     # (B_v, N_keep, C)
                        None, None,             # mask不需要 (tokens已筛选)
                    )  # (B_v, N_c, C)

                    # 融合extra token
                    extra_token = torch.stack(
                        [extra_token_cap, extra_token_long], dim=0
                    ).mean(dim=0)  # (B_v, 1, C)

                    mask_pack: Union[torch.Tensor, Tuple[torch.Tensor, ...]] = (
                        score_mask_cap, score_mask_long,
                    )
                else:
                    aggr_tokens = self.aggr_net(select_tokens_cap)  # (B_v, N_c, C)
                    extra_token = extra_token_cap
                    mask_pack = (score_mask_cap,)

                # 拼接聚合tokens和extra token
                keep_spatial_tokens = torch.cat([aggr_tokens, extra_token], dim=1)  # (B_v, N_c+1, C)

                # 添加[CLS] token
                if self.has_cls_token:
                    select_tokens = torch.cat((img_cls_emb, keep_spatial_tokens), dim=1)  # (B_v, N_c+2, C)
                else:
                    select_tokens = keep_spatial_tokens

                select_tokens = F.normalize(select_tokens, dim=-1)  # 归一化

                # ---------------------------------------------------------
                # Step 4d: HRPA - 计算相似度
                # ---------------------------------------------------------
                sim_one_text = self.hrpa(
                    patch_features=select_tokens,   # (B_v, N_c+1, C) or (B_v, N_c+2, C)
                    word_features=cap_i_expand,     # (B_v, M, C) - 只用稀疏文本
                )  # (B_v, 1)

                improve_sims.append(sim_one_text)
                score_mask_all.append(mask_pack)

            else:
                # 开源代码版本: 两分支分别处理
                aggr_tokens = self.aggr_net(select_tokens_cap)  # (B_v, N_c, C)
                keep_spatial_tokens = torch.cat([aggr_tokens, extra_token_cap], dim=1)

                if self.has_cls_token:
                    select_tokens = torch.cat((img_cls_emb, keep_spatial_tokens), dim=1)
                else:
                    select_tokens = keep_spatial_tokens

                select_tokens = F.normalize(select_tokens, dim=-1)

                # 稀疏文本相似度
                sim_one_text = self.hrpa(
                    patch_features=select_tokens,
                    word_features=cap_i_expand,
                )  # (B_v, 1)
                improve_sims.append(sim_one_text)
                score_mask_all.append(score_mask_cap)

                # 稠密文本相似度 (开源代码也计算)
                if select_tokens_long is not None and long_cap_i_expand is not None:
                    aggr_tokens_long = self.aggr_net(select_tokens_long)
                    keep_spatial_tokens = torch.cat([aggr_tokens_long, extra_token_long], dim=1)

                    if self.has_cls_token:
                        select_tokens_long_final = torch.cat((img_cls_emb, keep_spatial_tokens), dim=1)
                    else:
                        select_tokens_long_final = keep_spatial_tokens

                    select_tokens_long_final = F.normalize(select_tokens_long_final, dim=-1)

                    sim_one_text = self.hrpa(
                        patch_features=select_tokens_long_final,
                        word_features=long_cap_i_expand,  # 用稠密文本
                    )
                    long_sims.append(sim_one_text)
                    score_mask_long_all.append(score_mask_long)

        # =========================================================
        # Step 5: 整合结果
        # =========================================================
        improve_sims = torch.cat(improve_sims, dim=1)  # (B_v, B_t)

        # 开源代码版本: 相似度相加
        if not self.use_paper_version and long_sims:
            improve_sims = improve_sims + torch.cat(long_sims, dim=1)

        # 整理决策矩阵
        if self.use_paper_version:
            sparse_masks_list = [m[0] if isinstance(m, tuple) else m for m in score_mask_all]
            dense_masks_list = [m[1] for m in score_mask_all if isinstance(m, tuple) and len(m) > 1]

            sparse_stack = torch.stack(sparse_masks_list, dim=0)  # (B_t, B_v, N)
            dense_stack = (
                torch.stack(dense_masks_list, dim=0) if len(dense_masks_list) > 0 else None
            )

            score_mask_out: Union[torch.Tensor, Tuple[torch.Tensor, ...]]
            if dense_stack is not None:
                score_mask_out = (sparse_stack, dense_stack)
            else:
                score_mask_out = (sparse_stack,)
        else:
            score_mask_out = torch.stack(score_mask_all, dim=0)  # (B_t, B_v, N)
            if score_mask_long_all:
                score_mask_out = score_mask_out + torch.stack(score_mask_long_all, dim=0)

        # 返回结果
        if self.training:
            return improve_sims, score_mask_out  # 训练模式: 返回相似度和mask
        return improve_sims  # 推理模式: 只返回相似度


# =============================================================================
# 损失函数
# =============================================================================

class ContrastiveLoss(nn.Module):
    """
    对比损失 (Triplet Loss with Hard Negative Mining)

    功能: 拉近正样本对，推开负样本对
    公式: L = Σ [α - S(I,T) + S(I,T̂)]_+ + [α - S(I,T) + S(Î,T)]_+
    """

    def __init__(self, margin: float = 0.2, max_violation: bool = False):
        super().__init__()
        self.margin = margin            # α: margin
        self.max_violation = max_violation  # 是否使用hard negative mining
        self.mask_repeat = True         # 处理一图多文

    def max_violation_on(self):
        """开启hard negative mining"""
        self.max_violation = True

    def max_violation_off(self):
        """关闭hard negative mining"""
        self.max_violation = False

    def forward(
        self,
        scores: torch.Tensor,                      # (B, B) - 相似度矩阵
        img_ids: Optional[torch.Tensor] = None,    # (B,) - 图像ID
    ) -> torch.Tensor:
        """
        计算对比损失

        输入: scores - (B, B), scores[i,j] = S(I_i, T_j)
        输出: 标量损失
        """
        # 对角线是正样本对的相似度
        diagonal = scores.diag().view(scores.size(0), 1)  # (B, 1)
        d1 = diagonal.expand_as(scores)      # (B, B) - 按行扩展
        d2 = diagonal.t().expand_as(scores)  # (B, B) - 按列扩展

        # 文本检索损失: [α - S(I,T) + S(I,T̂)]_+
        cost_s = (self.margin + scores - d1).clamp(min=0)  # (B, B)

        # 图像检索损失: [α - S(I,T) + S(Î,T)]_+
        cost_im = (self.margin + scores - d2).clamp(min=0)  # (B, B)

        # 构建mask屏蔽正样本
        if not self.mask_repeat:
            mask = torch.eye(scores.size(0), dtype=torch.bool, device=scores.device)
        else:
            if img_ids is not None:
                # 同一图像的多个caption都视为正样本
                mask = img_ids.unsqueeze(0) == img_ids.unsqueeze(1)  # (B, B)
            else:
                mask = torch.eye(scores.size(0), dtype=torch.bool, device=scores.device)

        # 正样本位置损失设为0
        cost_s = cost_s.masked_fill_(mask, 0)    # (B, B)
        cost_im = cost_im.masked_fill_(mask, 0)

        # Hard Negative Mining
        if self.max_violation:
            # 只使用最难负样本 (得分最高的负样本)
            cost_s = cost_s.max(dim=1)[0]    # (B,)
            cost_im = cost_im.max(dim=0)[0]  # (B,)

        return cost_s.sum() + cost_im.sum()  # 标量


class RatioLoss(nn.Module):
    """
    比例约束损失

    功能: 约束选择的patch数量接近目标比例
    公式: L = (ρ - λ_1·mean(D_s) - λ_2·mean(D_d))²
    """

    def __init__(
        self,
        target_ratio: float = 0.5,     # ρ: 目标比例
        lambda_sparse: float = 1.0,    # λ_1: 稀疏权重
        lambda_dense: float = 1.0,     # λ_2: 稠密权重
    ):
        super().__init__()
        self.target_ratio = target_ratio
        self.lambda_sparse = lambda_sparse
        self.lambda_dense = lambda_dense

    def forward(
        self,
        score_mask: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
    ) -> torch.Tensor:
        """
        计算比例损失

        输入: score_mask - 决策矩阵
        输出: 标量损失
        """
        if isinstance(score_mask, tuple) or isinstance(score_mask, list):
            # 论文版本: 分别约束稀疏和稠密
            sparse_mask = score_mask[0]  # D_s
            dense_mask = score_mask[1] if len(score_mask) > 1 else None  # D_d

            # (ρ - mean(D_s))²
            sparse_loss = (sparse_mask.float().mean() - self.target_ratio) ** 2

            # (ρ - mean(D_d))²
            dense_loss = (
                (dense_mask.float().mean() - self.target_ratio) ** 2
                if dense_mask is not None
                else torch.tensor(0.0, device=sparse_mask.device)
            )

            # λ_1·L_s + λ_2·L_d
            return self.lambda_sparse * sparse_loss + self.lambda_dense * dense_loss

        # 简化版本
        return (score_mask.float().mean() - self.target_ratio) ** 2


class SEPSLoss(nn.Module):
    """
    SEPS完整损失函数

    功能: L = L_align + ratio_weight * L_ratio
    """

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
        self.contrastive_loss = ContrastiveLoss(margin=margin, max_violation=max_violation)
        self.ratio_loss = RatioLoss(
            target_ratio=target_ratio,
            lambda_sparse=lambda_sparse,
            lambda_dense=lambda_dense,
        )
        self.ratio_weight = ratio_weight

    def set_max_violation(self, max_violation: bool = True):
        """设置hard negative mining"""
        if max_violation:
            self.contrastive_loss.max_violation_on()
        else:
            self.contrastive_loss.max_violation_off()

    def forward(
        self,
        similarity_matrix: torch.Tensor,                           # (B_v, B_t)
        score_mask: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
        img_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        计算总损失

        输入:
            similarity_matrix: (B_v, B_t) - 相似度矩阵
            score_mask: 决策矩阵
            img_ids: (B,) - 图像ID

        输出:
            total_loss: 总损失
            align_loss: 对齐损失 (用于监控)
            ratio_loss: 比例损失 (用于监控)
        """
        align_loss = self.contrastive_loss(similarity_matrix, img_ids)  # L_align
        r_loss = self.ratio_loss(score_mask)                            # L_ratio
        total_loss = align_loss + self.ratio_weight * r_loss            # L_total

        return total_loss, align_loss, r_loss


# =============================================================================
# 便捷别名和工厂函数
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
    use_gumbel_softmax: bool = False,
    gumbel_tau: float = 1.0,
) -> CrossSparseAggrNet:
    """
    创建SEPS模型的工厂函数

    参数:
        embed_size: 特征维度
        num_patches: patch数量
        sparse_ratio: 选择比例
        aggr_ratio: 聚合比例
        use_paper_version: 是否使用论文版本
        use_gumbel_softmax: 是否使用Gumbel-Softmax
        gumbel_tau: Gumbel温度

    返回: CrossSparseAggrNet实例
    """
    return CrossSparseAggrNet(
        embed_size=embed_size,
        num_patches=num_patches,
        sparse_ratio=sparse_ratio,
        aggr_ratio=aggr_ratio,
        use_paper_version=use_paper_version,
        use_gumbel_softmax=use_gumbel_softmax,
        gumbel_tau=gumbel_tau,
    )


# =============================================================================
# 测试代码
# =============================================================================

if __name__ == "__main__":
    """
    简单的形状检查测试
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 测试参数
    B = 2                  # batch size
    num_patches = 16       # patch数量
    embed_size = 32        # 特征维度
    sparse_len = 5         # 稀疏文本长度
    dense_len = 8          # 稠密文本长度

    # 构造测试数据
    img = torch.randn(B, num_patches + 1, embed_size, device=device)  # (B, N+1, C)
    cap = torch.randn(B, sparse_len, embed_size, device=device)       # (B, M_s, C)
    cap_len = torch.full((B,), sparse_len, device=device)             # (B,)
    long_cap = torch.randn(B, dense_len, embed_size, device=device)   # (B, M_d, C)
    long_len = torch.full((B,), dense_len, device=device)             # (B,)

    print("=" * 70)
    print("测试开源代码版本 (use_paper_version=False)")
    print("=" * 70)

    # 创建开源代码版本模型
    seps_code = create_seps_model(
        embed_size=embed_size,
        num_patches=num_patches,
        sparse_ratio=0.5,
        aggr_ratio=0.5,
        use_paper_version=False,
    ).to(device)
    seps_code.train()

    # 前向传播
    sims, mask = seps_code(img, cap, cap_len, long_cap, long_len)
    print(f"Similarity shape: {sims.shape}")  # (B, B)
    print(f"Mask shape: {mask[0].shape if isinstance(mask, tuple) else mask.shape}")
    print(f"Parameters: {sum(p.numel() for p in seps_code.parameters()):,}")

    print("\n" + "=" * 70)
    print("测试论文版本 (use_paper_version=True)")
    print("=" * 70)

    # 创建论文版本模型
    seps_paper = create_seps_model(
        embed_size=embed_size,
        num_patches=num_patches,
        sparse_ratio=0.5,
        aggr_ratio=0.5,
        use_paper_version=True,
    ).to(device)
    seps_paper.train()

    # 前向传播
    sims, mask = seps_paper(img, cap, cap_len, long_cap, long_len)
    print(f"Similarity shape: {sims.shape}")
    print(f"Mask shape: {tuple(m.shape for m in mask) if isinstance(mask, tuple) else mask.shape}")
    print(f"Parameters: {sum(p.numel() for p in seps_paper.parameters()):,}")

    print("\n✓ 所有测试通过!")
