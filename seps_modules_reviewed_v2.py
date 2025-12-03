"""
SEPS: Semantic-Enhanced Patch Slimming Framework (审阅版 V2 - 详细注释版)
===========================================================================

论文: SEPS: Semantic-enhanced Patch Slimming Framework for fine-grained cross-modal alignment
发表: ICLR 2026

本文件基于 seps_modules_reviewed.py，添加了与论文一一对应的详细中文注释。
所有变量名、公式编号均与论文 arXiv:2511.01390 保持一致。

论文核心贡献:
1. SDTPS (Sparse and Dense Text-Aware Patch Selection) - 稀疏与稠密文本感知的Patch选择
   - Stage 1: Semantic Scoring (语义评分) - 对应论文公式(1)-(3)
   - Stage 2: Decision and Aggregation (决策与聚合) - 对应论文公式(4)

2. HRPA (Highly-Relevant Patch-Word Alignment) - 高相关性Patch-Word对齐
   - 对应论文公式(5)

3. 损失函数
   - L_align: 对齐损失 - 对应论文公式(6)
   - L_ratio: 比例约束损失 - 对应论文公式(7)

论文符号说明:
- V = {v_1, v_2, ..., v_N}: 视觉patch特征集合，N为patch数量
- T = {t_1, t_2, ..., t_M}: 文本word特征集合，M为word数量
- E_st: 稀疏文本(sparse text)全局嵌入向量
- E_dt: 稠密文本(dense text, MLLM生成)全局嵌入向量
- E_im: 图像全局嵌入向量
- s_i^p: 第i个patch的MLP预测得分
- s_i^st: 第i个patch的稀疏文本相关性得分
- s_i^dt: 第i个patch的稠密文本相关性得分
- s_i^im: 第i个patch的图像自注意力得分
- D_s, D_d: 稀疏/稠密文本的决策矩阵 (one-hot)
- W_s, W_d: 稀疏/稠密文本的聚合权重矩阵
- V̂ = {v̂_1, ..., v̂_{N_c}}: 聚合后的patch特征，N_c为聚合后数量
- A ∈ R^{N_c × M}: patch-word相似度矩阵
- S(I, T): 图像I与文本T的最终相似度得分

使用方式:
    >>> from seps_modules_reviewed_v2 import CrossSparseAggrNet, SEPSLoss
    >>>
    >>> # 论文完整版 (包含所有论文描述的机制)
    >>> model = CrossSparseAggrNet(
    ...     use_paper_version=True,      # 启用论文公式
    ...     use_dual_aggr=True,          # 启用 W_s/W_d 双权重聚合
    ...     use_gumbel_softmax=True,     # 启用可微决策矩阵
    ... )
    >>>
    >>> # 开源代码兼容版 (与 lib/cross_net.py 行为一致)
    >>> model = CrossSparseAggrNet(use_paper_version=False)
"""

# =============================================================================
# 依赖导入
# =============================================================================
import math                                    # 数学运算: sqrt, ceil
from typing import Iterable, Optional, Tuple, Union  # 类型注解

import torch                                   # PyTorch 核心
import torch.nn as nn                          # 神经网络模块
import torch.nn.functional as F                # 函数式API: normalize, softmax, leaky_relu

# =============================================================================
# 全局配置
# =============================================================================
USE_PAPER_VERSION_DEFAULT = False  # 默认使用开源代码版本，与 lib/cross_net.py 保持一致


# =============================================================================
# 辅助函数
# =============================================================================

def l2_normalize(x: torch.Tensor, dim: int = -1, eps: float = 1e-8) -> torch.Tensor:
    """
    L2归一化函数

    将向量归一化为单位向量（模长=1）。
    归一化后，两个向量的点积等于它们的余弦相似度。

    论文中用于计算 patch-word 相似度矩阵 A:
        A_{ij} = (v̂_i)^T t_j / (||v̂_i|| ||t_j||)

    Args:
        x: 输入张量
        dim: 归一化的维度，默认-1为最后一个维度（特征维度）
        eps: 防止除零的小常数

    Returns:
        归一化后的张量，指定维度上的模长为1
    """
    return F.normalize(x, p=2, dim=dim, eps=eps)


def is_sqr(n: int) -> bool:
    """
    检查n是否为完全平方数

    用途: 判断ViT输出是否包含[CLS] token
    - ViT输出形状为 (B, N+1, C)，其中+1是[CLS] token
    - 如果patch数是完全平方数(如196=14×14)，说明没有[CLS] token
    - 如果不是完全平方数(如197=196+1)，说明包含[CLS] token

    Args:
        n: 待检查的整数（patch序列长度）

    Returns:
        True 如果n是完全平方数，否则False
    """
    a = int(math.sqrt(n))  # 计算平方根并向下取整
    return a * a == n       # 检查平方后是否等于原数


# =============================================================================
# SDTPS Stage 1: TokenSparse (Semantic Scoring + Patch Selection)
# 对应论文 Section 3.2 - Sparse and Dense Text-Aware Patch Selection Module
# =============================================================================

class TokenSparse(nn.Module):
    """
    Token稀疏选择模块 - SDTPS的第一阶段

    实现论文 Section 3.2.1 "Semantic Scoring" 中描述的语义评分机制。

    ==================== 论文公式对应 ====================

    公式(1) - Score-aware Prediction Network:
        s_i^p = σ(MLP(v_i)), i ∈ {1, ..., N}

        其中:
        - s_i^p ∈ [0,1]: 第i个patch的预测显著性得分
        - v_i: 第i个视觉patch的特征向量
        - σ: sigmoid激活函数
        - MLP: 两层全连接网络

    公式(2) - 多源注意力得分:
        s_i^{st} = Norm(v_i^T · E_{st} / d)  # 稀疏文本相关性
        s_i^{dt} = Norm(v_i^T · E_{dt} / d)  # 稠密文本相关性
        s_i^{im} = Norm(v_i^T · E_{im} / d)  # 图像自注意力

        其中:
        - E_{st}: 稀疏文本(原始caption)的全局嵌入向量
        - E_{dt}: 稠密文本(MLLM生成)的全局嵌入向量
        - E_{im}: 图像的全局嵌入向量（所有patch的平均）
        - d: 嵌入维度
        - Norm: 归一化到[0,1]范围

    公式(3) - 综合显著性得分:
        s_i = (1-2β) · s_i^p + β · (s_i^{st} + s_i^{dt} + 2·s_i^{im})

        其中 β 是权重参数，控制MLP预测与注意力得分的融合比例

    ==================== 实现细节 ====================

    两种模式:
    - use_paper_version=True:  使用完整的论文公式(1)-(3)
    - use_paper_version=False: 使用开源代码的简化版本 score = attention_x + attention_y

    可选功能:
    - Gumbel-Softmax: 论文提到的可微决策矩阵采样（Section 3.2.2）

    Args:
        embed_dim: 特征维度 d（如512）
        sparse_ratio: 保留patch的比例 ρ（如0.5表示保留50%）
        use_paper_version: 是否使用论文描述的完整机制
    """

    def __init__(
        self,
        embed_dim: int = 512,           # 特征维度 d
        sparse_ratio: float = 0.6,      # 选择比例 ρ
        use_paper_version: bool = USE_PAPER_VERSION_DEFAULT,
    ):
        super().__init__()

        # 保存配置参数
        self.embed_dim = embed_dim
        self.sparse_ratio = sparse_ratio
        self.use_paper_version = use_paper_version

        # =====================================================
        # 论文公式(1): Score-aware Prediction Network
        # s_i^p = σ(MLP(v_i))
        # =====================================================
        if use_paper_version:
            self.score_predictor = nn.Sequential(
                nn.Linear(embed_dim, embed_dim // 4),  # 第一层: d -> d/4
                nn.GELU(),                              # GELU激活（比ReLU更平滑）
                nn.Linear(embed_dim // 4, 1),          # 第二层: d/4 -> 1
                nn.Sigmoid(),                           # σ: 输出压缩到[0,1]
            )

    def forward(
        self,
        tokens: torch.Tensor,                           # V = {v_1, ..., v_N}: patch特征
        attention_x: torch.Tensor,                      # s^{im}: 图像自注意力
        attention_y: torch.Tensor,                      # s^{st}: 稀疏文本注意力
        attention_y_dense: Optional[torch.Tensor] = None,  # s^{dt}: 稠密文本注意力
        beta: float = 0.25,                             # β: 权重参数
        use_gumbel: bool = False,                       # 是否使用Gumbel-Softmax
        gumbel_tau: float = 1.0,                        # Gumbel温度参数
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        执行语义评分和patch选择

        Args:
            tokens: (B, N, C) 视觉patch特征 V = {v_1, ..., v_N}
                   B=batch_size, N=patch数量, C=特征维度d
            attention_x: (B, N) 图像自注意力得分 s^{im}
                        论文公式(2): s_i^{im} = Norm(v_i^T · E_{im} / d)
            attention_y: (B, N) 稀疏文本交叉注意力得分 s^{st}
                        论文公式(2): s_i^{st} = Norm(v_i^T · E_{st} / d)
            attention_y_dense: (B, N) 稠密文本交叉注意力得分 s^{dt}
                              论文公式(2): s_i^{dt} = Norm(v_i^T · E_{dt} / d)
            beta: 论文公式(3)中的权重参数 β
            use_gumbel: 是否使用Gumbel-Softmax可微采样（论文Section 3.2.2）
            gumbel_tau: Gumbel-Softmax的温度参数 τ

        Returns:
            select_tokens: (B, N_keep, C) 选中的patch特征 V_s 或 V_d
                          N_keep = ceil(N × ρ)，ρ为sparse_ratio
            extra_token: (B, 1, C) 融合token
                        由被丢弃patch加权融合得到，保留全局信息
            score_mask: (B, N) 决策矩阵 D_s 或 D_d
                       1表示选中（显著patch），0表示丢弃（冗余patch）
        """
        # 获取输入形状
        B_v, L_v, C = tokens.size()  # B_v=batch, L_v=N(patch数), C=d(特征维度)

        if self.use_paper_version:
            # =====================================================
            # 论文版本: 完整实现公式(1)-(3)
            # =====================================================

            # -----------------------------------------------------
            # 公式(1): Score-aware Prediction Network
            # s_i^p = σ(MLP(v_i)), i ∈ {1, ..., N}
            # -----------------------------------------------------
            s_pred = self.score_predictor(tokens).squeeze(-1)  # (B, N, 1) -> (B, N)
            # s_pred[b,i] = s_i^p 表示第b个样本第i个patch的MLP预测得分

            # -----------------------------------------------------
            # 公式(2): 归一化注意力得分到[0,1]范围
            # Norm函数: min-max归一化
            # -----------------------------------------------------
            def normalize_score(s: torch.Tensor) -> torch.Tensor:
                """
                将得分归一化到[0,1]范围
                对应论文公式(2)中的Norm操作
                """
                s_min = s.min(dim=-1, keepdim=True)[0]  # 每个样本的最小值
                s_max = s.max(dim=-1, keepdim=True)[0]  # 每个样本的最大值
                return (s - s_min) / (s_max - s_min + 1e-8)  # 防止除零

            # s_i^{im}: 图像自注意力得分（已在外部计算）
            s_im = normalize_score(attention_x)

            # s_i^{st}: 稀疏文本相关性得分
            s_st = normalize_score(attention_y)

            # s_i^{dt}: 稠密文本相关性得分
            # 论文核心创新: 引入MLLM生成的稠密文本来增强语义指导
            s_dt = (
                normalize_score(attention_y_dense)
                if attention_y_dense is not None
                else torch.zeros_like(s_st)  # 如果没有稠密文本，则为0
            )

            # -----------------------------------------------------
            # 公式(3): 综合显著性得分
            # s_i = (1-2β) · s_i^p + β · (s_i^{st} + s_i^{dt} + 2·s_i^{im})
            # -----------------------------------------------------
            # 当β=0.25时:
            # - MLP预测权重: 1-2×0.25 = 0.5
            # - 注意力权重: 0.25 × (s_st + s_dt + 2×s_im)
            # 图像自注意力权重是文本注意力的2倍，强调视觉显著性
            score = (1 - 2 * beta) * s_pred + beta * (s_st + s_dt + 2 * s_im)
        else:
            # =====================================================
            # 开源代码版本: 简化实现
            # score = attention_x + attention_y
            # =====================================================
            score = attention_x + attention_y

        # -----------------------------------------------------
        # 选择Top-K个显著patch
        # N_keep = ceil(N × ρ)
        # -----------------------------------------------------
        num_keep_token = max(1, math.ceil(L_v * self.sparse_ratio))

        # 按得分降序排序
        score_sort, score_index = torch.sort(score, dim=1, descending=True)
        # score_sort: 排序后的得分（从高到低）
        # score_index: 排序后的原始索引

        # 选择得分最高的前 num_keep_token 个
        keep_policy = score_index[:, :num_keep_token]  # (B, N_keep)

        # -----------------------------------------------------
        # 生成决策矩阵 D（论文Section 3.2.2）
        # D是one-hot矩阵：1表示显著patch，0表示冗余patch
        # -----------------------------------------------------
        if use_gumbel:
            # =====================================================
            # Gumbel-Softmax 可微采样（论文引用[maddison2016concrete]）
            # 实现 Straight-Through Estimator:
            # 前向传播使用hard决策，反向传播使用soft梯度
            # =====================================================

            # 添加Gumbel噪声实现随机采样
            gumbel_noise = -torch.log(
                -torch.log(torch.rand_like(score) + 1e-9) + 1e-9
            )
            # Gumbel-Softmax: 软决策
            soft_mask = F.softmax((score + gumbel_noise) / gumbel_tau, dim=1)

            # Hard决策: Top-K选择
            hard_mask = torch.zeros_like(score).scatter(1, keep_policy, 1.0)

            # Straight-Through: 前向用hard，反向用soft的梯度
            score_mask = hard_mask + (soft_mask - soft_mask.detach())
        else:
            # 标准Top-K选择（不可微）
            score_mask = torch.zeros_like(score).scatter(1, keep_policy, 1.0)

        # -----------------------------------------------------
        # 根据决策矩阵选择显著patch
        # V_s = {v_1^s, ..., v_{N_s}^s} 或 V_d = {v_1^d, ..., v_{N_d}^d}
        # -----------------------------------------------------
        select_tokens = torch.gather(
            tokens, dim=1, index=keep_policy.unsqueeze(-1).expand(-1, -1, C)
        )  # (B, N_keep, C)

        # -----------------------------------------------------
        # 融合被丢弃的patch为extra token
        # 保留冗余patch的部分信息，避免完全丢失
        # -----------------------------------------------------
        non_keep_policy = score_index[:, num_keep_token:]  # 被丢弃patch的索引
        non_tokens = torch.gather(
            tokens, dim=1, index=non_keep_policy.unsqueeze(-1).expand(-1, -1, C)
        )  # (B, N-N_keep, C)

        # 使用softmax加权融合（得分越高权重越大）
        non_keep_score = score_sort[:, num_keep_token:]  # 被丢弃patch的得分
        non_keep_score = F.softmax(non_keep_score, dim=1).unsqueeze(-1)  # (B, N-N_keep, 1)

        # 加权求和得到融合token
        extra_token = torch.sum(non_tokens * non_keep_score, dim=1, keepdim=True)  # (B, 1, C)

        return select_tokens, extra_token, score_mask


# =============================================================================
# SDTPS Stage 2a: TokenAggregation (单分支聚合 - 开源代码版本)
# 对应论文 Section 3.2.2 - Decision and Aggregation
# =============================================================================

class TokenAggregation(nn.Module):
    """
    Token聚合模块 - 单分支版本（与开源代码一致）

    实现论文公式(4)的简化版本，只使用单一权重矩阵W。

    ==================== 论文公式(4)简化版 ====================

    原公式(4):
        v̂_j = Σ_{i=1}^{N_s} (W_s)_{ij} · v_i^s + Σ_{i=1}^{N_d} (W_d)_{ij} · v_i^d

    简化版（本类实现）:
        v̂_j = Σ_{i=1}^{N} W_{ij} · v_i, j ∈ {1, ..., N_c}

        其中:
        - W ∈ R^{N × N_c}: 聚合权重矩阵
        - N: 输入patch数量
        - N_c: 输出聚合patch数量 (N_c < N)
        - Σ_i W_{ij} = 1: 权重归一化（通过softmax实现）

    权重矩阵学习:
        W = Softmax(MLP(V))

    Args:
        dim: 特征维度 d
        keeped_patches: 聚合后的patch数量 N_c
        dim_ratio: MLP隐藏层维度比例
    """

    def __init__(
        self,
        dim: int = 512,             # 特征维度 d
        keeped_patches: int = 64,   # 聚合后patch数量 N_c
        dim_ratio: float = 0.2,     # 隐藏层维度比例
    ):
        super().__init__()

        hidden_dim = int(dim * dim_ratio)  # 隐藏层维度

        # 权重生成网络: MLP
        # 输入: patch特征 v_i ∈ R^d
        # 输出: 该patch对各聚合位置的贡献 ∈ R^{N_c}
        self.weight = nn.Sequential(
            nn.LayerNorm(dim),              # 层归一化，稳定训练
            nn.Linear(dim, hidden_dim),     # 降维: d -> hidden_dim
            nn.GELU(),                       # GELU激活
            nn.Linear(hidden_dim, keeped_patches),  # 输出: hidden_dim -> N_c
        )

        # 可学习的缩放因子
        self.scale = nn.Parameter(torch.ones(1, 1, 1))

    def forward(
        self,
        x: torch.Tensor,                              # 输入patch特征
        keep_policy: Optional[torch.Tensor] = None,   # 可选mask
    ) -> torch.Tensor:
        """
        聚合patches

        实现: v̂_j = Σ_i W_{ij} · v_i

        Args:
            x: (B, N, C) 输入patch特征 V
            keep_policy: (B, N) 可选的mask，用于屏蔽无效位置

        Returns:
            aggregated: (B, N_c, C) 聚合后的patch特征 V̂
        """
        # 生成权重: (B, N, N_c)
        weight = self.weight(x).transpose(2, 1) * self.scale  # (B, N_c, N)
        # weight[b,j,i] 表示第i个输入patch对第j个输出patch的贡献

        # 如果有mask，屏蔽无效位置
        if keep_policy is not None:
            keep_policy = keep_policy.unsqueeze(1)  # (B, 1, N)
            weight = weight - (1 - keep_policy) * 1e10  # 无效位置设为极小值

        # Softmax归一化: Σ_i W_{ij} = 1
        weight = F.softmax(weight, dim=2)

        # 加权聚合: (B, N_c, N) × (B, N, C) -> (B, N_c, C)
        return torch.bmm(weight, x)


# =============================================================================
# SDTPS Stage 2b: DualTokenAggregation (双分支联合聚合 - 论文版本)
# 对应论文 Section 3.2.2 公式(4)的完整实现
# =============================================================================

class DualTokenAggregation(nn.Module):
    """
    双分支Token聚合模块 - 论文公式(4)的完整实现

    ==================== 论文公式(4) ====================

    v̂_j = Σ_{i=1}^{N_s} (W_s)_{ij} · v_i^s + Σ_{i=1}^{N_d} (W_d)_{ij} · v_i^d, j ∈ {1, ..., N_c}

    其中:
    - V_s = {v_1^s, ..., v_{N_s}^s}: 稀疏文本选择的显著patch
    - V_d = {v_1^d, ..., v_{N_d}^d}: 稠密文本选择的显著patch
    - W_s ∈ R^{N_s × N_c}: 稀疏文本的聚合权重矩阵
    - W_d ∈ R^{N_d × N_c}: 稠密文本的聚合权重矩阵
    - N_c: 聚合后的patch数量 (N_c < max(N_s, N_d))

    权重矩阵学习（论文描述）:
        W_s = Softmax(MLP(V_s))
        W_d = Softmax(MLP(V_d))

    论文原文 (Section 3.2.2):
    "These binary decisions are subsequently processed through an aggregation network
    that learns multiple aggregation weights and aggregates N_s and N_d significant
    patches to generate N_c informative patches."

    Args:
        dim: 特征维度 d
        keeped_patches: 聚合后的patch数量 N_c
        dim_ratio: MLP隐藏层维度比例
    """

    def __init__(
        self,
        dim: int = 512,             # 特征维度 d
        keeped_patches: int = 64,   # N_c: 聚合后patch数量
        dim_ratio: float = 0.2,     # 隐藏层维度比例
    ):
        super().__init__()

        hidden_dim = int(dim * dim_ratio)

        # W_s: 稀疏文本分支的权重网络
        # 论文: W_s = Softmax(MLP(V_s))
        self.weight_sparse = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, keeped_patches),
        )

        # W_d: 稠密文本分支的权重网络
        # 论文: W_d = Softmax(MLP(V_d))
        self.weight_dense = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, keeped_patches),
        )

        # 可学习的缩放因子
        self.scale = nn.Parameter(torch.ones(1, 1, 1))

    def _aggregate(
        self,
        x: torch.Tensor,               # 输入patch特征
        mask: Optional[torch.Tensor],  # 决策矩阵 D
        weight_net: nn.Module,         # 权重网络
    ) -> torch.Tensor:
        """
        单分支聚合辅助函数

        实现: v̂_j = Σ_i W_{ij} · v_i

        Args:
            x: (B, N, C) 输入patch特征
            mask: (B, N) 决策矩阵D，用于选择有效patch
            weight_net: 权重生成网络

        Returns:
            (B, N_c, C) 聚合后的特征
        """
        weight = weight_net(x).transpose(2, 1) * self.scale  # (B, N_c, N)

        if mask is not None:
            # 论文: "we treat the decision matrices D_s and D_d as mask matrices
            # to select the significant patch features V_s and V_d"
            weight = weight - (1 - mask.unsqueeze(1)) * 1e10

        weight = F.softmax(weight, dim=2)  # 归一化: Σ_i W_{ij} = 1
        return torch.bmm(weight, x)

    def forward(
        self,
        tokens_sparse: torch.Tensor,                    # V_s: 稀疏文本选择的patch
        tokens_dense: Optional[torch.Tensor] = None,    # V_d: 稠密文本选择的patch
        mask_sparse: Optional[torch.Tensor] = None,     # D_s: 稀疏决策矩阵
        mask_dense: Optional[torch.Tensor] = None,      # D_d: 稠密决策矩阵
    ) -> torch.Tensor:
        """
        双分支联合聚合

        实现论文公式(4):
        v̂_j = Σ_{i=1}^{N_s} (W_s)_{ij} · v_i^s + Σ_{i=1}^{N_d} (W_d)_{ij} · v_i^d

        Args:
            tokens_sparse: (B, N_s, C) 稀疏文本选择的显著patch V_s
            tokens_dense: (B, N_d, C) 稠密文本选择的显著patch V_d
            mask_sparse: (B, N_s) 可选mask，用于屏蔽tokens_sparse中的无效位置
            mask_dense: (B, N_d) 可选mask，用于屏蔽tokens_dense中的无效位置

        Returns:
            aggregated: (B, N_c, C) 聚合后的patch特征 V̂

        注意事项:
            如果传入的 tokens 已经通过 torch.gather 筛选（如 TokenSparse 的输出），
            则 mask 应传 None，因为：
            - tokens 形状: (B, N_keep, C)，N_keep < N
            - 原始 score_mask 形状: (B, N)
            维度不匹配会导致运行时错误。只有当 tokens 包含 padding 需要屏蔽时才传入 mask，
            且 mask 的第二维必须与 tokens 的第二维（序列长度）一致。
        """
        # 稀疏分支聚合: Σ (W_s)_{ij} · v_i^s
        out = self._aggregate(tokens_sparse, mask_sparse, self.weight_sparse)

        # 稠密分支聚合: + Σ (W_d)_{ij} · v_i^d
        if tokens_dense is not None:
            out = out + self._aggregate(tokens_dense, mask_dense, self.weight_dense)

        return out


# =============================================================================
# HRPA: Highly-Relevant Patch-Word Alignment
# 对应论文 Section 3.3 - Highly-Relevant Patch-Word Alignment
# =============================================================================

def mask_xattn_one_text(
    img_embs: torch.Tensor,                           # V̂: 聚合后的patch特征
    cap_i_expand: torch.Tensor,                       # T: 文本word特征
    img_mask: Optional[torch.Tensor] = None,          # patch mask
    i2t: bool = True,                                 # 是否计算双向对齐
    scan: bool = True,                                # 是否使用LeakyReLU（SCAN技巧）
    use_paper_version: bool = USE_PAPER_VERSION_DEFAULT,
    top_k: int = 5,                                   # TopK参数
    relevance_mlp: Optional[nn.Module] = None,        # 相关性学习网络
) -> torch.Tensor:
    """
    HRPA: 高相关性Patch-Word对齐函数

    实现论文 Section 3.3 中描述的相似度计算机制。

    ==================== 论文公式(5) ====================

    S(I, T) = [patch-to-word alignment] + [word-to-patch alignment]

    patch-to-word alignment:
        (1/N_c) Σ_{i=1}^{N_c} max_j(A)_{ij} + MLP(TOPK(max_j(A)_{ij}))

    word-to-patch alignment:
        (1/M) Σ_{j=1}^{M} max_i(A)_{ij} + MLP(TOPK(max_i(A)_{ij}))

    其中:
    - A ∈ R^{N_c × M}: patch-word相似度矩阵
    - A_{ij} = (v̂_i)^T t_j / (||v̂_i|| ||t_j||): 余弦相似度
    - N_c: 聚合后的patch数量
    - M: word数量
    - TOPK: 选择前K个最大值
    - MLP: 相关性学习网络

    ==================== 双向对齐策略 ====================

    论文原文:
    "We identify the most aligned textual token (or visual patch) for each
    visual patch (or textual token), and use the relevant learning network
    to transform the selected maximum scores into a scalar value."

    1. patch-to-word: 对每个patch找最相关的word，然后平均
    2. word-to-patch: 对每个word找最相关的patch，然后平均

    ==================== 论文 vs 开源代码差异 ====================

    - 论文: 使用纯余弦相似度（无LeakyReLU）
    - 开源代码: 沿用SCAN的LeakyReLU技巧
    - scan参数控制这一行为

    Args:
        img_embs: (B_v, N_c, C) 已归一化的聚合patch特征 V̂
        cap_i_expand: (B_v, M, C) 已归一化的文本word特征 T
        img_mask: (B_v, N_c) 可选的patch mask
        i2t: 是否计算双向对齐
        scan: 是否使用LeakyReLU（论文不用，开源代码用）
        use_paper_version: 是否使用论文的TopK+MLP机制
        top_k: TOPK的K值
        relevance_mlp: 相关性学习网络 MLP

    Returns:
        sim_one_text: (B_v, 1) 图像-文本相似度得分 S(I, T)
    """
    # -----------------------------------------------------
    # 计算patch-word相似度矩阵 A
    # A_{ij} = (v̂_i)^T t_j / (||v̂_i|| ||t_j||)
    # 由于输入已归一化，点积即为余弦相似度
    # -----------------------------------------------------
    # cap_i_expand: (B_v, M, C)
    # img_embs.T: (B_v, C, N_c)
    # A: (B_v, M, N_c)
    cap2img_sim = torch.bmm(cap_i_expand, img_embs.transpose(1, 2))

    # SCAN技巧: LeakyReLU抑制负相似度
    # 论文原本不使用，开源代码使用
    if scan:
        cap2img_sim = F.leaky_relu(cap2img_sim, negative_slope=0.1)

    # -----------------------------------------------------
    # word-to-patch对齐 (t2i方向)
    # 对每个word找最相关的patch: max_i(A)_{ij}
    # 然后平均: (1/M) Σ_j max_i(A)_{ij}
    # -----------------------------------------------------
    if img_mask is None:
        row_sim = cap2img_sim.max(dim=2)[0]  # (B_v, M)
    else:
        # 屏蔽无效patch
        row_sim = (cap2img_sim - 1000 * (1 - img_mask).unsqueeze(1)).max(dim=2)[0]

    row_sim_mean = row_sim.mean(dim=1, keepdim=True)  # (B_v, 1)

    # 论文公式(5): + MLP(TOPK(max_i(A)_{ij}))
    if use_paper_version and relevance_mlp is not None:
        B_v, M = row_sim.shape
        k = min(top_k, M)
        row_topk, _ = row_sim.topk(k, dim=1)  # 选择TopK个最大相似度
        if k < top_k:
            padding = torch.zeros(B_v, top_k - k, device=row_topk.device)
            row_topk = torch.cat([row_topk, padding], dim=1)
        row_sim_mean = row_sim_mean + relevance_mlp(row_topk)  # MLP映射为标量

    # -----------------------------------------------------
    # patch-to-word对齐 (i2t方向)
    # 对每个patch找最相关的word: max_j(A)_{ij}
    # 然后平均: (1/N_c) Σ_i max_j(A)_{ij}
    # -----------------------------------------------------
    if i2t:
        column_sim = cap2img_sim.max(dim=1)[0]  # (B_v, N_c)

        if img_mask is None:
            column_sim_mean = column_sim.mean(dim=1, keepdim=True)  # (B_v, 1)
        else:
            # 只对有效patch求平均
            column_sim_mean = (column_sim * img_mask).sum(dim=-1, keepdim=True) / (
                img_mask.sum(dim=-1, keepdim=True) + 1e-8
            )

        # 论文公式(5): + MLP(TOPK(max_j(A)_{ij}))
        if use_paper_version and relevance_mlp is not None:
            B_v, N = column_sim.shape
            k = min(top_k, N)
            col_topk, _ = column_sim.topk(k, dim=1)
            if k < top_k:
                padding = torch.zeros(B_v, top_k - k, device=col_topk.device)
                col_topk = torch.cat([col_topk, padding], dim=1)
            column_sim_mean = column_sim_mean + relevance_mlp(col_topk)

        # 双向对齐相加
        sim_one_text = row_sim_mean + column_sim_mean  # (B_v, 1)
    else:
        sim_one_text = row_sim_mean

    return sim_one_text


class HRPA(nn.Module):
    """
    HRPA模块的类封装版本

    将 mask_xattn_one_text 函数封装为 nn.Module，方便管理参数。

    ==================== 论文对应 ====================

    实现论文 Section 3.3 "Highly-Relevant Patch-Word Alignment"

    论文原文:
    "The HRPA module introduces relevance-aware selection with mean value
    computation to facilitate nuanced fine-grained interactions, amplifying
    highly-relevant patch-word correspondences."

    Args:
        embed_dim: 特征维度 d（接口预留）
        top_k: TOPK的K值
        use_paper_version: 是否使用论文机制
        bidirectional: 是否使用双向对齐
        scan: 是否使用LeakyReLU
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
        self.scan = scan  # 论文模式下应为False（纯余弦）

        # 论文公式(5)中的MLP: 相关性学习网络
        # 将TopK相似度映射为标量
        if use_paper_version:
            self.relevance_mlp = nn.Sequential(
                nn.Linear(top_k, top_k * 2),  # 升维
                nn.GELU(),
                nn.Linear(top_k * 2, 1),      # 输出标量
            )
        else:
            self.relevance_mlp = None

    def forward(
        self,
        patch_features: torch.Tensor,                  # V̂: 聚合后的patch特征
        word_features: torch.Tensor,                   # T: 文本word特征
        patch_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        计算图像-文本相似度 S(I, T)

        Args:
            patch_features: (B_v, N_c, C) 已归一化的聚合patch特征 V̂
            word_features: (B_v, M, C) 已归一化的文本word特征 T
            patch_mask: (B_v, N_c) 可选的patch mask

        Returns:
            similarity: (B_v, 1) 图像-文本相似度 S(I, T)
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
# 完整SDTPS模块: CrossSparseAggrNet
# 整合 TokenSparse + TokenAggregation + HRPA
# =============================================================================

class CrossSparseAggrNet(nn.Module):
    """
    完整的SDTPS + HRPA流程

    整合论文所有核心模块，实现完整的跨模态对齐流程。

    ==================== 论文流程对应 ====================

    1. 特征提取（外部完成）:
       - V = {v_1, ..., v_N}: 视觉patch特征
       - T = {t_1, ..., t_M}: 稀疏文本word特征
       - T_d: 稠密文本word特征

    2. SDTPS Stage 1 - Semantic Scoring (Section 3.2.1):
       - 计算 s^{im}, s^{st}, s^{dt}
       - 综合得分: s_i = (1-2β)·s_i^p + β·(s_i^{st} + s_i^{dt} + 2·s_i^{im})

    3. SDTPS Stage 2 - Decision and Aggregation (Section 3.2.2):
       - 生成决策矩阵 D_s, D_d (Gumbel-Softmax)
       - 聚合: v̂_j = Σ (W_s)_{ij}·v_i^s + Σ (W_d)_{ij}·v_i^d

    4. HRPA - Patch-Word Alignment (Section 3.3):
       - 计算相似度矩阵 A
       - 双向对齐: S(I, T) = patch-to-word + word-to-patch

    ==================== 论文 vs 开源代码差异 ====================

    | 特性 | 论文 | 开源代码 |
    |------|------|---------|
    | s_dt | 使用稠密文本 | 不使用 |
    | 聚合 | W_s + W_d 双权重 | 单一共享权重 |
    | 稠密文本 | 只用于选择 | 也计算相似度 |
    | HRPA | 纯余弦 | LeakyReLU |
    | 决策 | Gumbel-Softmax | Hard Top-K |

    Args:
        embed_size: 特征维度 d
        num_patches: 输入patch数量 N（不含CLS token）
        sparse_ratio: 选择比例 ρ
        aggr_ratio: 聚合比例
        use_paper_version: 是否使用论文完整机制
        top_k: HRPA的TopK参数
        use_gumbel_softmax: 是否使用Gumbel-Softmax可微决策
        gumbel_tau: Gumbel温度参数
        use_dual_aggr: 是否使用双分支聚合（W_s + W_d）
        beta: 公式(3)中的权重参数 β
    """

    def __init__(
        self,
        embed_size: int = 512,                          # 特征维度 d
        num_patches: int = 196,                         # patch数量 N
        sparse_ratio: float = 0.5,                      # 选择比例 ρ
        aggr_ratio: float = 0.4,                        # 聚合比例
        use_paper_version: bool = USE_PAPER_VERSION_DEFAULT,
        top_k: int = 5,                                 # HRPA TopK
        use_gumbel_softmax: bool = False,               # Gumbel-Softmax开关
        gumbel_tau: float = 1.0,                        # Gumbel温度
        use_dual_aggr: bool = True,                     # 双分支聚合开关
        beta: float = 0.25,                             # 公式(3)中的β
    ):
        super().__init__()

        # 保存配置
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

        # N_c: 聚合后的patch数量
        # N_c = N × ρ × aggr_ratio
        self.keeped_patches = int(num_patches * aggr_ratio * sparse_ratio)

        # =====================================================
        # SDTPS Stage 1: TokenSparse (稀疏文本分支)
        # =====================================================
        self.sparse_net_cap = TokenSparse(
            embed_dim=self.hidden_dim,
            sparse_ratio=self.sparse_ratio,
            use_paper_version=use_paper_version,
        )

        # SDTPS Stage 1: TokenSparse (稠密文本分支)
        self.sparse_net_long = TokenSparse(
            embed_dim=self.hidden_dim,
            sparse_ratio=self.sparse_ratio,
            use_paper_version=use_paper_version,
        )

        # =====================================================
        # SDTPS Stage 2: Aggregation
        # =====================================================
        if use_paper_version and use_dual_aggr:
            # 论文版本: 双分支聚合 (W_s + W_d)
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

        # =====================================================
        # HRPA: Highly-Relevant Patch-Word Alignment
        # =====================================================
        self.hrpa = HRPA(
            embed_dim=self.hidden_dim,
            top_k=self.top_k,
            use_paper_version=self.use_paper_version,
            bidirectional=True,
            # 论文: 纯余弦相似度 (scan=False)
            # 开源代码: LeakyReLU (scan=True)
            scan=not self.use_paper_version,
        )

    def forward(
        self,
        img_embs: torch.Tensor,                         # V: 图像patch特征
        cap_embs: torch.Tensor,                         # T: 稀疏文本特征
        cap_lens: torch.Tensor,                         # 稀疏文本长度
        long_cap_embs: Optional[torch.Tensor] = None,   # T_d: 稠密文本特征
        long_cap_lens: Optional[torch.Tensor] = None,   # 稠密文本长度
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Union[torch.Tensor, Tuple[torch.Tensor, ...]]]]:
        """
        计算图像-文本相似度矩阵

        完整流程:
        1. 特征L2归一化
        2. 分离CLS token和空间patch
        3. 计算图像自注意力 s^{im}
        4. 对每个文本:
           a. 计算文本-图像交叉注意力 s^{st}, s^{dt}
           b. SDTPS Stage 1: TokenSparse选择显著patch
           c. SDTPS Stage 2: TokenAggregation聚合
           d. HRPA: 计算图像-文本相似度 S(I, T)
        5. 返回相似度矩阵和决策矩阵

        Args:
            img_embs: (B_v, N+1, C) 图像patch特征（含CLS token）
            cap_embs: (B_t, L_s, C) 稀疏文本特征
            cap_lens: (B_t,) 稀疏文本实际长度
            long_cap_embs: (B_t, L_d, C) 稠密文本特征（MLLM生成）
            long_cap_lens: (B_t,) 稠密文本实际长度

        Returns:
            训练时: (similarity_matrix, score_mask)
                - similarity_matrix: (B_v, B_t) 相似度矩阵
                - score_mask: 决策矩阵，用于计算L_ratio
            推理时: similarity_matrix
        """
        B_v, L_v, C = img_embs.shape

        # -----------------------------------------------------
        # Step 1: L2归一化
        # 归一化后点积等于余弦相似度
        # -----------------------------------------------------
        img_embs_norm = F.normalize(img_embs, dim=-1)
        cap_embs_norm = F.normalize(cap_embs, dim=-1)
        long_cap_embs_norm = (
            F.normalize(long_cap_embs, dim=-1)
            if long_cap_embs is not None
            else None
        )

        # -----------------------------------------------------
        # Step 2: 分离CLS token
        # ViT输出: [CLS, patch_1, patch_2, ..., patch_N]
        # -----------------------------------------------------
        self.has_cls_token = not is_sqr(img_embs.shape[1])

        if self.has_cls_token:
            img_cls_emb = img_embs[:, 0:1, :]           # (B_v, 1, C)
            img_spatial_embs = img_embs[:, 1:, :]      # (B_v, N, C)
            img_spatial_embs_norm = img_embs_norm[:, 1:, :]
        else:
            img_cls_emb = None
            img_spatial_embs = img_embs
            img_spatial_embs_norm = img_embs_norm

        # -----------------------------------------------------
        # Step 3: 计算图像自注意力 s^{im}
        # 论文公式(2): s_i^{im} = Norm(v_i^T · E_{im} / d)
        # E_{im} = 所有patch的平均（全局嵌入）
        # -----------------------------------------------------
        with torch.no_grad():
            # E_{im}: 图像全局嵌入向量
            img_spatial_glo_norm = F.normalize(
                img_spatial_embs.mean(dim=1, keepdim=True), dim=-1
            )  # (B_v, 1, C)

            # s_i^{im} = v_i^T · E_{im} (已归一化，点积即余弦)
            img_spatial_self_attention = (
                img_spatial_glo_norm * img_spatial_embs_norm
            ).sum(dim=-1)  # (B_v, N)

        # 结果收集列表
        improve_sims = []          # 相似度列表
        long_sims = []             # 稠密文本相似度（开源代码版本用）
        score_mask_all = []        # 决策矩阵列表
        score_mask_long_all = []   # 稠密决策矩阵（开源代码版本用）

        # -----------------------------------------------------
        # Step 4: 对每个文本进行处理
        # -----------------------------------------------------
        for i in range(len(cap_lens)):
            # 获取第i个稀疏文本
            n_word = int(cap_lens[i])  # M: word数量
            cap_i = cap_embs[i, :n_word, :]  # (M, C)
            cap_i_expand = cap_embs_norm[i, :n_word, :].unsqueeze(0).repeat(B_v, 1, 1)  # (B_v, M, C)

            # 初始化稠密文本相关变量
            dense_attn = None
            long_cap_i_expand = None
            select_tokens_long = extra_token_long = score_mask_long = None

            # -------------------------------------------------
            # Step 4a: 计算文本-图像交叉注意力
            # -------------------------------------------------
            with torch.no_grad():
                # E_{st}: 稀疏文本全局嵌入
                cap_i_glo = F.normalize(cap_i.mean(0, keepdim=True).unsqueeze(0), dim=-1)  # (1, 1, C)

                # s_i^{st} = v_i^T · E_{st}
                attn_cap = (cap_i_glo * img_spatial_embs_norm).sum(dim=-1)  # (B_v, N)

                # 如果有稠密文本，计算 s_i^{dt}
                if long_cap_embs_norm is not None and long_cap_lens is not None:
                    n_word_long = int(long_cap_lens[i])
                    long_cap_i = long_cap_embs[i, :n_word_long, :]
                    long_cap_i_expand = (
                        long_cap_embs_norm[i, :n_word_long, :]
                        .unsqueeze(0)
                        .repeat(B_v, 1, 1)
                    )
                    # E_{dt}: 稠密文本全局嵌入
                    long_cap_i_glo = F.normalize(
                        long_cap_i.mean(0, keepdim=True).unsqueeze(0), dim=-1
                    )
                    # s_i^{dt} = v_i^T · E_{dt}
                    dense_attn = (long_cap_i_glo * img_spatial_embs_norm).sum(dim=-1)

            # -------------------------------------------------
            # Step 4b: SDTPS Stage 1 - TokenSparse
            # 使用稀疏文本分支选择显著patch
            # -------------------------------------------------
            select_tokens_cap, extra_token_cap, score_mask_cap = self.sparse_net_cap(
                tokens=img_spatial_embs,
                attention_x=img_spatial_self_attention,      # s^{im}
                attention_y=attn_cap,                         # s^{st}
                # 论文版本: 传入稠密文本注意力 s^{dt}
                attention_y_dense=dense_attn if self.use_paper_version else None,
                beta=self.beta,
                use_gumbel=self.use_gumbel_softmax,
                gumbel_tau=self.gumbel_tau,
            )

            # 如果有稠密文本，使用稠密文本分支选择
            if dense_attn is not None:
                select_tokens_long, extra_token_long, score_mask_long = self.sparse_net_long(
                    tokens=img_spatial_embs,
                    attention_x=img_spatial_self_attention,
                    attention_y=dense_attn,                    # 稠密文本作为主导
                    attention_y_dense=attn_cap if self.use_paper_version else None,
                    beta=self.beta,
                    use_gumbel=self.use_gumbel_softmax,
                    gumbel_tau=self.gumbel_tau,
                )

            # -------------------------------------------------
            # Step 4c: SDTPS Stage 2 - Aggregation
            # -------------------------------------------------
            if self.use_paper_version:
                # =============================================
                # 论文版本: 双分支联合聚合 + 稠密文本只用于选择
                # =============================================
                if (
                    self.use_dual_aggr
                    and select_tokens_long is not None
                    and extra_token_long is not None
                ):
                    # 公式(4): v̂_j = Σ (W_s)·v^s + Σ (W_d)·v^d
                    # 注意: 传入的tokens已经是筛选后的，不需要mask
                    # select_tokens_cap 形状: (B, N_keep, C)
                    # score_mask_cap 形状: (B, N)，维度不匹配，因此传None
                    aggr_tokens = self.aggr_net(
                        select_tokens_cap,      # V_s (已筛选)
                        select_tokens_long,     # V_d (已筛选)
                        None,                   # mask不需要，tokens已筛选
                        None,                   # mask不需要，tokens已筛选
                    )
                    # 融合两个分支的extra token
                    extra_token = torch.stack(
                        [extra_token_cap, extra_token_long], dim=0
                    ).mean(dim=0)
                    # 返回两个决策矩阵用于分别计算L_ratio
                    mask_pack: Union[torch.Tensor, Tuple[torch.Tensor, ...]] = (
                        score_mask_cap,
                        score_mask_long,
                    )
                else:
                    aggr_tokens = self.aggr_net(select_tokens_cap)
                    extra_token = extra_token_cap
                    mask_pack = (score_mask_cap,)

                # 拼接聚合tokens和extra token
                keep_spatial_tokens = torch.cat([aggr_tokens, extra_token], dim=1)

                # 添加CLS token
                if self.has_cls_token:
                    select_tokens = torch.cat((img_cls_emb, keep_spatial_tokens), dim=1)
                else:
                    select_tokens = keep_spatial_tokens

                select_tokens = F.normalize(select_tokens, dim=-1)

                # -------------------------------------------------
                # Step 4d: HRPA - 计算相似度
                # 论文: 只用稀疏文本计算相似度
                # -------------------------------------------------
                sim_one_text = self.hrpa(
                    patch_features=select_tokens,   # V̂
                    word_features=cap_i_expand,     # T (稀疏文本)
                )

                improve_sims.append(sim_one_text)
                score_mask_all.append(mask_pack)

            else:
                # =============================================
                # 开源代码版本: 两分支分别处理，相似度相加
                # =============================================
                aggr_tokens = self.aggr_net(select_tokens_cap)
                keep_spatial_tokens = torch.cat([aggr_tokens, extra_token_cap], dim=1)

                if self.has_cls_token:
                    select_tokens = torch.cat((img_cls_emb, keep_spatial_tokens), dim=1)
                else:
                    select_tokens = keep_spatial_tokens

                select_tokens = F.normalize(select_tokens, dim=-1)

                # 稀疏文本分支相似度
                sim_one_text = self.hrpa(
                    patch_features=select_tokens,
                    word_features=cap_i_expand,
                )
                improve_sims.append(sim_one_text)
                score_mask_all.append(score_mask_cap)

                # 稠密文本分支也计算相似度（开源代码行为）
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

                    # 稠密文本分支相似度
                    sim_one_text = self.hrpa(
                        patch_features=select_tokens_long_final,
                        word_features=long_cap_i_expand,  # T_d (稠密文本)
                    )
                    long_sims.append(sim_one_text)
                    score_mask_long_all.append(score_mask_long)

        # -----------------------------------------------------
        # Step 5: 整合结果
        # -----------------------------------------------------
        improve_sims = torch.cat(improve_sims, dim=1)  # (B_v, B_t)

        # 开源代码版本: 相似度相加
        if not self.use_paper_version and long_sims:
            improve_sims = improve_sims + torch.cat(long_sims, dim=1)

        # 整理决策矩阵用于L_ratio计算
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
                score_mask_out = (sparse_stack, dense_stack)  # (D_s, D_d)
            else:
                score_mask_out = (sparse_stack,)
        else:
            score_mask_out = torch.stack(score_mask_all, dim=0)
            if score_mask_long_all:
                score_mask_out = score_mask_out + torch.stack(score_mask_long_all, dim=0)

        # 返回结果
        if self.training:
            return improve_sims, score_mask_out
        return improve_sims


# =============================================================================
# 损失函数
# 对应论文 Section 3.3 末尾的损失函数描述
# =============================================================================

class ContrastiveLoss(nn.Module):
    """
    对比损失 (Triplet Loss with Hard Negative Mining)

    ==================== 论文公式(6) ====================

    L_align = Σ_{(I,T)} ([α - S(I,T) + S(I,T̂)]_+ + [α - S(I,T) + S(Î,T)]_+)

    其中:
    - α: margin（边界值）
    - (I, T): 正样本图像-文本对
    - T̂ = argmax_{j≠T} S(I,j): 最难负样本文本（hard negative）
    - Î = argmax_{i≠I} S(i,T): 最难负样本图像（hard negative）
    - [x]_+ = max(x, 0): hinge函数

    论文原文 (Section 3.3):
    "Following prior work, we adopt a bidirectional triplet loss with hard
    negative mining."

    Args:
        margin: 边界值 α（默认0.2）
        max_violation: 是否使用hard negative mining
    """

    def __init__(self, margin: float = 0.2, max_violation: bool = False):
        super().__init__()
        self.margin = margin
        self.max_violation = max_violation
        self.mask_repeat = True  # 处理重复图像（一图多文）

    def max_violation_on(self):
        """开启hard negative mining"""
        self.max_violation = True

    def max_violation_off(self):
        """关闭hard negative mining"""
        self.max_violation = False

    def forward(
        self,
        scores: torch.Tensor,                      # S(I,T): 相似度矩阵
        img_ids: Optional[torch.Tensor] = None,    # 图像ID（处理一图多文）
    ) -> torch.Tensor:
        """
        计算对比损失 L_align

        Args:
            scores: (B, B) 相似度矩阵 S(I,T)
            img_ids: (B,) 图像ID

        Returns:
            loss: L_align
        """
        # 对角线是正样本对的相似度 S(I_i, T_i)
        diagonal = scores.diag().view(scores.size(0), 1)
        d1 = diagonal.expand_as(scores)      # 按行扩展
        d2 = diagonal.t().expand_as(scores)  # 按列扩展

        # 文本检索损失: [α - S(I,T) + S(I,T̂)]_+
        cost_s = (self.margin + scores - d1).clamp(min=0)

        # 图像检索损失: [α - S(I,T) + S(Î,T)]_+
        cost_im = (self.margin + scores - d2).clamp(min=0)

        # 构建mask屏蔽正样本
        if not self.mask_repeat:
            mask = torch.eye(scores.size(0), dtype=torch.bool, device=scores.device)
        else:
            if img_ids is not None:
                # 同一图像的多个caption都视为正样本
                mask = img_ids.unsqueeze(0) == img_ids.unsqueeze(1)
            else:
                mask = torch.eye(scores.size(0), dtype=torch.bool, device=scores.device)

        # 正样本位置损失设为0
        cost_s = cost_s.masked_fill_(mask, 0)
        cost_im = cost_im.masked_fill_(mask, 0)

        # Hard Negative Mining
        if self.max_violation:
            # 只使用最难负样本
            cost_s = cost_s.max(dim=1)[0]    # T̂ = argmax S(I,j)
            cost_im = cost_im.max(dim=0)[0]  # Î = argmax S(i,T)

        return cost_s.sum() + cost_im.sum()


class RatioLoss(nn.Module):
    """
    比例约束损失

    ==================== 论文公式(7) ====================

    L_ratio = (ρ - λ_1 · (1/N_s) Σ (D_s)_i - λ_2 · (1/N_d) Σ (D_d)_i)²

    其中:
    - ρ: 目标选择比例（超参数）
    - λ_1, λ_2: 稀疏/稠密文本的权重系数
    - D_s, D_d: 决策矩阵
    - (D_s)_i ∈ {0, 1}: 第i个patch是否被选中

    论文原文 (Section 3.3):
    "Furthermore, to enhance training stability, we constrain the proportion
    of selected patches to a target value ρ, and supervise this constraint
    using mean-squared-error losses computed from the sparse-text and
    dense-text views, respectively."

    Args:
        target_ratio: 目标选择比例 ρ
        lambda_sparse: 稀疏文本权重 λ_1
        lambda_dense: 稠密文本权重 λ_2
    """

    def __init__(
        self,
        target_ratio: float = 0.5,     # ρ
        lambda_sparse: float = 1.0,    # λ_1
        lambda_dense: float = 1.0,     # λ_2
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
        计算比例约束损失 L_ratio

        Args:
            score_mask: 决策矩阵
                - 单个Tensor: 简化版本
                - Tuple[D_s, D_d]: 论文版本（分别约束）

        Returns:
            loss: L_ratio
        """
        if isinstance(score_mask, tuple) or isinstance(score_mask, list):
            # 论文版本: 分别计算稀疏和稠密的比例损失
            sparse_mask = score_mask[0]  # D_s
            dense_mask = score_mask[1] if len(score_mask) > 1 else None  # D_d

            # (ρ - (1/N_s) Σ (D_s)_i)²
            sparse_loss = (sparse_mask.float().mean() - self.target_ratio) ** 2

            # (ρ - (1/N_d) Σ (D_d)_i)²
            dense_loss = (
                (dense_mask.float().mean() - self.target_ratio) ** 2
                if dense_mask is not None
                else torch.tensor(0.0, device=sparse_mask.device)
            )

            # λ_1 · sparse_loss + λ_2 · dense_loss
            return self.lambda_sparse * sparse_loss + self.lambda_dense * dense_loss

        # 简化版本
        return (score_mask.float().mean() - self.target_ratio) ** 2


class SEPSLoss(nn.Module):
    """
    SEPS完整损失函数

    ==================== 论文公式(7)末尾 ====================

    L = L_align + L_ratio

    其中:
    - L_align: 对比损失（公式6）
    - L_ratio: 比例约束损失（公式7）

    Args:
        margin: 对比损失的margin α
        target_ratio: 目标选择比例 ρ
        ratio_weight: L_ratio的权重
        max_violation: 是否使用hard negative mining
        lambda_sparse: 稀疏文本权重 λ_1
        lambda_dense: 稠密文本权重 λ_2
    """

    def __init__(
        self,
        margin: float = 0.2,           # α
        target_ratio: float = 0.5,     # ρ
        ratio_weight: float = 2.0,     # L_ratio权重
        max_violation: bool = False,
        lambda_sparse: float = 1.0,    # λ_1
        lambda_dense: float = 1.0,     # λ_2
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
        """动态设置hard negative mining"""
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
        """
        计算总损失 L = L_align + L_ratio

        Args:
            similarity_matrix: (B_v, B_t) 相似度矩阵
            score_mask: 决策矩阵
            img_ids: (B,) 图像ID

        Returns:
            total_loss: L
            align_loss: L_align（用于监控）
            ratio_loss: L_ratio（用于监控）
        """
        align_loss = self.contrastive_loss(similarity_matrix, img_ids)
        r_loss = self.ratio_loss(score_mask)
        total_loss = align_loss + self.ratio_weight * r_loss

        return total_loss, align_loss, r_loss


# =============================================================================
# 便捷别名
# =============================================================================

SDTPS = CrossSparseAggrNet           # SDTPS模块别名
SDTPS_TokenSparse = TokenSparse      # TokenSparse别名
SDTPS_TokenAggregation = TokenAggregation  # TokenAggregation别名
HRPA_function = mask_xattn_one_text  # HRPA函数别名


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
    创建SEPS模型的便捷工厂函数

    Args:
        embed_size: 特征维度 d
        num_patches: patch数量 N
        sparse_ratio: 选择比例 ρ
        aggr_ratio: 聚合比例
        use_paper_version: 是否使用论文完整机制
        use_gumbel_softmax: 是否使用Gumbel-Softmax
        gumbel_tau: Gumbel温度

    Returns:
        model: 配置好的CrossSparseAggrNet实例
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
    简单形状自检

    验证模型输入输出形状是否正确
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 测试参数
    B = 2                  # batch size
    num_patches = 16       # N: patch数量
    embed_size = 32        # d: 特征维度
    sparse_len = 5         # M_s: 稀疏文本长度
    dense_len = 8          # M_d: 稠密文本长度

    # 构造测试数据
    img = torch.randn(B, num_patches + 1, embed_size, device=device)  # (B, N+1, d)
    cap = torch.randn(B, sparse_len, embed_size, device=device)       # (B, M_s, d)
    cap_len = torch.full((B,), sparse_len, device=device)
    long_cap = torch.randn(B, dense_len, embed_size, device=device)   # (B, M_d, d)
    long_len = torch.full((B,), dense_len, device=device)

    print("=" * 70)
    print("测试开源代码版本 (use_paper_version=False)")
    print("=" * 70)

    seps_code = create_seps_model(
        embed_size=embed_size,
        num_patches=num_patches,
        sparse_ratio=0.5,
        aggr_ratio=0.5,
        use_paper_version=False,
    ).to(device)
    seps_code.train()

    sims, mask = seps_code(img, cap, cap_len, long_cap, long_len)
    print(f"Similarity shape: {sims.shape}")
    print(f"Mask shape: {mask[0].shape if isinstance(mask, tuple) else mask.shape}")
    print(f"Parameters: {sum(p.numel() for p in seps_code.parameters()):,}")

    print("\n" + "=" * 70)
    print("测试论文版本 (use_paper_version=True)")
    print("=" * 70)

    seps_paper = create_seps_model(
        embed_size=embed_size,
        num_patches=num_patches,
        sparse_ratio=0.5,
        aggr_ratio=0.5,
        use_paper_version=True,
    ).to(device)
    seps_paper.train()

    sims, mask = seps_paper(img, cap, cap_len, long_cap, long_len)
    print(f"Similarity shape: {sims.shape}")
    print(f"Mask shape: {tuple(m.shape for m in mask) if isinstance(mask, tuple) else mask.shape}")
    print(f"Parameters: {sum(p.numel() for p in seps_paper.parameters()):,}")

    print("\n✓ 所有测试通过!")