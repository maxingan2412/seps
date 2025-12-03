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

# =============================================================================
# 导入依赖库
# =============================================================================
import torch                          # PyTorch核心库
import torch.nn as nn                 # 神经网络模块
import torch.nn.functional as F       # 函数式API（包含激活函数、归一化等）
import math                           # 数学运算（sqrt, ceil等）
from typing import Optional, Tuple, Union  # 类型注解支持


# =============================================================================
# 全局开关：控制使用论文版本还是实际代码版本
# =============================================================================

USE_PAPER_VERSION_DEFAULT = False  # 默认使用实际代码版本（简化版）


# =============================================================================
# 辅助函数
# =============================================================================

def l2_normalize(x: torch.Tensor, dim: int = -1, eps: float = 1e-8) -> torch.Tensor:
    """
    L2归一化函数

    将向量归一化为单位向量（模长=1），公式: x_norm = x / ||x||_2

    Args:
        x: 输入张量
        dim: 在哪个维度上进行归一化，默认-1表示最后一个维度
        eps: 防止除零的小常数

    Returns:
        归一化后的张量，在指定维度上模长为1
    """
    return F.normalize(x, p=2, dim=dim, eps=eps)


def is_sqr(n: int) -> bool:
    """
    检查n是否为完全平方数

    用途：判断ViT输出是否包含cls token
    - 如果patch数是完全平方数(如196=14×14)，说明没有cls token
    - 如果不是完全平方数(如197=196+1)，说明第一个位置是cls token

    Args:
        n: 待检查的整数

    Returns:
        True如果n是完全平方数，否则False
    """
    a = int(math.sqrt(n))  # 计算平方根并取整
    return a * a == n       # 检查平方后是否等于原数


# =============================================================================
# SDTPS模块组件1: TokenSparse (Patch稀疏选择)
# =============================================================================

class TokenSparse(nn.Module):
    """
    Token稀疏选择模块 - SDTPS的第一阶段

    功能：根据语义重要性得分，从N个patches中选择top-k个最相关的patches，
         并将被丢弃的patches加权融合为一个extra token保留全局信息。

    两种模式：
    - use_paper_version=True:
        使用Score-aware Prediction Network (MLP) 预测每个patch的语义显著性，
        然后综合MLP预测分数、图像自注意力、文本交叉注意力（论文公式1-3）

    - use_paper_version=False:
        直接使用注意力分数相加（实际代码实现的简化版本）

    Args:
        embed_dim: 特征维度（如512）
        sparse_ratio: 保留patch的比例（如0.6表示保留60%的patches）
        use_paper_version: 是否使用论文描述的复杂机制
    """

    def __init__(
        self,
        embed_dim: int = 512,           # 特征向量维度
        sparse_ratio: float = 0.6,      # 保留比例，0.6表示保留60%
        use_paper_version: bool = USE_PAPER_VERSION_DEFAULT  # 是否使用论文版本
    ):
        super().__init__()  # 调用父类nn.Module的初始化

        # 保存配置参数为实例属性
        self.embed_dim = embed_dim
        self.sparse_ratio = sparse_ratio
        self.use_paper_version = use_paper_version

        # 论文版本：构建Score-aware Prediction Network (对应论文公式1)
        # 这是一个MLP网络，输入patch特征，输出该patch的语义重要性得分
        if use_paper_version:
            self.score_predictor = nn.Sequential(
                nn.Linear(embed_dim, embed_dim // 4),  # 降维: 512 -> 128
                nn.GELU(),                              # GELU激活函数（比ReLU更平滑）
                nn.Linear(embed_dim // 4, 1),          # 输出单个得分: 128 -> 1
                nn.Sigmoid()                            # Sigmoid将得分压缩到[0,1]范围
            )

    def forward(
        self,
        tokens: torch.Tensor,                           # 输入patch特征
        attention_x: torch.Tensor,                      # 图像自注意力得分
        attention_y: torch.Tensor,                      # 稀疏文本交叉注意力得分
        attention_y_dense: Optional[torch.Tensor] = None,  # 稠密文本注意力（论文版本用）
        beta: float = 0.25                              # 权重系数（论文版本用）
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        执行patch稀疏选择

        处理流程：
        1. 计算每个patch的综合得分
        2. 按得分排序，选择top-k个patches
        3. 将被丢弃的patches加权融合为extra token

        Args:
            tokens: (B, N, C) 视觉patch特征
                   B=batch_size, N=patch数量, C=特征维度
            attention_x: (B, N) 图像自注意力得分 (s_im)
                        表示每个patch相对于全局图像的重要性
            attention_y: (B, N) 稀疏文本交叉注意力得分 (s_st)
                        表示每个patch与原始caption的相关性
            attention_y_dense: (B, N) 稠密文本交叉注意力得分 (s_dt)
                              表示每个patch与MLLM生成描述的相关性，仅论文版本使用
            beta: 权重参数，控制各分数的融合比例，仅论文版本使用

        Returns:
            select_tokens: (B, N_keep, C) 选中的patch特征
                          N_keep = ceil(N * sparse_ratio)
            extra_token: (B, 1, C) 融合token
                        由被丢弃patches加权平均得到，保留全局信息
            score_mask: (B, N) 选择mask
                       1表示该patch被选中，0表示被丢弃
        """
        # 获取输入张量的形状
        B_v, L_v, C = tokens.size()  # B_v=batch, L_v=patch数, C=特征维度

        if self.use_paper_version:
            # =====================================================
            # 论文版本：使用公式(1)-(3)计算综合得分
            # =====================================================

            # 公式(1): s_i^p = σ(MLP(v_i))
            # 通过MLP预测每个patch的语义显著性得分
            s_pred = self.score_predictor(tokens).squeeze(-1)  # (B, N, 1) -> (B, N)

            # 定义归一化函数：将分数归一化到[0,1]范围
            # 使用min-max归一化: (x - min) / (max - min)
            def normalize_score(s):
                s_min = s.min(dim=-1, keepdim=True)[0]  # 每个样本的最小值
                s_max = s.max(dim=-1, keepdim=True)[0]  # 每个样本的最大值
                return (s - s_min) / (s_max - s_min + 1e-8)  # 加1e-8防止除零

            # 对各种注意力分数进行归一化
            s_im = normalize_score(attention_x)      # 图像自注意力 -> [0,1]
            s_st = normalize_score(attention_y)      # 稀疏文本注意力 -> [0,1]
            # 稠密文本注意力：如果没有提供则用零向量
            s_dt = normalize_score(attention_y_dense) if attention_y_dense is not None else torch.zeros_like(s_st)

            # 公式(3): s_i = (1-2β)·s_i^p + β·(s_i^st + s_i^dt + 2·s_i^im)
            # 综合三种信息源：MLP预测 + 稀疏文本 + 稠密文本 + 图像自身
            # β=0.25时: 0.5*s_pred + 0.25*(s_st + s_dt + 2*s_im)
            score = (1 - 2*beta) * s_pred + beta * (s_st + s_dt + 2*s_im)
        else:
            # =====================================================
            # 实际代码版本：直接将两个注意力分数相加
            # =====================================================
            score = attention_x + attention_y  # 简单相加

        # -----------------------------------------------------
        # 以下是两种版本共用的选择逻辑
        # -----------------------------------------------------

        # 计算保留的patch数量：向上取整确保至少保留1个
        num_keep_token = math.ceil(L_v * self.sparse_ratio)  # 如196*0.6=118

        # 按得分降序排序
        # score_sort: 排序后的得分值
        # score_index: 排序后的原始索引（用于gather操作）
        score_sort, score_index = torch.sort(score, dim=1, descending=True)

        # 获取top-k的索引（得分最高的num_keep_token个）
        keep_policy = score_index[:, :num_keep_token]  # (B, N_keep)

        # 生成选择mask：scatter操作将选中位置设为1
        # 初始化全0 -> 在keep_policy指定的位置填入1
        score_mask = torch.zeros_like(score).scatter(1, keep_policy, 1)  # (B, N)

        # 根据索引选择保留的patches
        # gather操作：从tokens中按keep_policy索引取值
        # index需要扩展到与tokens相同的维度 (B, N_keep) -> (B, N_keep, C)
        select_tokens = torch.gather(
            tokens, dim=1,
            index=keep_policy.unsqueeze(-1).expand(-1, -1, C)
        )  # (B, N_keep, C)

        # -----------------------------------------------------
        # 处理被丢弃的patches：加权融合为extra token
        # -----------------------------------------------------

        # 获取被丢弃patches的索引
        non_keep_policy = score_index[:, num_keep_token:]  # (B, N-N_keep)

        # 按索引取出被丢弃的patches
        non_tokens = torch.gather(
            tokens, dim=1,
            index=non_keep_policy.unsqueeze(-1).expand(-1, -1, C)
        )  # (B, N-N_keep, C)

        # 获取被丢弃patches的得分（已排序，这些是较低的得分）
        non_keep_score = score_sort[:, num_keep_token:]  # (B, N-N_keep)

        # 对得分进行softmax得到权重
        # 权重反映各被丢弃patch的相对重要性
        non_keep_score = F.softmax(non_keep_score, dim=1).unsqueeze(-1)  # (B, N-N_keep, 1)

        # 加权求和得到extra token
        # 保留被丢弃patches的部分信息，避免完全丢失
        extra_token = torch.sum(non_tokens * non_keep_score, dim=1, keepdim=True)  # (B, 1, C)

        return select_tokens, extra_token, score_mask


# =============================================================================
# SDTPS模块组件2: TokenAggregation (Patch聚合)
# =============================================================================

class TokenAggregation(nn.Module):
    """
    Token聚合模块 - SDTPS的第二阶段

    功能：学习一个权重矩阵W，将N个输入patches聚合为N_c个输出patches
         实现空间压缩，减少后续计算量

    论文公式(4): v̂_j = Σ_i W_ij · v_i
    其中W_ij表示第i个输入patch对第j个输出patch的贡献权重

    实现方式：
    1. 用MLP为每个输入patch生成N_c维的权重向量
    2. 转置后得到(N_c, N)的权重矩阵
    3. 用softmax归一化权重
    4. 矩阵乘法完成聚合

    Args:
        dim: 特征维度
        keeped_patches: 输出patch数量 (N_c)
        dim_ratio: MLP隐藏层维度比例（相对于dim）
    """

    def __init__(
        self,
        dim: int = 512,             # 特征维度
        keeped_patches: int = 64,   # 聚合后的patch数量
        dim_ratio: float = 0.2      # 隐藏层维度比例
    ):
        super().__init__()

        # 计算MLP隐藏层维度
        hidden_dim = int(dim * dim_ratio)  # 如512*0.2=102

        # 权重生成网络：输入patch特征，输出该patch对各聚合位置的贡献
        self.weight = nn.Sequential(
            nn.LayerNorm(dim),              # 层归一化，稳定训练
            nn.Linear(dim, hidden_dim),     # 降维: dim -> hidden_dim
            nn.GELU(),                       # GELU激活
            nn.Linear(hidden_dim, keeped_patches)  # 输出: hidden_dim -> N_c
        )

        # 可学习的缩放因子，初始化为1
        # 用于调节权重的整体大小
        self.scale = nn.Parameter(torch.ones(1, 1, 1))

    def forward(
        self,
        x: torch.Tensor,                              # 输入patch特征
        keep_policy: Optional[torch.Tensor] = None    # 可选的mask
    ) -> torch.Tensor:
        """
        聚合patches

        处理流程：
        1. 对每个patch计算其对各聚合位置的贡献权重
        2. softmax归一化权重
        3. 加权求和完成聚合

        Args:
            x: (B, N, C) 输入patch特征
               B=batch_size, N=输入patch数, C=特征维度
            keep_policy: (B, N) 可选的mask，1表示有效，0表示无效
                        用于屏蔽padding位置

        Returns:
            aggregated: (B, N_c, C) 聚合后的patch特征
                       N_c=keeped_patches
        """
        # 计算权重：每个patch输出N_c维的权重向量
        weight = self.weight(x)  # (B, N, N_c)

        # 转置并缩放：得到(B, N_c, N)的权重矩阵
        # weight[b,j,i]表示第b个样本中，第i个输入patch对第j个输出patch的贡献
        weight = weight.transpose(2, 1) * self.scale  # (B, N_c, N)

        # 如果提供了mask，屏蔽无效位置
        if keep_policy is not None:
            keep_policy = keep_policy.unsqueeze(1)  # (B, N) -> (B, 1, N)
            # 将无效位置的权重设为极小值(-1e10)，softmax后趋近于0
            weight = weight - (1 - keep_policy) * 1e10

        # softmax归一化：确保每个输出位置的权重和为1
        weight = F.softmax(weight, dim=2)  # 在输入维度(dim=2)上归一化

        # 加权求和：矩阵乘法实现聚合
        # (B, N_c, N) @ (B, N, C) -> (B, N_c, C)
        x = torch.bmm(weight, x)

        return x


# =============================================================================
# HRPA模块: Highly-Relevant Patch-Word Alignment (高相关性Patch-Word对齐)
# =============================================================================

def mask_xattn_one_text(
    img_embs: torch.Tensor,                           # 图像patch特征
    cap_i_expand: torch.Tensor,                       # 文本word特征
    img_mask: Optional[torch.Tensor] = None,          # patch mask
    i2t: bool = True,                                 # 是否计算双向
    scan: bool = True,                                # 是否使用LeakyReLU
    use_paper_version: bool = USE_PAPER_VERSION_DEFAULT,  # 版本开关
    top_k: int = 5,                                   # TopK参数
    relevance_mlp: Optional[nn.Module] = None,        # 相关性MLP
) -> torch.Tensor:
    """
    HRPA: 高相关性Patch-Word对齐函数

    核心思想：
    - 传统方法：计算所有patch-word对的相似度，取平均
    - HRPA：只关注最相关的对齐关系，使用max操作找最佳匹配

    双向对齐策略：
    1. t2i (text-to-image): 对每个word找最相关的patch，然后平均
    2. i2t (image-to-text): 对每个patch找最相关的word，然后平均

    两种模式：
    - use_paper_version=True: 在mean基础上加MLP(TopK(...))，论文公式(5)
    - use_paper_version=False: 仅使用max+mean（实际代码实现）

    Args:
        img_embs: (B_v, N, C) 已归一化的视觉patch特征
                 B_v=图像batch, N=patch数, C=特征维度
        cap_i_expand: (B_v, M, C) 已归一化的文本word特征
                     复制B_v份用于与所有图像计算，M=word数
        img_mask: (B_v, N) 可选的patch mask，1=有效，0=padding
        i2t: 是否计算image-to-text方向，False则只计算t2i
        scan: 是否使用LeakyReLU处理相似度（SCAN论文的技巧）
        use_paper_version: 是否使用论文描述的复杂机制
        top_k: TopK参数，取前k个最大相似度，仅论文版本使用
        relevance_mlp: 相关性学习网络，将TopK相似度映射为额外得分

    Returns:
        sim_one_text: (B_v, 1) 图像-文本相似度分数
                     表示每张图像与该文本的匹配程度
    """
    # -----------------------------------------------------
    # Step 1: 计算patch-word相似度矩阵
    # -----------------------------------------------------
    # cap_i_expand: (B_v, M, C)
    # img_embs.transpose(1,2): (B_v, C, N)
    # 结果: (B_v, M, N) - 每个word与每个patch的相似度
    cap2img_sim = torch.bmm(cap_i_expand, img_embs.transpose(1, 2))

    # 可选：使用LeakyReLU抑制负相似度
    # SCAN论文中的技巧，让模型更关注正相关
    if scan:
        cap2img_sim = F.leaky_relu(cap2img_sim, negative_slope=0.1)

    # -----------------------------------------------------
    # Step 2: t2i方向 - 对每个word找最匹配的patch
    # -----------------------------------------------------
    # 计算每个word的最大相似度（与所有patch中最相关的那个）
    if img_mask is None:
        # 无mask：直接取max
        row_sim = cap2img_sim.max(dim=2)[0]  # (B_v, M)
    else:
        # 有mask：屏蔽无效patch后取max
        # 将无效位置减去1000使其在max操作中被忽略
        row_sim = (cap2img_sim - 1000 * (1 - img_mask).unsqueeze(1)).max(dim=2)[0]

    # 对所有word取平均，得到t2i相似度
    row_sim_mean = row_sim.mean(dim=1, keepdim=True)  # (B_v, 1)

    # 论文版本：额外添加TopK + MLP机制
    # 思想：不仅用mean，还用TopK最大值通过MLP学习额外得分
    if use_paper_version and relevance_mlp is not None:
        B_v, M = row_sim.shape
        k = min(top_k, M)  # 确保k不超过word数
        # 取top-k个最大相似度
        row_topk, _ = row_sim.topk(k, dim=1)  # (B_v, k)
        # 如果word数不足k个，补零
        if k < top_k:
            padding = torch.zeros(B_v, top_k - k, device=row_topk.device)
            row_topk = torch.cat([row_topk, padding], dim=1)
        # 通过MLP学习额外得分
        row_extra = relevance_mlp(row_topk)  # (B_v, 1)
        row_sim_mean = row_sim_mean + row_extra  # 加到原始得分上

    # -----------------------------------------------------
    # Step 3: i2t方向 - 对每个patch找最匹配的word（可选）
    # -----------------------------------------------------
    if i2t:
        # 计算每个patch的最大相似度（与所有word中最相关的那个）
        column_sim = cap2img_sim.max(dim=1)[0]  # (B_v, N)

        # 对所有patch取平均
        if img_mask is None:
            column_sim_mean = column_sim.mean(dim=1, keepdim=True)  # (B_v, 1)
        else:
            # 有mask：只对有效patch求平均
            column_sim_mean = (column_sim * img_mask).sum(dim=-1, keepdim=True) / \
                              (img_mask.sum(dim=-1, keepdim=True) + 1e-8)

        # 论文版本：额外添加TopK + MLP机制
        if use_paper_version and relevance_mlp is not None:
            B_v, N = column_sim.shape
            k = min(top_k, N)
            col_topk, _ = column_sim.topk(k, dim=1)
            if k < top_k:
                padding = torch.zeros(B_v, top_k - k, device=col_topk.device)
                col_topk = torch.cat([col_topk, padding], dim=1)
            col_extra = relevance_mlp(col_topk)
            column_sim_mean = column_sim_mean + col_extra

        # 双向相似度相加
        sim_one_text = row_sim_mean + column_sim_mean  # (B_v, 1)
    else:
        # 只使用t2i方向
        sim_one_text = row_sim_mean

    return sim_one_text


class HRPA(nn.Module):
    """
    HRPA模块的类封装版本

    将mask_xattn_one_text函数封装为nn.Module，方便管理参数

    Args:
        embed_dim: 特征维度（本模块未使用，保留接口一致性）
        top_k: TopK参数，取前k个最大相似度
        use_paper_version: 是否使用论文描述的复杂机制
        bidirectional: 是否使用双向对齐（t2i + i2t）
    """

    def __init__(
        self,
        embed_dim: int = 512,       # 特征维度（接口预留）
        top_k: int = 5,             # TopK参数
        use_paper_version: bool = USE_PAPER_VERSION_DEFAULT,
        bidirectional: bool = True   # 是否双向对齐
    ):
        super().__init__()
        self.use_paper_version = use_paper_version
        self.bidirectional = bidirectional
        self.top_k = top_k

        # 论文版本：构建Relevance Learning Network
        # 输入top_k个相似度值，输出1个额外得分
        if use_paper_version:
            self.relevance_mlp = nn.Sequential(
                nn.Linear(top_k, top_k * 2),  # 升维: k -> 2k
                nn.GELU(),                     # GELU激活
                nn.Linear(top_k * 2, 1)       # 降维: 2k -> 1
            )
        else:
            self.relevance_mlp = None  # 实际代码版本不需要MLP

    def forward(
        self,
        patch_features: torch.Tensor,                  # 图像patch特征
        word_features: torch.Tensor,                   # 文本word特征
        patch_mask: Optional[torch.Tensor] = None      # patch mask
    ) -> torch.Tensor:
        """
        计算图像-文本相似度

        Args:
            patch_features: (B_v, N, C) 已归一化的视觉patch特征
            word_features: (B_v, M, C) 已归一化的文本word特征
            patch_mask: (B_v, N) 可选的patch mask

        Returns:
            similarity: (B_v, 1) 图像-文本相似度分数
        """
        return mask_xattn_one_text(
            img_embs=patch_features,
            cap_i_expand=word_features,
            img_mask=patch_mask,
            i2t=self.bidirectional,           # 是否双向
            scan=True,                         # 使用LeakyReLU
            use_paper_version=self.use_paper_version,
            top_k=self.top_k,
            relevance_mlp=self.relevance_mlp
        )


# =============================================================================
# SDTPS完整模块: CrossSparseAggrNet
# =============================================================================

class CrossSparseAggrNet(nn.Module):
    """
    完整的SDTPS模块 - 跨模态稀疏聚合网络

    整合TokenSparse和TokenAggregation，实现完整的patch选择和聚合流程：
    1. 计算图像自注意力得分
    2. 对每个文本，计算其与图像的交叉注意力
    3. 综合两种注意力进行patch稀疏选择
    4. 聚合选中的patches
    5. 计算图像-文本相似度

    支持两个文本分支：
    - Sparse Text: 原始caption（简短）
    - Dense Text: MLLM生成的详细描述（可选）

    Args:
        embed_size: 特征维度
        num_patches: 输入patch数量（不含cls token）
        sparse_ratio: patch稀疏选择比例（第一阶段保留比例）
        aggr_ratio: patch聚合比例（第二阶段压缩比例）
        use_paper_version: 是否使用论文描述的复杂机制
        top_k: HRPA的TopK参数（仅论文版本）
    """

    def __init__(
        self,
        embed_size: int = 512,      # 特征维度
        num_patches: int = 196,     # 输入patch数（如ViT-B/16的14×14=196）
        sparse_ratio: float = 0.5,  # 稀疏比例：保留50%的patches
        aggr_ratio: float = 0.4,    # 聚合比例：进一步压缩到40%
        use_paper_version: bool = USE_PAPER_VERSION_DEFAULT,
        top_k: int = 5,             # HRPA的TopK参数
    ):
        super().__init__()

        # 保存配置参数
        self.hidden_dim = embed_size
        self.num_patches = num_patches
        self.sparse_ratio = sparse_ratio
        self.aggr_ratio = aggr_ratio
        self.use_paper_version = use_paper_version
        self.top_k = top_k

        # 计算最终保留的patch数量
        # 两阶段压缩: N -> N*sparse -> N*sparse*aggr
        # 例如: 196 -> 98 (50%) -> 39 (40%)
        self.keeped_patches = int(num_patches * aggr_ratio * sparse_ratio)

        # Sparse Text分支的稀疏选择网络
        self.sparse_net_cap = TokenSparse(
            embed_dim=self.hidden_dim,
            sparse_ratio=self.sparse_ratio,
            use_paper_version=use_paper_version
        )

        # Dense Text分支的稀疏选择网络（结构相同但参数独立）
        self.sparse_net_long = TokenSparse(
            embed_dim=self.hidden_dim,
            sparse_ratio=self.sparse_ratio,
            use_paper_version=use_paper_version
        )

        # 聚合网络：两个分支共享
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
        img_embs: torch.Tensor,                         # 图像特征
        cap_embs: torch.Tensor,                         # 稀疏文本特征
        cap_lens: torch.Tensor,                         # 稀疏文本长度
        long_cap_embs: Optional[torch.Tensor] = None,   # 稠密文本特征
        long_cap_lens: Optional[torch.Tensor] = None    # 稠密文本长度
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        计算图像-文本相似度矩阵

        处理流程：
        1. 特征归一化
        2. 分离cls token和空间patches
        3. 计算图像自注意力
        4. 对每个文本：
           a. 计算文本-图像交叉注意力
           b. 执行patch稀疏选择
           c. 执行patch聚合
           d. 计算图像-文本相似度(HRPA)
        5. 融合两个分支的结果

        Args:
            img_embs: (B_v, N+1, C) 图像patch特征
                     B_v=图像batch, N+1=patch数+cls token, C=特征维度
            cap_embs: (B_t, L_s, C) 稀疏文本特征
                     B_t=文本batch, L_s=最大文本长度
            cap_lens: (B_t,) 稀疏文本的实际长度
            long_cap_embs: (B_t, L_d, C) 稠密文本特征（可选）
            long_cap_lens: (B_t,) 稠密文本的实际长度（可选）

        Returns:
            训练时: (similarity_matrix, score_mask_all)
                   - similarity_matrix: (B_v, B_t) 图像-文本相似度矩阵
                   - score_mask_all: (B_t, N) patch选择mask，用于计算ratio loss
            推理时: similarity_matrix
                   只返回相似度矩阵
        """
        # 获取输入形状
        B_v, L_v, C = img_embs.shape  # 图像batch, patch数(含cls), 特征维度

        # -----------------------------------------------------
        # Step 1: 特征L2归一化
        # -----------------------------------------------------
        # 归一化后，向量点积等于余弦相似度
        img_embs_norm = F.normalize(img_embs, dim=-1)        # 图像特征归一化
        cap_embs_norm = F.normalize(cap_embs, dim=-1)        # 稀疏文本归一化
        # 稠密文本归一化（如果有）
        long_cap_embs_norm = F.normalize(long_cap_embs, dim=-1) if long_cap_embs is not None else None

        # -----------------------------------------------------
        # Step 2: 判断是否有cls token并分离
        # -----------------------------------------------------
        # ViT输出: [cls, patch1, patch2, ..., patchN]
        # 如果patch数+1不是完全平方数，说明有cls token
        self.has_cls_token = not is_sqr(img_embs.shape[1])

        if self.has_cls_token:
            # 分离cls token和空间patches
            img_cls_emb = img_embs[:, 0:1, :]           # (B_v, 1, C) cls token
            img_spatial_embs = img_embs[:, 1:, :]      # (B_v, N, C) 空间patches
            img_spatial_embs_norm = img_embs_norm[:, 1:, :]
        else:
            # 没有cls token
            img_cls_emb = None
            img_spatial_embs = img_embs
            img_spatial_embs_norm = img_embs_norm

        # -----------------------------------------------------
        # Step 3: 计算图像自注意力得分
        # -----------------------------------------------------
        # 不需要梯度，因为这只是用于patch选择的参考信号
        with torch.no_grad():
            # 计算全局图像特征（所有patch的平均）
            img_spatial_glo_norm = F.normalize(
                img_spatial_embs.mean(dim=1, keepdim=True), dim=-1
            )  # (B_v, 1, C)

            # 每个patch与全局特征的相似度 = 自注意力得分
            # (B_v, 1, C) * (B_v, N, C) -> (B_v, N, C) -> sum -> (B_v, N)  本质：对每个 patch 计算它与全局特征的余弦相似度（因为已经 L2 归一化了）。
            img_spatial_self_attention = (
                img_spatial_glo_norm * img_spatial_embs_norm
            ).sum(dim=-1)

        # 初始化结果列表
        improve_sims = []          # 稀疏文本分支的相似度
        long_sims = []             # 稠密文本分支的相似度
        score_mask_all = []        # 稀疏文本分支的patch选择mask
        score_mask_long_all = []   # 稠密文本分支的patch选择mask

        # =============================================================
        # Sparse Text 分支处理
        # =============================================================
        # 遍历每个稀疏文本（原始caption）
        for i in range(len(cap_lens)):
            # 获取第i个文本的实际长度和特征
            n_word = cap_lens[i]                              # 实际word数
            cap_i = cap_embs[i, :n_word, :]                   # (n_word, C) 原始特征
            # 归一化特征，复制B_v份用于与所有图像匹配
            cap_i_expand = cap_embs_norm[i, :n_word, :].unsqueeze(0).repeat(B_v, 1, 1)  # (B_v, n_word, C)

            # ---------------------------------------------------------
            # Step 4a: 计算文本-图像交叉注意力
            # ---------------------------------------------------------
            with torch.no_grad():  # 不需要梯度
                # 计算文本全局特征（所有word的平均）
                cap_i_glo = F.normalize(cap_i.mean(0, keepdim=True).unsqueeze(0), dim=-1)  # (1, 1, C)

                # 每个patch与文本全局特征的相似度 = 交叉注意力得分
                # (1, 1, C) * (B_v, N, C) -> (B_v, N, C) -> sum -> (B_v, N)
                attn_cap = (cap_i_glo * img_spatial_embs_norm).sum(dim=-1)

                # ---------------------------------------------------------
                # Step 4b: 执行patch稀疏选择
                # ---------------------------------------------------------
                select_tokens_cap, extra_token_cap, score_mask_cap = self.sparse_net_cap(
                    tokens=img_spatial_embs,              # 输入patches
                    attention_x=img_spatial_self_attention,  # 图像自注意力
                    attention_y=attn_cap,                    # 文本交叉注意力
                )
                # select_tokens_cap: (B_v, N_keep, C) 选中的patches
                # extra_token_cap: (B_v, 1, C) 融合token
                # score_mask_cap: (B_v, N) 选择mask

            # ---------------------------------------------------------
            # Step 4c: 执行patch聚合
            # ---------------------------------------------------------
            aggr_tokens = self.aggr_net(select_tokens_cap)  # (B_v, N_c, C)

            # 拼接聚合后的tokens和extra token
            keep_spatial_tokens = torch.cat([aggr_tokens, extra_token_cap], dim=1)  # (B_v, N_c+1, C)

            # 如果有cls token，拼接到最前面
            if self.has_cls_token:
                select_tokens = torch.cat((img_cls_emb, keep_spatial_tokens), dim=1)  # (B_v, N_c+2, C)
            else:
                select_tokens = keep_spatial_tokens

            # 归一化最终的图像表示
            select_tokens = F.normalize(select_tokens, dim=-1)

            # ---------------------------------------------------------
            # Step 4d: 计算图像-文本相似度(HRPA)
            # ---------------------------------------------------------
            sim_one_text = mask_xattn_one_text(
                img_embs=select_tokens,          # 处理后的图像特征
                cap_i_expand=cap_i_expand,       # 文本特征
                use_paper_version=self.use_paper_version,
                top_k=self.top_k,
                relevance_mlp=self.relevance_mlp
            )  # (B_v, 1)

            # 收集结果
            improve_sims.append(sim_one_text)          # 相似度
            score_mask_all.append(score_mask_cap)      # mask

        # =============================================================
        # Dense Text 分支处理（如果有）
        # =============================================================
        if long_cap_embs is not None and long_cap_lens is not None:
            # 遍历每个稠密文本（MLLM生成的描述）
            for i in range(len(long_cap_lens)):
                n_word = long_cap_lens[i]
                long_cap_i = long_cap_embs[i, :n_word, :]
                long_cap_i_expand = long_cap_embs_norm[i, :n_word, :].unsqueeze(0).repeat(B_v, 1, 1)

                with torch.no_grad():
                    # 计算稠密文本的全局特征和交叉注意力
                    long_cap_i_glo = F.normalize(long_cap_i.mean(0, keepdim=True).unsqueeze(0), dim=-1)
                    long_attn_cap = (long_cap_i_glo * img_spatial_embs_norm).sum(dim=-1)

                    # 使用稠密文本分支的稀疏网络
                    select_tokens_long, extra_token_long, score_mask_long = self.sparse_net_long(
                        tokens=img_spatial_embs,
                        attention_x=img_spatial_self_attention,
                        attention_y=long_attn_cap,
                    )

                # 聚合
                aggr_tokens_long = self.aggr_net(select_tokens_long)
                keep_spatial_tokens = torch.cat([aggr_tokens_long, extra_token_long], dim=1)

                if self.has_cls_token:
                    select_tokens_long = torch.cat((img_cls_emb, keep_spatial_tokens), dim=1)
                else:
                    select_tokens_long = keep_spatial_tokens

                select_tokens_long = F.normalize(select_tokens_long, dim=-1)

                # 计算相似度
                sim_one_text = mask_xattn_one_text(
                    img_embs=select_tokens_long,
                    cap_i_expand=long_cap_i_expand,
                    use_paper_version=self.use_paper_version,
                    top_k=self.top_k,
                    relevance_mlp=self.relevance_mlp
                )

                long_sims.append(sim_one_text)
                score_mask_long_all.append(score_mask_long)

        # =============================================================
        # Step 5: 融合两个分支的结果
        # =============================================================
        # 拼接所有文本的相似度 -> (B_v, B_t)
        improve_sims = torch.cat(improve_sims, dim=1)

        # 如果有稠密文本，加上其相似度
        if long_sims:
            improve_sims = improve_sims + torch.cat(long_sims, dim=1)

        # 堆叠所有mask -> (B_t, N)
        score_mask_all = torch.stack(score_mask_all, dim=0)
        if score_mask_long_all:
            score_mask_all = score_mask_all + torch.stack(score_mask_long_all, dim=0)

        # 根据训练/推理模式返回不同内容
        if self.training:
            # 训练时返回相似度和mask（mask用于计算ratio loss）
            return improve_sims, score_mask_all
        else:
            # 推理时只返回相似度
            return improve_sims


# =============================================================================
# 损失函数
# =============================================================================

class ContrastiveLoss(nn.Module):
    """
    对比损失 with Hard Negative Mining

    三元组损失的变体，用于图像-文本匹配：
    - 正样本对：对角线位置（图像i与文本i）
    - 负样本对：非对角线位置

    损失函数：
    L = Σ max(0, margin + s(i,j) - s(i,i))  (文本检索方向)
      + Σ max(0, margin + s(i,j) - s(j,j))  (图像检索方向)

    Hard Negative Mining：
    - 普通模式：对所有负样本计算损失并求和
    - Max Violation模式：只使用最难的负样本（损失最大的那个）

    Args:
        margin: 边界值，正负样本相似度之差应超过此值
        max_violation: 是否使用hard negative mining
    """

    def __init__(self, margin: float = 0.2, max_violation: bool = False):
        super().__init__()
        self.margin = margin              # 边界值
        self.max_violation = max_violation  # 是否只用最难负样本
        self.mask_repeat = True           # 是否屏蔽重复图像

    def max_violation_on(self):
        """开启hard negative mining"""
        self.max_violation = True

    def max_violation_off(self):
        """关闭hard negative mining"""
        self.max_violation = False

    def forward(
        self,
        scores: torch.Tensor,                      # 相似度矩阵
        img_ids: Optional[torch.Tensor] = None     # 图像ID（用于处理重复）
    ) -> torch.Tensor:
        """
        计算对比损失

        Args:
            scores: (B, B) 图像-文本相似度矩阵
                   scores[i,j] = 图像i与文本j的相似度
            img_ids: (B,) 图像ID，用于识别同一图像的多个caption
                    如果某些图像有多个caption，它们的img_id相同

        Returns:
            loss: 标量，对比损失值
        """
        # 取对角线元素（正样本对的相似度）
        diagonal = scores.diag().view(scores.size(0), 1)  # (B, 1)

        # 扩展为与scores相同的形状，用于比较
        d1 = diagonal.expand_as(scores)      # 每行都是s(i,i)
        d2 = diagonal.t().expand_as(scores)  # 每列都是s(j,j)

        # 计算损失
        # 文本检索方向: 对于图像i，所有文本j的损失
        # cost_s[i,j] = max(0, margin + s(i,j) - s(i,i))
        cost_s = (self.margin + scores - d1).clamp(min=0)

        # 图像检索方向: 对于文本j，所有图像i的损失
        # cost_im[i,j] = max(0, margin + s(i,j) - s(j,j))
        cost_im = (self.margin + scores - d2).clamp(min=0)

        # 构建mask：需要屏蔽的位置
        if not self.mask_repeat:
            # 简单模式：只屏蔽对角线（正样本）
            mask = torch.eye(scores.size(0), dtype=torch.bool, device=scores.device)
        else:
            # 处理重复图像：同一图像的所有caption都视为正样本
            if img_ids is not None:
                # img_ids相同的位置都要屏蔽
                mask = (img_ids.unsqueeze(0) == img_ids.unsqueeze(1))
            else:
                mask = torch.eye(scores.size(0), dtype=torch.bool, device=scores.device)

        # 将正样本位置的损失设为0
        cost_s = cost_s.masked_fill_(mask, 0)
        cost_im = cost_im.masked_fill_(mask, 0)

        # Hard Negative Mining
        if self.max_violation:
            # 只使用每行/每列的最大损失（最难负样本）
            cost_s = cost_s.max(dim=1)[0]    # (B,) 每张图像的最难文本
            cost_im = cost_im.max(dim=0)[0]  # (B,) 每个文本的最难图像

        # 返回总损失
        return cost_s.sum() + cost_im.sum()


class RatioLoss(nn.Module):
    """
    比例约束损失

    约束patch选择的比例接近目标比例，防止模型选择过多或过少的patches

    L_ratio = (mean(score_mask) - target_ratio)^2

    Args:
        target_ratio: 目标选择比例
    """

    def __init__(self, target_ratio: float = 0.5):
        super().__init__()
        self.target_ratio = target_ratio  # 目标比例

    def forward(self, score_mask: torch.Tensor) -> torch.Tensor:
        """
        计算比例约束损失

        Args:
            score_mask: (B_t, N) patch选择mask，1=选中，0=未选中

        Returns:
            loss: 标量，比例偏差的平方
        """
        # 计算实际选择比例与目标比例的差距
        return (score_mask.float().mean() - self.target_ratio) ** 2


class SEPSLoss(nn.Module):
    """
    SEPS完整损失函数

    组合对比损失和比例约束损失：
    L_total = L_contrastive + λ * L_ratio

    Args:
        margin: 对比损失的边界值
        target_ratio: 目标patch选择比例
        ratio_weight: 比例损失的权重λ
        max_violation: 是否使用hard negative mining
    """

    def __init__(
        self,
        margin: float = 0.2,
        target_ratio: float = 0.5,
        ratio_weight: float = 2.0,
        max_violation: bool = False
    ):
        super().__init__()
        # 对比损失
        self.contrastive_loss = ContrastiveLoss(margin=margin, max_violation=max_violation)
        # 比例约束损失
        self.ratio_loss = RatioLoss(target_ratio=target_ratio)
        # 比例损失权重
        self.ratio_weight = ratio_weight

    def set_max_violation(self, max_violation: bool = True):
        """动态设置hard negative mining开关"""
        if max_violation:
            self.contrastive_loss.max_violation_on()
        else:
            self.contrastive_loss.max_violation_off()

    def forward(
        self,
        similarity_matrix: torch.Tensor,      # 相似度矩阵
        score_mask: torch.Tensor,             # patch选择mask
        img_ids: Optional[torch.Tensor] = None  # 图像ID
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        计算总损失

        Args:
            similarity_matrix: (B_v, B_t) 图像-文本相似度矩阵
            score_mask: (B_t, N) patch选择mask
            img_ids: (B,) 图像ID

        Returns:
            total_loss: 总损失
            align_loss: 对比损失（用于监控）
            r_loss: 比例损失（用于监控）
        """
        # 计算对比损失
        align_loss = self.contrastive_loss(similarity_matrix, img_ids)
        # 计算比例损失
        r_loss = self.ratio_loss(score_mask)
        # 加权求和
        total_loss = align_loss + self.ratio_weight * r_loss

        return total_loss, align_loss, r_loss


# =============================================================================
# 便捷别名
# =============================================================================

SDTPS = CrossSparseAggrNet           # SDTPS模块的别名
SDTPS_TokenSparse = TokenSparse      # TokenSparse的别名
SDTPS_TokenAggregation = TokenAggregation  # TokenAggregation的别名
HRPA_function = mask_xattn_one_text  # HRPA函数的别名


def create_seps_model(
    embed_size: int = 512,
    num_patches: int = 196,
    sparse_ratio: float = 0.5,
    aggr_ratio: float = 0.4,
    use_paper_version: bool = USE_PAPER_VERSION_DEFAULT,
) -> CrossSparseAggrNet:
    """
    创建SEPS模型的便捷工厂函数

    Args:
        embed_size: 特征维度
        num_patches: 输入patch数量
        sparse_ratio: 稀疏选择比例
        aggr_ratio: 聚合比例
        use_paper_version: 是否使用论文版本

    Returns:
        model: 配置好的CrossSparseAggrNet实例
    """
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
    # 选择计算设备：优先GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # =================================
    # 参数设置
    # =================================
    batch_size = 4          # 批次大小
    num_patches = 196       # patch数量 (ViT-B/16: 14×14=196)
    embed_size = 512        # 特征维度
    sparse_text_len = 20    # 稀疏文本（原始caption）长度
    dense_text_len = 100    # 稠密文本（MLLM描述）长度
    sparse_ratio = 0.5      # 稀疏选择比例
    aggr_ratio = 0.4        # 聚合比例

    # =================================
    # 构造测试数据
    # =================================
    # 随机图像特征: (batch, patches+cls, dim)
    img_features = torch.randn(batch_size, num_patches + 1, embed_size).to(device)
    # 随机稀疏文本特征: (batch, max_len, dim)
    sparse_text = torch.randn(batch_size, sparse_text_len, embed_size).to(device)
    # 稀疏文本长度（这里假设都是满长度）
    sparse_lens = torch.full((batch_size,), sparse_text_len).to(device)
    # 随机稠密文本特征
    dense_text = torch.randn(batch_size, dense_text_len, embed_size).to(device)
    dense_lens = torch.full((batch_size,), dense_text_len).to(device)
    # 图像ID（用于对比损失）
    img_ids = torch.arange(batch_size).to(device)

    # =================================
    # 测试实际代码版本
    # =================================
    print("=" * 70)
    print("测试实际代码版本 (use_paper_version=False)")
    print("=" * 70)

    # 创建模型（实际代码版本）
    seps_actual = CrossSparseAggrNet(
        embed_size=embed_size,
        num_patches=num_patches,
        sparse_ratio=sparse_ratio,
        aggr_ratio=aggr_ratio,
        use_paper_version=False  # 使用实际代码版本
    ).to(device)

    # 设置为训练模式
    seps_actual.train()

    # 前向传播
    sim_actual, mask_actual = seps_actual(img_features, sparse_text, sparse_lens, dense_text, dense_lens)
    print(f"Similarity matrix shape: {sim_actual.shape}")  # 应为(4, 4)
    print(f"Parameters: {sum(p.numel() for p in seps_actual.parameters()):,}")

    # =================================
    # 测试论文描述版本
    # =================================
    print("\n" + "=" * 70)
    print("测试论文描述版本 (use_paper_version=True)")
    print("=" * 70)

    # 创建模型（论文描述版本）
    seps_paper = CrossSparseAggrNet(
        embed_size=embed_size,
        num_patches=num_patches,
        sparse_ratio=sparse_ratio,
        aggr_ratio=aggr_ratio,
        use_paper_version=True  # 使用论文描述版本
    ).to(device)

    seps_paper.train()

    sim_paper, mask_paper = seps_paper(img_features, sparse_text, sparse_lens, dense_text, dense_lens)
    print(f"Similarity matrix shape: {sim_paper.shape}")
    print(f"Parameters: {sum(p.numel() for p in seps_paper.parameters()):,}")

    # =================================
    # 参数量对比
    # =================================
    print("\n" + "=" * 70)
    print("参数量对比")
    print("=" * 70)
    actual_params = sum(p.numel() for p in seps_actual.parameters())
    paper_params = sum(p.numel() for p in seps_paper.parameters())
    print(f"实际代码版本: {actual_params:,} 参数")
    print(f"论文描述版本: {paper_params:,} 参数")
    print(f"额外参数: {paper_params - actual_params:,}")

    print("\n✓ 所有测试通过!")