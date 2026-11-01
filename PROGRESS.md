# PROGRESS

## 最后更新
- 日期: 2026-10-23
- 当前状态: 完成方法原理梳理、训练框架确认、对比实验设计，并定位到当前训练在 Kaggle 上的显存瓶颈。
- 当前阶段: 关键问题已从“算法逻辑”转移到“GPU 显存可行性”，下一步需通过更保守的验证设置继续确认方法有效性。

## 本轮实验的核心目标
- 验证“结构先验 + 深度学习”联合去噪框架的合理性
- 检查 bitonic prior 能否为 CNN 分支提供结构感知约束
- 设计对比实验，确认结合方案是否优于纯 CNN 或无门控版本
- 在 Kaggle 环境中定位工程可运行性问题，并确认其是否为算法本身的本质缺陷

## 已完成的工作
- 已理解项目整体结构，并确认核心方法属于 hybrid denoising 思路：
  - CNN 分支负责端到端学习
  - bitonic prior 提供局部结构/强度约束
  - gate 融合决定不同区域的输出组合权重
- 已确认模型主入口和训练逻辑位于 [dl-bitonic/train.py](dl-bitonic/train.py)
- 已确认核心实现位于：
  - [dl-bitonic/src/dl_bitonic/model.py](dl-bitonic/src/dl_bitonic/model.py)
  - [dl-bitonic/src/dl_bitonic/bitonic.py](dl-bitonic/src/dl_bitonic/bitonic.py)
  - [dl-bitonic/src/dl_bitonic/dataset.py](dl-bitonic/src/dl_bitonic/dataset.py)
  - [dl-bitonic/src/dl_bitonic/losses.py](dl-bitonic/src/dl_bitonic/losses.py)
  - [dl-bitonic/src/dl_bitonic/metrics.py](dl-bitonic/src/dl_bitonic/metrics.py)
- 已搭建对比实验脚本：
  - [dl-bitonic/src/dl_bitonic/compare_denoising_experiments.py](dl-bitonic/src/dl_bitonic/compare_denoising_experiments.py)
- 当前对照组已明确包括：
  - hybrid
  - cnn_only
  - hybrid_no_gate
  - traditional baseline
- 已确认对比实验设计的意义：验证“结合”本身是否有效，而不是只验证网络容量增长带来的效果提升。

## 结合思路的理论解释
- 纯 CNN 去噪模型容易在高噪声区或复杂纹理区出现过平滑和结构丢失。
- bitonic prior 通过局部窗口内的结构关系与强度差异来为恢复提供结构感知约束。
- 通过 sigma-aware tau 和门控机制，使模型可以在不同位置动态决定更依赖先验还是依赖 CNN 输出。
- 这样的方法在原理上是合理的：它兼顾了学习能力和结构保真性。

## 当前阶段的实验观察
- 训练脚本在逻辑上可执行，训练过程可以进入 epoch，并且 loss 能正常下降。
- 问题出现于 validation 阶段，而不是训练阶段本身。
- 报错定位于 bitonic prior 的权重构造步骤，具体在 [dl-bitonic/src/dl_bitonic/bitonic.py](dl-bitonic/src/dl_bitonic/bitonic.py) 的局部窗口展开和权重广播中。
- 报错信息为 CUDA Out of Memory，说明当前 Kaggle 运行环境的 GPU 可用显存不足以支撑当前 validation 规模。

## 关键结论
- 目前没有证据表明方法思路本身错误；更可能是实现对显存要求较高。
- 该问题集中在 bitonic prior 的中间张量构造，尤其在 full validation 时会放大内存峰值。
- 跟“代码突然失效”不同，更接近于“当前硬件/运行环境条件不再支持当前实现的资源消耗”。
- 因此，下一步不是重写方法核心思想，而是控制实验规模，降低内存峰值，继续验证其可行性。

## 已确认的瓶颈
- 在当前 Kaggle 环境中，GPU 空闲显存极少，约为 386 MiB
- 在 validation 阶段，bitonic prior 需要生成较大的局部权重张量，导致 OOM
- 这说明当前实现对资源敏感，需采用更保守的 batch、patch 和 validation 设置

## 下一步计划
1. 先减小验证阶段负载：
   - batch-size 降到 1
   - patch-size 降到 32 或更小
   - 限制 validation 样本数
2. 继续观察训练是否稳定并能正常保存 checkpoint
3. 等内存稳定后，再恢复正常规模的对比实验
4. 最终证明 hybrid 方案是否在 PSNR / SSIM 上优于纯 CNN 和无门控版本

## 实验最终价值
- 这两天的工作已经完成了方法论方向、模型框架和实验设计的基本闭环。
- 当前阶段的核心进展不在于“模型已经完全跑通”，而在于“结合思路已经成型，瓶颈已定位并可针对性解决”。
- 从研究推进角度看，项目已经从“搭建代码”进入“验证有效性与工程适配”的关键阶段。

## 现阶段总结一句话
- 结合方法在理论上合理，实验框架已建立，当前关键困难是 Kaggle 显存限制导致的 bitonic validation OOM；在控制显存负载后，方法的有效性仍可继续验证。

## 接下来建议
- 暂停大规模再训练，先做极小规模 sanity run
- 继续保留当前训练入口与恢复思路
- 等环境稳定后，再做正式对比实验并准备汇报材料
