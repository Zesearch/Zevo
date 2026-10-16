# 训练中途评估与自适应早停待办

记录日期：2026-10-15。2026-10-16 按用户要求完成本地实现，尚未部署。

目标：为 32B 等训练耗时较长的任务提供中途 Validation 反馈，不必等完整 epoch 结束才决定后续方向。

## 用户已确认的方向

- Launch Run 提供是否允许中途评估的开关，建议名称为 `Adaptive mid-training evaluation`。
- 开启代表允许 agent 根据中途 Validation 结果决定评估时机和是否提前停止。
- 用户不必填写固定评估间隔、patience 等细节；agent 负责制定计划、调整时机并解释决策。

## 建议的执行流程

1. 训练前，Orchestrator 根据预计训练时长、checkpoint 保存成本和评估耗时，在 Train payload 中指定 `first_validation_fraction`。Train agent 按完整训练步数乘该比例向上取整，系统冻结并校验首次评估位置，不固定等到 epoch 边界。
2. 在安全位置保存完整且可恢复的 checkpoint，执行与 baseline 一致的 Validation inference 和 evaluation。
3. Orchestrator 根据连续评估的趋势、改善幅度、波动和剩余成本，决定继续训练、调整下次评估时机或提前停止，并记录理由。
4. 单次没有提升不应直接早停。Trainer validation loss 仅用于辅助诊断，Test 不参与评估时机、checkpoint 选择或早停决策。
5. 提前停止按正常完成记录，明确原因；保留本轮 Validation 最好的 checkpoint，供后续正式 Test 和模型保存使用。

## 建议的实现约束

- 开关默认关闭；具体默认值和界面文案在实施时再确认。
- 中途评估属于同一个 iteration，显示 checkpoint 对应的 step，不消耗额外 iteration。
- 资源受限时，在安全位置暂停训练、释放训练进程占用的显存后评估；恢复时保持 optimizer、scheduler、训练步数及必要的数据读取和随机状态。资源允许时才考虑并行评估。
- checkpoint 写入完成后才能评估，评估期间不得覆盖或清理该 checkpoint。
- 评估失败或基础设施异常不能当作没有提升，应重试或跳过。
- 控制评估开销，避免频繁保存、加载大模型反而拖慢训练。
- 平台提供可靠的保存、暂停、评估和恢复机制；agent 在用户允许的范围内选择时机和停止策略。

## 本地实现

Launch Run 的 Auto、Standard 和 Customized 路径均可设置开关，默认关闭。
新 Run 将设置记录在 lifecycle；现有 Run 不会自动开启。

系统控制器保存完整可恢复的训练状态后退出训练子进程，再按冻结的
Validation 数据、baseline predictor 和 evaluator 评分。结果通过已有的进度
事件送到 Orchestrator；它通过专用 decision API 决定下一检查 step 或提前结束。
提前结束至少需要三次成功的 Validation 检查；失败不记成负向分数。
评估在同一个 Train Ticket 和 iteration 内完成。最终发布本轮 Validation
最好的 checkpoint，后续走正常的最终 Validation 和 Test 流程。

控制器保留最佳和最近的完整 checkpoint，清理自身已提交的其他内部恢复点；
普通 Trainer、DDP、ZeRO、FSDP 恢复仍使用各框架原生状态，分片模型在发布时
通过框架导出最佳权重。等待决策时不占用训练显存，但当前集群 GPU allocation
仍被保留。等待超时进入可恢复错误，不伪造完成。

本地回归覆盖开关传递、评分、决策送达、重复事件、恢复及结果提交。
尚未进行真实 GPU 端到端运行验证。

Run 的 Train 阶段详情新增 Mid-training Validation 面板，沿用详情页的 3 秒刷新。
显示各 checkpoint 的 step、进度、各 Validation 成员分数、suite average、
Orchestrator 决策和理由，以及最佳 checkpoint 和下次评估点。没有记录时隐藏。
最终评估和提前停止也发布进度快照；后端保留验证通过的最终报告，以免日志尾部丢失导致漏显。
重复进度不会重复显示，训练中途结果不写入正式 iteration 结果表。
