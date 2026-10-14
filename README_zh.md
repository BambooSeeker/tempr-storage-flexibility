# 研究代码，1.2.0

按照 README.md 创建 Python 3.12 环境并安装 requirements-inference.txt。
`python infer_figures.py` 使用原始 LoRA 适配器，实际生成两周预测；
`python demo_market.py` 随后求解一日预算选择。其他模型曲线来自对应日期的存档预测。

1.2.0 已按目标结束时刻清理分区交界窗口，训练标签单独拟合归一化器。
五个受影响的对照模型重新训练，相同 80 个测试日保持不变。
完整数据持有者可运行 forecast/run_corrected_baselines.py。

公开样例范围保持为 14 日和首日所需的 1024 点历史，完整市场数据库另行授权。
全部市场机制与结算代码保留，主市场实验仍使用原先冻结的 LoRA 预测。
使用范围、许可证、复现入口与验证结果分别见 DATA_AVAILABILITY.md、THIRD_PARTY_NOTICES.md、
docs/REPRODUCING.md 和 VALIDATION.md。源代码仓库：https://github.com/BambooSeeker/tempr-storage-flexibility。版本存档见仓库的 v1.2.0 release。
