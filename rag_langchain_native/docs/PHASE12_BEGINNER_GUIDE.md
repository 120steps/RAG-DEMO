# Phase 12 初学者指南

- **CI**：Push/PR 后自动检查；**CD**：把已验证版本交付或部署。本项目不会自动部署生产。
- **Workflow**：一个 YAML 自动化；**Job** 是可并行任务；**Step** 是 Action 或命令。
- **Artifact**：一次运行留下的报告。本项目不上传数据库、PDF、Prompt 或 Secret。
- **Quality Gate**：用阈值判定质量；退出码 0/1/2 表示通过/退化/无法评价。
- **Regression Test**：修改后重复证明旧能力没有倒退。
- **Baseline**：明确数据、配置、样本数的历史结果；不同机器延迟不可直接公平比较。

先读 `eval/quality_gate.py::evaluate_gate()`，再读 `.github/workflows/`。普通 PR 只做离线确定性测试；真实 Gemini Evaluation 必须手动触发且使用受保护 Secret。Dependabot 只创建升级 PR，不自动合并。
