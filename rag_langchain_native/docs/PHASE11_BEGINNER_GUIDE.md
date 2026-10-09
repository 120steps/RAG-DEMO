# Phase 11 初学者指南

推荐依次阅读：

1. `config.py::Settings.validate_security()`：生产缺 Secret 为什么必须拒绝启动。
2. `security.py::AuthService.verify_token()`：Authentication（你是谁）与 Authorization（你能看什么）。
3. `security.py::build_authorization_scope()`：权限为何必须在 Retrieval 前生效。
4. `lifecycle.py::validate_pdf_content()`：扩展名为何不能证明文件是 PDF。
5. `api.py::security_middleware()`：大小、Rate Limit、Concurrency、安全头。
6. `chain.py::format_context()`：文档是“不可信数据”而非指令。
7. `backup.py::BackupService`：一致性、Manifest 与哈希验证。

**Least Privilege** 是只授予最小权限；空授权集合必须返回空结果，不能退回全库。**Prompt Injection** 是不可信文本诱导模型越过规则，Prompt 分隔只是纵深防御，ACL 才是机密边界。**Docker Image** 是程序模板，**Container** 是运行实例，**Volume** 保存容器重建后仍需保留的数据。**Reverse Proxy** 在客户端与 FastAPI 之间承担 HTTPS、域名和网关限制。

不要误解：测试中恶意文本没越过 ACL，不等于真实 Gemini 对全部攻击免疫；Dockerfile 存在不等于本机已构建成功；开发 Token 不等于生产身份平台。
