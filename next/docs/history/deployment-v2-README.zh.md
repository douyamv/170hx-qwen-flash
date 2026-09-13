# Qwen3.8 Flash-Next 262K 启动说明

服务器：`douya@192.168.3.130`。启动程序和参数保存在 `/home/douya/qwen-3.8-next/`，模型权重仍在机械盘 `/mnt/slowdisk/AI-archive/models/Qwen3.8-Flash-Next-GGUF/`。

## 启动与查看状态

登录服务器后运行：

```bash
~/start_qwen_3.8_next.sh
```

默认流程：校验程序、配置、四张指定显卡和模型分片大小；保护正在生成的请求；停止冲突的推理服务；必要时通过 cmpunlocker 热加载恢复 Gen2；使用真实模型元数据预检主模型与 MTP 的 262K 初始化；通过后才读取大权重并启动 8080 代理；等待并验证实际模型名称及 262144 上下文。脚本不包含整机重启或关机操作。

```bash
~/start_qwen_3.8_next.sh --status
~/start_qwen_3.8_next.sh --check
~/start_qwen_3.8_next.sh --force
~/start_qwen_3.8_next.sh --no-wait
```

- `--status` 只查询，不停止服务。
- `--check` 只进行启动前检查。
- `--force` 允许中断正在生成的请求后重新启动；仍受启动互斥锁保护。
- `--no-wait` 提交启动后返回，不代表模型已经就绪。
- 同时重复运行启动脚本会被拒绝，避免重启加载到一半的模型。
- 默认等待模型就绪最多 3600 秒，可通过 `--timeout 秒数` 调整。超时会明确报错并保留服务，不会自动重启。

机械盘冷加载约需 30 分钟，具体取决于盘速与缓存。前台启动时每 20 秒输出累计读盘量；`--status` 能区分服务进程存在与 API 真正就绪。

## 四卡分配

按 UUID 选择设备，避免重启或插卡后 GPU 数字编号变化：

| 逻辑设备 | PCIe 地址（部署时） | UUID | 用途 |
|---|---|---|---|
| CUDA0 | 02:00.0 | GPU-524355f6-cdea-1376-4724-94c1d248c7d8 | 主模型第 0–15 层 |
| CUDA1 | 01:00.0 | GPU-3ba5a344-6eac-6688-805c-c2adcbaa93c3 | 主模型第 16–31 层 |
| CUDA2 | 82:00.0 | GPU-fbc3ea3b-e64d-a2cd-3b2e-47b3ed0aa5c2 | 主模型第 32–47 层、输出及 MTP |
| CUDA3 | 81:00.0 | GPU-bb191baf-c42d-d155-3ce1-f9b757be98f3 | 约 27 GiB 的 PLE 大嵌入表，不分配主模型层 |

主模型：UD-Q4_K_XL；总上下文 262144；单请求槽；主模型 KV 为 Q8_0，MTP KV 为 F16；默认 MTP 最大草稿长度 3。配置文件为 `launch.json`。

约 644 MiB 的共享输入词嵌入保留在主机内存，供主模型与仅使用 CUDA2 的 MTP 调度器共同访问。不能把它强制放到 MTP 调度器未包含的 CUDA0，否则会在初始化时报错。`--override-tensor` 只传一次；如需多项，使用同一参数内的逗号分隔语法。

81:00.0 当前是 Gen2 x2；另三张是 Gen2 x8/x16/x16。该卡通道宽度问题没有被宣称修复，采用 PLE 分配来减少其每步传输量。

## API 与日志

- 应用使用的 API：`http://192.168.3.130:8080/v1`
- 模型名称：`Qwen3.8-Flash-Next-UD-Q4_K_XL`
- 后端监听本机 `127.0.0.1:8093`。
- 旧的 `Qwen3.8-27B-FP8` 及 child32k 路由别名由代理映射到当前 Next 模型，以兼容已有 Harness 任务。
- 主服务：`qwen38-flashnext-optimized-262k.service`
- 代理服务：`qwen38-lb.service`
- 启动日志：`/home/douya/qwen-3.8-next/logs/start-日期时间.log`
- 模型日志：`/home/douya/qwen-3.8-next/logs/server.log`
- 代理日志：`/home/douya/qwen-3.8-next/logs/proxy.log`

```bash
tail -f ~/qwen-3.8-next/logs/server.log
curl --noproxy '*' http://127.0.0.1:8080/health
curl --noproxy '*' http://127.0.0.1:8080/v1/models
```

主线程会在模型加载完成后固定到 CPU 0，已创建的工作线程保留原 CPU 集合。

## Gen2 与后台服务

独立热加载工具：`sudo /usr/local/sbin/cmp170hx-gen2-hotload`；只读检查可加 `--check`。启动脚本使用运行包内同一版本的工具。该工具仅对已验证的 610.57.04 cmpunlocker 安装执行模块热加载，并开启 persistence；不写显卡固件、不触发 PCIe 总线复位、不重启整机。日志位于 `/var/log/cmp170hx-hotload/`。

Vast.ai 后台按用户要求长期停用：相关服务被屏蔽，自动重启与更新的 cron 入口已移除，备份位于 `/root/vast-disabled-20260913-121825/`。旧的 80 GiB 开机检查与失效的早期 Gen2 重训脚本已移出活动路径，备份位于 `/root/cmp-cleanup-20260913-121106/`。

当前有效 NVIDIA 模块仍是 `/lib/modules/7.0.0-30-generic/updates/cmpunlocker/` 下配套的 610.57.04；没有删除正在使用的 Linux 内核。

## 校验范围

启动时对运行程序和参数做 SHA-256 校验，对模型分片检查存在性及大小；不在每次启动时重新通读 104 GiB 权重做全量哈希。模型原始下载的哈希验证证据保存在此前测试目录中。

`bin/next-init-check` 使用无权重分配模式预检完整模型结构和 MTP。该检查曾在约 5 秒内复现错误的跨 GPU 共享配置，修正配置约 6 秒通过。还使用了约 21 MiB 实际磁盘占用的稀疏零填充 GGUF，验证真实尺寸显存分配、262K 初始化以及 8 和 64 token 的生成与 MTP 草稿验证流程。这些零填充结果只用于运行检查，不用于质量、接受率或性能评估。

本说明不把进程启动、API 就绪、满上下文速度与长期稳定性混为同一个验证结论。实际部署测量结果另见完成后的结果报告。
