- kv cache
- 参数解析器
- launcher
- 调度器


### TODO：支持 KV Projection 与 KV Cache 写入融合

**背景：** 当前 MiniSGL 先通过线性投影计算 K、V，生成中间 Tensor，再调用 `store_kv()`，通过独立的 CUDA Kernel 将 KV 写入 Cache Pool。这会引入额外的 Kernel Launch 和显存读写开销。

**优化思路：**

- 探索将 KV Projection（`K = WkX, V = WvX`）与 KV Cache 写入融合到同一个 CUDA Kernel。
- 融合 Kernel 直接获取 KV Cache Pool 的存储指针及 `out_loc`，将计算结果写入目标物理槽位。
- 避免 K/V 中间 Tensor 在全局显存物化，减少额外的显存读写与 Kernel Launch。
- 保留现有 `MHAKVCache` 存储布局和 `store_kv()` 普通写入路径，融合路径直接写入 Pool。
- 后续可进一步探索 QKV Projection、Attention 与 KV Cache 写入的融合。

**关键问题：** 当前 KV Cache Pool 的存储结构基本能够支持直接写入，主要需要调整模型计算层与 Attention Backend 的接口，使融合 Kernel 能够直接访问目标 KV Cache。

**验证目标：** 对比融合前后的 Kernel Launch 次数、显存流量、推理延迟及吞吐，重点关注 Decode 阶段的收益，并验证 KV 写入和 Attention 结果的正确性。