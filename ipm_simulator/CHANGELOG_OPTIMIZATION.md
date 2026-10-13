# IPM 性能优化改动记录

## 背景与目标
- **初始状态**：`IPM::GenerateIPMImage` 逐像素遍历并对所有满足几何条件的相机像素做累加，平均开销约 60 ms/帧；标签量化消耗 60–70 ms/帧；帧率抖动明显。
- **优化目标**：通过算法与实现层的迭代，将 IPM 合成阶段稳定压缩到 10–15 ms，并在后续进一步压缩到 1–2 ms 常态（偶发 8–10 ms 尖峰），同时保持功能正确。

## 阶段一：热点识别 & 基础优化
- 使用 `std::chrono` 打印合成和量化耗时，定位瓶颈集中在 `GenerateIPMImage` 的嵌套循环及 `cv::Mat::at` 函数调用。
- 引入 `Eigen::initParallel()`、OpenMP 并行加速主循环。
- 自 `src/ipm_fun.cc` 开始缓存 `T_cam_vehicle`、排序权重等数据，避免重复计算。
- 在 `ipm_node.cpp` 中对标签图添加并行量化（OpenMP）以降低 70+ ms 的尾部开销。

## 阶段二：查表驱动的投影重写
- 增加 `PixelLookupEntry` 与 `EnsureLookupTable()`/`BuildLookupTable()`：
  - 生成阶段一次性预计算 IPM 像素到各相机的投影关系，保留权重最高的候选。
  - 运行时查表仅做内存访问与常量次乘加，实现大幅提速。
- `GenerateIPMImageImpl` 调整为：
  - 预缓存每个相机图像的指针和步长。
  - 用指针访问代替 `cv::Mat::at`，去掉边界检查开销。
  - 使用 `schedule(static,64)`/`256` 等固定调度策略，减少线程负载波动。

## 阶段三：标签管线重构
- 在后续尝试中探索“索引图 + LUT 着色”的优化，但未在最终版本保留；保留的是回退后的 `quantizeLabelImage` 并行量化方案。

## 阶段四：稳定性增强与进一步压缩
- 自 2024-xx 起的重点改动：
  - **权重预归一化**：在 LUT 构建时仅保留权重最大的两台相机，计算 `inv_weight_sum`，运行时不再累计 `weight_sum`。
  - **新结构体**：`PixelContribution` 存储 `camera_idx + offset + weight`，`PixelLookupEntry` 仅保留 `std::array<PixelContribution, 2>`、`float inv_weight_sum` 和 `uint8_t count`。
  - **运行时循环**：
    - 直接用 `inv_weight_sum` 归一化，避免除法和分支；
    - 使用固定 `schedule(static,256)`，确保线程负载均衡；
    - 只访问预存偏移 `offset`，不再逐帧计算 `v * stride + u`。
  - **线程设置**：
    - 构造函数中 `omp_set_dynamic(0)` 禁止 OpenMP 自调线程；
    - `omp_set_num_threads(min(std::thread::hardware_concurrency(), 8))` 避免过度切换。
  - **日志策略**：`ipm_node` 中保留 20 帧预热后再打印日志，减少启动噪声；始终在 IPM 延迟基础上计算 FPS 并叠加到画面。

## 已测试行为与性能
- 在回退前的日志显示 IPM generation 常态 1–2 ms，偶发 8–10 ms；Quantization 维持 3–4 ms；Total 5–14 ms.
- 回退后的版本仍保持上述表现，为后续进一步抑制尖峰打好基础。

## 未来工作（可选）
- 针对偶发尖峰，可尝试 tile-based 预取、CPU 亲和等更激进手段，但需谨慎验证以免性能倒退。
- 可视硬件添加 SIMD（SSE/NEON）实现 `count==2` 的双通道累加。

