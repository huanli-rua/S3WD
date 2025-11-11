# S3WD v02 · Reference Tuple 主线说明

本说明文档概述 v02 版本在原 S3WD-GWB 流程上的升级要点：

- **参考元组 (Reference Tuple)**：基于分桶样本，分别维护 Γ_pos/Ψ_neg 参考集合，结合 GWB 置信度挑选典型样本。
- **混合相似度**：使用数值 RBF（径向基函数）与类别匹配权重合成样本与参考元组的相关度。
- **批级小网格阈值搜索**：以期望成本或 Fβ 为目标，在护栏约束下搜索 (α, β)，并通过 EMA 平滑 + 步长限幅稳定阈值。
- **漂移闭环**：对接 PSI/TV/PosRate/性能四类指标，触发 S1/S2/S3 分级响应（调 σ、重建参考元组、重建 GWB 索引与收紧护栏）。
- **兼容旧主线**：通过 YAML 中的 `SWITCH.enable_ref_tuple / enable_pso` 可切换至 v02 主线或原 PSO 主线。

## 快速上手

1. 准备 `configs/s3wd_airline_v02.yaml`（示例配置已包含 v02 所需键位，可在此基础上调整路径与参数）。
2. 运行 `notebooks/02_s3wd_gwb.ipynb`，Notebook 会依次完成数据加载、参考元组构建、流式主循环、漂移响应、可视化与 CSV 导出。
3. 输出文件默认存放于 `DATA.data_dir` 指定目录，包括：
   - `threshold_trace_v02.csv`
   - `window_metrics.csv`
   - `drift_events.csv`
   - `yearly_metrics.csv`
   - `yearly_*.png`

## 模块一览

| 模块 | 核心接口职责 |
| --- | --- |
| `s3wdlib/bucketizer.py` | `configure()` 注入分桶规则，`assign_buckets()` 输出候选桶，并在样本稀缺时回退到更粗粒度 |
| `s3wdlib/ref_tuple.py` | `build_ref_tuples()` 构建 Γ_pos/Ψ_neg 参考集合，`combine_history()` 用于季节桶合并 |
| `s3wdlib/similarity.py` | `configure()`/`current_config()` 维护混合相似度参数，`corr_to_set()` 计算样本与参考集的相关度 |
| `s3wdlib/batch_measure.py` | `to_trisect_probs()` 生成三域概率，`compute_region_masks()` 等提供批级指标评估 |
| `s3wdlib/threshold_selector.py` | `select_alpha_beta()` 在护栏约束下搜索 (α, β)，支持网格缩放 |
| `s3wdlib/smoothing.py` | `ema_clip()` 对阈值进行 EMA 平滑并限制步长 |
| `s3wdlib/drift_controller.py` | `detect_drift()` 判级，`apply_actions()` 执行 S1/S2/S3 动作（调 σ / 重建 ΓΨ / 紧护栏 / 重建 GWB） |

## 结果复现

运行 `notebooks/02_s3wd_gwb.ipynb` 后会生成以下文件，均保存在 `DATA.data_dir` 指定目录：

- `threshold_trace_v02.csv`：按月记录 α/β 网格解与平滑值。
- `window_metrics.csv`：月度评估明细，含九项分类指标、BND 占比、POS 覆盖率等。
- `drift_events.csv`：漂移告警日志。
- `yearly_metrics.csv`：按样本数加权的年度汇总。
- `yearly_*.png`：九张年度折线图（Precision/Recall/F1/BAC/MCC/Kappa/AUC/BND_ratio/POS_coverage）。

Notebook 顶部会打印 warmup/stream 窗口序列，强调“历史同月仅用于构建参考库，不混入当月评估”。尾部单元会展示 CSV 导出路径，并渲染月度轨迹图与 3×3 年度折线图矩阵，便于快速核对指标走势。

## 配置默认值一览（缺键安全）

下表列出 `config_loader.py` 中为 v02 注入的默认键，未显式配置时会自动补齐，避免 KeyError：

| 键 | 含义 | 默认值 |
| --- | --- | --- |
| `BUCKET.keys` | 初始分桶字段 | `['UniqueCarrier', 'Origin', 'Dest', 'dep_hour']` |
| `BUCKET.min_bucket` | 每桶最小样本数 | `500` |
| `BUCKET.backoff` | 分桶回退链 | `[[UniqueCarrier, Origin, Dest], [Origin, Dest], [Origin]]` |
| `REF_TUPLE.topk_per_class` | 每类保留参考元组数 | `256` |
| `REF_TUPLE.pos_quantile` | 正类置信度分位数门槛 | `0.7` |
| `REF_TUPLE.keep_history_ratio` | 历史同月保留比例 | `0.3` |
| `REF_TUPLE.use_gwb_weight` | 是否采用 GWB 权重 | `True` |
| `SIMILARITY.kernel` | 数值相似度核函数 | `rbf` |
| `SIMILARITY.sigma` | 数值核宽度 | `0.5` |
| `SIMILARITY.cat_weights` | 类别特征权重 | `{carrier:0.4, origin:0.2, dest:0.2, dow:0.1, month:0.1}` |
| `SIMILARITY.combine` | 数值与类别融合方式 | `product` |
| `SIMILARITY.mix_alpha` | 数值占比 | `0.7` |
| `MEASURE.objective` | 阈值目标函数 | `expected_cost` |
| `MEASURE.costs` | 成本系数 (FN/FP/BND) | `{c_fn:1.0, c_fp:0.4, c_bnd:0.2}` |
| `MEASURE.grid` | α/β 网格范围 | `{alpha:[0.55,0.9,0.02], beta:[0.05,0.45,0.02]}` |
| `MEASURE.constraints` | 护栏约束 | `{keep_gap:0.05, min_pos_coverage:0.02, bnd_cap:0.35}` |
| `SMOOTH.ema_alpha` | 阈值 EMA 系数 | `0.6` |
| `SMOOTH.step_cap` | 阈值步长限幅 | `{alpha:0.08, beta:0.08}` |
| `DRIFT.method` | 漂移检测算法 | `kswin` |
| `DRIFT.window_size` | 滑窗大小 | `512` |
| `DRIFT.stat_size` | 统计窗口 | `128` |
| `DRIFT.psi_thresholds` | PSI 告警阈值 | `{warn:0.1, alert:0.25}` |
| `DRIFT.tv_thresholds` | TV 告警阈值 | `{warn:0.1, alert:0.2}` |
| `DRIFT.posrate_shift` | 正例率漂移阈值 | `{warn:0.05, alert:0.10}` |
| `DRIFT.perf_drop` | 目标下降阈值 | `{warn:0.10, alert:0.20}` |
| `DRIFT.debounce_windows` | 去抖窗口数 | `3` |
| `DRIFT.actions.S1.sigma_factor` | S1 调 σ 因子 | `[1.2, 0.85]` |
| `DRIFT.actions.S2.keep_history_ratio` | S2 保留比例 | `0.3` |
| `DRIFT.actions.S3.tighten` | S3 护栏收紧 | `{keep_gap:0.08, bnd_cap:0.25, step_cap:0.05}` |
| `SWITCH.enable_ref_tuple` | 启用 Reference Tuple 主线 | `True` |
| `SWITCH.enable_pso` | 是否启用 PSO 主线 | `False` |
| `FLOW.warmup_windows` | warmup 月数 | `6` |
| `FLOW.recent_windows` | 邻近月补充数量 | `3` |
| `FLOW.time_decay` | 时间衰减系数 | `0.15` |
| `FLOW.seasonal_neighbor` | 相邻月份权重 | `0.7` |
| `FLOW.seasonal_other` | 其他月份权重 | `0.4` |
| `FLOW.weight_max` | 历史权重上限 | `2.5` |
| `FLOW.history_cap` | 同月历史窗口上限 | `12` |
| `TIME.split` | 时间窗口拆分方式 | `year_month` |
| `VAL.inline_delay` | 验证是否延迟标注 | `True` |

更多细节请参阅 Notebook 中的中文注释与日志。
