# 智慧教育课程推荐系统

本项目实现 `任务.md` 中的课程推荐实验代码。默认数据目录为 `课程作业数据集/`，提交代码包时不要包含原始数据集。

## 最终结果

[查看完整报告 PDF](report/course_recommendation_report.pdf)

当前报告采用 `adaptive_kg_gate_validated` / `RAKF-M-G` 作为最终模型。该模型在全课程排序、遮蔽训练正例的统一评测口径下取得：

| 指标 | 数值 |
| --- | ---: |
| HR@10 | 0.5871 |
| HR@20 | 0.6995 |
| NDCG@10 | 0.3747 |
| NDCG@20 | 0.4053 |

核心思路是把 EASE、ItemKNN、RP3beta、Popularity 和课程侧 KG profile 作为多源排序证据，再按用户训练历史长度进行可靠性自适应融合，并用训练内验证门控决定 KG profile 是否参与最终排序。最终系数来自训练集内部 leave-one-out 验证用户，不直接用测试集选系数。

## 作业要求对照

| 要求 | 本项目对应内容 |
| --- | --- |
| 四个指标 | `HR@10`、`HR@20`、`NDCG@10`、`NDCG@20` 写入报告表 4 和最终截图 |
| 知识图谱辅助 | 使用 `course-concept`、`concept-field`、`parent-son`、`teacher-course`、`school-course`、`prerequisite-dependency` 构造课程画像 |
| LightGCN/推荐模型 | 实现并汇报 `MF-BPR`、`LightGCN-BPR`、`ItemKNN`、`EASE`、`RP3beta` 等对照 |
| 消融实验 | 报告表 5、图 7 给出 w/o KG candidate、non-gated KG mix、w/o RP3、w/o EASE、w/o adaptive grouping |
| 运行过程截图 | `report/figures/real_data_check.png`、`real_final_eval.png`、`real_gpu_environment.png` |
| 中文信息学报格式 | `report/course_recommendation_report.tex` 和最终 PDF 按双栏期刊格式排版 |
| 复现与防泄漏 | README 中列出防泄漏规则，融合系数从训练集内部验证选择 |

## 已实现内容

- 全排序评测：对每个测试用户在全部课程上排序，mask 掉训练集中已学课程。
- 指标：`HR@10`、`HR@20`、`NDCG@10`、`NDCG@20`，并额外输出 `Recall@10/20` 便于检查多正例口径。
- 基线：`MostPop`、`ItemKNN`、`EASE`、`RP3beta`、`MF-BPR`、`LightGCN-BPR`。
- KG 特征：基于课程侧 `course-concept`、`concept-field`、`parent-son`、`teacher-course`、`school-course`、`prerequisite-dependency` 构造课程画像。
- 融合：`EASE + KG profile + popularity` 分数融合。
- 当前主结果：`adaptive_kg_gate_validated` / `RAKF-M-G`，在 `EASE(reg=100)`、`ItemKNN`、`RP3beta(alpha=0.9,beta=0.2,topk=300)`、Popularity 和 KG profile 之上加入按用户历史长度划分的分组线性融合系数，并通过验证门控关闭不稳定的 KG 排序贡献。融合系数在训练集内部验证用户上选择，最终 test 只用于汇报。
- 稳定性检查：在三个随机种子下重复训练内验证抽样，结果保存在 `results/stability_light/validation_seed_summary.csv`，用于说明分组融合不是单次抽样偶然结果。
- PyTorch 训练脚本：`MF-BPR`、`LightGCN-BPR`，本地没有 PyTorch 时可在服务器运行。

## 本地快速运行

```powershell
python scripts/check_data.py --data-dir "课程作业数据集"

$env:PYTHONPATH="src"
python -m course_rec.run_experiment --config configs/popularity.yaml --data-dir "课程作业数据集"
python -m course_rec.run_experiment --config configs/ease.yaml --data-dir "课程作业数据集"
python -m course_rec.run_experiment --config configs/kg_profile.yaml --data-dir "课程作业数据集"
python -m course_rec.run_experiment --config configs/fusion.yaml --data-dir "课程作业数据集"
python -m course_rec.run_experiment --config configs/rp3beta_best.yaml --data-dir "课程作业数据集"
python -m course_rec.run_experiment --config configs/fusion_best_rp3.yaml --data-dir "课程作业数据集"
python -m course_rec.run_experiment --config configs/fusion_validated.yaml --data-dir "课程作业数据集"
python -m course_rec.run_experiment --config configs/adaptive_fusion_validated.yaml --data-dir "课程作业数据集"
python -m course_rec.run_experiment --config configs/adaptive_kg_gate_validated.yaml --data-dir "课程作业数据集"

python scripts/summarize_results.py --outputs outputs --out results/summary.csv
python scripts/make_paper_figures.py
```

调参不要反复看 `test.csv`。先从训练集内部切 validation：

```powershell
python scripts/tune_validation.py --data-dir "课程作业数据集" --out results/validation_tuning.csv
```

训练内验证稳定性检查：

```powershell
$env:PYTHONPATH="src"
python scripts/search_adaptive_fusion_validation.py --data-dir "课程作业数据集" --seed 20260604 --max-valid-users 15000 --skip-test --out results/stability_light/adaptive_validation_seed20260604.csv --fixed-out results/stability_light/fixed_validation_seed20260604.csv
```

## GPU 训练

服务器上建议先安装 PyTorch GPU 版本，再运行：

```bash
export PYTHONPATH=src
python -m course_rec.train_torch --config configs/mf_bpr.yaml --data-dir "课程作业数据集"
python -m course_rec.train_torch --config configs/lightgcn.yaml --data-dir "课程作业数据集"
```

如果通过反向代理隧道连接服务器，SSH 和传输都走同一个端口：

```bash
ssh root@223.2.31.209 -p 54478
rsync -avP -e "ssh -p 54478" ./ root@223.2.31.209:/home/user/course_rec/
```

长任务必须放进 `tmux` 或 `screen`，防止隧道断开影响训练进程。

## 防泄漏规则

- 不能直接使用 `MOOCCube/relations/user-course.json` 作为训练边，因为它包含测试交互。
- 不使用完整 `entities/user.json` 中的 `course_order` 或未来选课时间。
- 默认不使用全量 `user-video.json` 或 `user_video_act.json`，避免由视频行为推出测试课程。
- 调参应从训练集内部切 validation；最终 test 只做汇报。

## 输出文件

每次实验会生成：

- `outputs/<run_name>/config.yaml`
- `outputs/<run_name>/metrics.json`
- `outputs/<run_name>/topk_sample.csv`
- `outputs/<run_name>/run.log`

报告写作时从 `results/summary_with_gpu.csv`、`reports/tables/segment_model_comparison.csv` 和各实验输出中整理主实验表、分组对比表、消融表和可视化图。运行过程证据保存在 `report/figures/real_*.png` 与 `reports/run_logs/`。

当前报告主结果在 `outputs/adaptive_kg_gate_validated/metrics.json` 和 `results/summary.csv`：

- `adaptive_kg_gate_validated` / `RAKF-M-G`
- `HR@10=0.5871`
- `HR@20=0.6995`
- `NDCG@10=0.3747`
- `NDCG@20=0.4053`

最终报告文件：

- `report/course_recommendation_report.pdf`
- `report/course_recommendation_report.tex`

提交代码包时应排除 `课程作业数据集/` 原始数据目录。
