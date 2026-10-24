# 仪表视觉识读 · RDK X5 部署交接

当前主模型：**meter_labeler_v3 / YOLO11n / 3类检测**。本仓库提供原始权重、电脑端 ONNX、已导出的 X5 浮点 ONNX、读数代码、样例、训练记录及部署步骤。

**先读 [队友部署手册](deployment/rdk_x5/README.md)。** X5 浮点 ONNX 已通过电脑检查及同图对照；**BPU `.bin` 尚未编译，板端和摄像头尚未验收**。本仓库不承诺现场精度或帧率。

## 队友先做这几步

```bash
git clone https://github.com/19820960710/meter-identification.git
cd meter-identification
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python tools/verify_bundle.py

# 仅检测：不用 PyTorch、不用点刻度
python tools/run_onnx.py --image samples/images/763.jpg --output outputs/763-detect

# 已标定原图读数：输出约 0.3776 MPa，属于电脑 ONNX 浮点基线
python tools/run_onnx.py --image samples/images/0.jpg \
  --profile configs/gauges/photo_fae0688c1fa78bd7.json --output outputs/0-read

# 16张原图的电脑协议对照，包括 X5 浮点导出与原 ONNX 的检测对照
python tools/validate_models.py
```

Windows PowerShell 激活命令改用 `.venv\Scripts\Activate.ps1`。轻量入口使用 ONNX Runtime **CPU**，在 X5 上运行这个入口也不会自动使用 BPU。保存的 `annotated.jpg` 和 `result.json` 位于指定输出目录。

## 三份模型文件的用途

| 文件 | 用途 | 当前验证状态 |
|---|---|---|
| `models/annotation/meter_labeler_v3/best.pt` | 原始训练权重；原 PT 读数、导出或继续训练 | 原字节保持 |
| `models/annotation/meter_labeler_v3/best.onnx` | 电脑 ONNX 基线，单输出 `[1,7,8400]` | 已做原图 CPU 推理 |
| `models/rdk_x5/meter_labeler_v3_raw_heads.onnx` | 官方 X5 导出格式，6个 NHWC 原始输出，opset11 | 已检查、已做 CPU 协议对照；待量化编译 |

两份 ONNX 的输出协议不同，不能共用同一个输出解析分支。权重中没有直接的“示数”输出。模型检测表盘、指针搜索区和轴心搜索区；真实轴帽、针向与数值换算由后处理完成。

## 已实现与待完成

- 已有：三类检测；SHA 绑定的单图标定与读数；763 模板在四张已见同表型照片上的迁移候选；原图、错图拒绝、几何歧义检查。
- 待完成：X5 BPU 编译/量化差异/板端耗时、相机新帧读数、可靠跨照片刻度对应、独立实物读数验证。
- 0～0.6 MPa 的 `0 → 10` 跨照片实验仍需复核，**10 没有已接受的数值**。不要将目视约0.33当成程序结果。
- 百分表完整计圈、总位移和设备运行状态不属于本交付版能力。

单图标定绑定原图 SHA 和尺寸，不能直接用于另一张照片。只有通过模板配准与支撑核验，才能迁移；未通过就返回 `needs_review` 和空值。

## 可选：763 单模板实验

在电脑参考环境安装完整依赖后，从仓库根目录运行：

```bash
pip install -r requirements-full.txt
python experiments/template-transfer-20261024/read_template_photo.py \
  --image samples/images/832.jpg --output outputs/template-832
```

该入口只读取 763 的原标定，832 候选约 **6.31343 MPa**。四图核对记录见 [evidence/template-portability-check.json](evidence/template-portability-check.json)。这是已见训练/开发来源上的候选，不是四张独立实物测试，也不是全量程精度证明。该实验限定0～25 MPa模板，不能直接给其他量程套用。

## 资料位置

- [模型说明与输入输出](MODEL_CARD.md)
- [X5部署手册与验收清单](deployment/rdk_x5/README.md)
- [训练参数](training/args.yaml)、[100轮记录](training/results.csv)、[训练集SHA与分组](training/dataset_manifest.json)
- [16张演示图片与标定索引](samples/manifest.json)、[32张量化统计图片](data/quantization/manifest.json)
- [CPU对照结果](evidence/model-validation.json)、[代码验证记录](evidence/handoff-verification.json)
- [算法与移植说明](docs/ALGORITHM_AND_PORTABILITY.md)、[依赖及来源](THIRD_PARTY_NOTICES.md)

完整255张训练图片没有全部打包；已提供其分组、SHA和255份YOLO标签，并附训练初始化权重。队友做部署不需要先重训。量化统计图片不需要点击表盘刻度，也不能作为独立测试集。
