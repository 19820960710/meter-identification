# 队友部署手册：从拉取到 X5 验证

本手册对应 **meter_labeler_v3**，不是此前泰山派的RKNN模型。当前没有已经验收的X5 `.bin`，也没有板端读数/摄像头性能结论。

## 1. 拉取并核验交付文件

```bash
git clone https://github.com/19820960710/meter-identification.git
cd meter-identification
python3 tools/verify_bundle.py
mkdir -p outputs
python3 tools/probe_rdk.py > outputs/rdk-environment.json
```

将板卡系统、Python、`hbm_runtime`可用性记入结果。官方 `rdk_x5` 分支指南推荐 RDK OS≥3.5.0，使用 `hbm_runtime`；旧镜像的接口不应直接按新分支代码假定兼容。先检查实际环境，再选匹配的工具链/运行库。

## 2. 可先做轻量 CPU 离线核对

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python tools/run_onnx.py --image samples/images/763.jpg --output outputs/cpu-detect
python tools/run_onnx.py --image samples/images/0.jpg \
  --profile configs/gauges/photo_fae0688c1fa78bd7.json --output outputs/cpu-read
python tools/validate_models.py
```

这条路径使用ONNX Runtime CPU，适合先核对文件、坐标和数值。ARM上的安装与性能仍须实测，不把电脑运行速度外推到板端。`run_onnx.py`无profile时只输出检测；错误SHA的profile在推理前拒绝。

## 3. 在 x86 Linux 开发机准备编译环境

模型编译在x86 Linux主机进行，板卡执行已编译模型。官方当前提供pip工具链和OpenExplorer Docker两种路线；以下按官方指南组织。若团队已有与板端运行库配套的工具链，优先使用已有版本。

```bash
# x86 Linux；官方pip路线推荐Ubuntu22.04、Python3.10
python3 -m venv .venv-x5
source .venv-x5/bin/activate
pip install rdkx5-yolo-mapper
hb_mapper --version
```

Docker路线见官方说明，CPU工具链镜像同样可执行转换，不要求开发机有GPU。Windows电脑可使用团队已有x86 Linux虚拟机；本交付没有验收任意WSL/Docker宿主组合。

## 4. 固定官方 Model Zoo 版本

在本仓库根目录操作，后续开发机和板端都使用同一commit：

```bash
export METER_REPO="$(pwd)"
git clone https://github.com/D-Robotics/rdk_model_zoo.git outputs/rdk_model_zoo
git -C outputs/rdk_model_zoo checkout 4f378d74fcf3507ba1336a14f5bf66bfecab2990
```

`third_party/rdk_x5_export_monkey_patch.py` 来自该commit，未修改，Apache2.0版权声明和LICENSE一并保留。X5格式浮点ONNX已提供，正常可以直接进入下一步。

若要重导出，在安装了 `requirements-full.txt` 的训练/导出环境中另存权重再导出，避免覆盖电脑基线：

```bash
mkdir -p outputs/re-export
cp models/annotation/meter_labeler_v3/best.pt outputs/re-export/meter_labeler_v3_raw_heads.pt
python third_party/rdk_x5_export_monkey_patch.py \
  --pt outputs/re-export/meter_labeler_v3_raw_heads.pt
```

不要用普通单输出 `best.onnx` 冒充官方六输出格式。导出后查看 `models/rdk_x5/export_info.json` 的输出顺序/形状；ONNX检查通过也不等于BPU编译通过。

## 5. 量化和编译

已提供32张训练来源的量化统计图片，清单包含源图SHA。这里的“校准”是模型量化统计，**不用点击表盘刻度**。这些图片还不代表将来的真实相机域，取得实际现场画面后需要复核代表性。

在上述x86工具链环境中，从本仓库根目录设置路径：

```bash
export METER_REPO="$(pwd)"
cd "$METER_REPO/outputs/rdk_model_zoo/samples/vision/ultralytics_yolo/conversion"
python3 mapper.py \
  --onnx "$METER_REPO/models/rdk_x5/meter_labeler_v3_raw_heads.onnx" \
  --cal-images "$METER_REPO/data/quantization/images" \
  --output-dir "$METER_REPO/outputs/rdk-build"
```

记录工具链版本、算子检查/编译日志、实际生成BIN文件及SHA。再按安装版本支持的命令检查模型信息和性能：

```bash
hb_model_info /absolute/path/to/generated.bin
hrt_model_exec model_info --model_file /absolute/path/to/generated.bin
hrt_model_exec perf --model_file /absolute/path/to/generated.bin --thread_num 1
```

这里的BIN路径需换成编译实际输出路径。本交付未执行这些命令，不能预先保证所有算子均上BPU、转换零误差或具体耗时。编译失败时保留完整日志，先检查工具链与接口版本及算子报告。

## 6. 板端先跑单张图

将生成的BIN复制到板端；板端也拉取本仓库和上面固定commit的Model Zoo。使用官方运行例，类别数量必须是3：

```bash
export METER_REPO="$(pwd)"
export METER_BIN=/absolute/path/to/generated.bin
cd "$METER_REPO/outputs/rdk_model_zoo/samples/vision/ultralytics_yolo/runtime/python"
python3 main.py --task detect \
  --model-path "$METER_BIN" \
  --test-img "$METER_REPO/samples/images/763.jpg" \
  --label-file "$METER_REPO/deployment/rdk_x5/classes.names" \
  --classes-num 3 --score-thres 0.25 --nms-thres 0.45 --resize-type 1 \
  --img-save-path "$METER_REPO/outputs/rdk-detection.jpg"
```

官方例子此步骤只做检测和叠图，不自动接通本项目的数值后处理。量化BIN输出可能包含编译器融合/筛选，使用对应官方runtime解析，不直接传给电脑的 `[1,7,8400]` 解码函数。

## 7. 验收和后续接线

- 核对16张相同原图：类别顺序、检测数、同类框坐标/IoU和分数；逐图保留JSON与叠图，不能只看一张“有框”。
- 对已有标定的同一原图，将BPU解码结果转换成 `label/score/box`（原图坐标）后接入 `tools/run_onnx.py` 的 `read_geometry`。保持 `src/meter_reader.py` 和 `src/calibration_binding.py` 的拒绝行为，再对照电脑数值。该BPU连接步骤尚未板端集成验收。
- 先验收同图一致性，再接摄像头采集。新帧SHA变化时，禁止直接复用单图像素刻度；需要通过模板迁移核验。当前10号图仍需复核。
- 分别记录采集FPS、推理更新时间、CPU几何耗时和完整读数耗时。预览流畅不能代表连续读数正确或低延迟。
- 上交结果至少包括：`rdk-environment.json`、工具链/runtime版本、BIN SHA、编译日志、逐图PC/BPU对照、实际耗时。板端工作完成后再更新 `deployment_status.json`。

## 官方依据

- [本交付锁定版本的转换说明](https://github.com/D-Robotics/rdk_model_zoo/blob/4f378d74fcf3507ba1336a14f5bf66bfecab2990/samples/vision/ultralytics_yolo/conversion/README_cn.md)
- [本交付锁定版本的板端入口和参数](https://github.com/D-Robotics/rdk_model_zoo/blob/4f378d74fcf3507ba1336a14f5bf66bfecab2990/samples/vision/ultralytics_yolo/runtime/python/main.py)
- [X5 Model Zoo接口指南](https://d-robotics.github.io/rdk_doc/en/rdk_s/Algorithm_Application/model_zoo/rdk_x5_guide/)
