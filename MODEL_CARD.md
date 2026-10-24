# meter_labeler_v3 模型说明

交接日期：2026-10-24。目标：RDK X5 初步部署验证。

## 模型和类别

- 框架：Ultralytics 8.4.175，YOLO11n Detect；训练原模型2,605,369参数，导出融合后2,597,537参数。
- 训练输入：640；导出固定 batch1、RGB、NCHW、640×640；float32 输入除以255。
- 原 `.pt` 为部署时剥离的checkpoint，`epoch=-1`，不据此认定未训练。checkpoint和原CSV共同保留训练证据。
- 3类顺序固定：`0 panel`、`1 needle`、`2 center_region`。
- `panel`：表盘区域框。`needle`：针身及配重附近的搜索框，可能截断细端。`center_region`：轴心搜索框，仍可能误检商标；不等于真实轴心坐标。
- 本模型不是分割模型或关键点模型，不输出物理针尖、示数、量程或单位。

## 原始训练信息

在先前的三类 `meter_labeler_v2` 初始化上训练，100轮，batch4，imgsz640，seed0，device0。完整参数见 [training/args.yaml](training/args.yaml)。原始运行环境见 [training/environment-reference.json](training/environment-reference.json)；其中 CUDA 版 PyTorch 是原电脑参考环境，X5部署不要求照抄该CUDA环境。

训练导出集255张：248训练、7开发验证；853框。数据来源和重复分组见 [训练集索引](training/dataset_manifest.json)。7张已被多轮开发观察，不能当新的独立测试集。

`best.pt` 元数据记录：

| 指标 | 数值 |
|---|---:|
| 检测 Precision | 0.89616 |
| 检测 Recall | 0.90000 |
| 检测 mAP50 | 0.91731 |
| 检测 mAP50–95 | 0.79926 |

原CSV最高mAP50–95出现在第1轮，与该checkpoint指标一致；最后一轮mAP50–95为0.78469。不要把最后一轮指标写成best权重指标。上述均为框检测指标，**不能写成91.7%仪表读数准确率**。

## 电脑 ONNX 协议

`models/annotation/meter_labeler_v3/best.onnx`：opset17、IR8，输入 `images [1,3,640,640]`，输出 `output0 [1,7,8400]`。

每列为 `cx,cy,w,h,panel_score,needle_score,center_region_score`；得分已在图内转为概率，无额外objectness；图内没有NMS。

默认处理：保持比例缩放并补114灰边；恢复框时使用实际整数resize的横纵比例；每个位置取最大类别分数；conf0.25、分类别NMS IoU0.45、最多300框。`src/onnx_detector.py` 的六头解码只用于**浮点ONNX CPU对照**，不能直接解析编译后的量化BIN张量。

## X5 浮点导出协议

`models/rdk_x5/meter_labeler_v3_raw_heads.onnx` 使用锁定的官方导出器导出，opset11、IR6；权重没有重训。

输入仍为RGB/NCHW float32。六输出按顺序是：

| stride | 分类logits NHWC | 框DFL logits NHWC |
|---|---|---|
| 8 | `[1,80,80,3]` | `[1,80,80,64]` |
| 16 | `[1,40,40,3]` | `[1,40,40,64]` |
| 32 | `[1,20,20,3]` | `[1,20,20,64]` |

分类需Sigmoid，框为4×16的DFL。编译后输入协议按官方配置转换为NV12，输出节点可能被编译器改写；板端使用同一官方版本的解码器和 `--classes-num 3`。详见 [export_info.json](models/rdk_x5/export_info.json) 和部署手册。

## 数值读数、来源和局限

数值由 `src/meter_reader.py` 的图像几何完成：轴帽Hough圆、近轴长边、可见针身与竞争方向检查，再用 `atan2` 跨零处理和相邻人工刻度分段插值。显示小数不代表测量精度。

16张原图的电脑协议核验、11张历史ONNX读数复现、763模板四目标核对均为开发证据。没有实物新观测，不提供现场准确率；没有X5运行，不提供板端FPS或延迟。没有读取计圈针或设备限值，`operating_status=not_evaluated`。

主权重SHA256：`40b1c77a30bcc6c5588e001408094ecbd57aa9de50a549d45e741cf75dc6431b`。

原电脑ONNX SHA256：`d53d4e58b388ebd4f0687dda6454339c773e30eca2b2205e88e83bc0bfbb74e3`。

全部交付文件哈希见 [BUNDLE_SHA256.json](BUNDLE_SHA256.json)。
