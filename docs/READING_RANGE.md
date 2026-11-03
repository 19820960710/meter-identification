# 候选读数范围判断

2026-11-03新增。用户尚未确定真实允许范围，因此`configs/inspection_rules.json`的`assets`为空。默认返回`unable_to_determine`，不会把未配置或识读失败当作正常。

功能已接入`tools/run_onnx.py`命令行入口的`result.json`，使用标准库规则模块`tools/reading_range.py`；原视觉算法、模型、几何及数值计算保持。图像仍为原检测/读数叠图，范围判断看JSON字段。本次不包含本地后续bar实验链或Windows桌面UI，不能把其回归数据作为本仓库模型成绩。

## 使用

```bash
python tools/run_onnx.py --image samples/images/0.jpg \
  --profile configs/gauges/photo_fae0688c1fa78bd7.json --output outputs/0-range
```

此时可产生原约0.3776 MPa读数，但范围判断为无法判断，因为没有真实规则。

标准确定后，参考`configs/inspection_rules.example.json`填写资产ID、巡检点、参考标定文件SHA256、量纲、单位、允许上下限、开闭边界、来源、scope及可选工况。模板默认禁用、上下限为空，不会自动加载。

```bash
python tools/run_onnx.py --image samples/images/0.jpg \
  --profile configs/gauges/photo_fae0688c1fa78bd7.json \
  --rules path/to/confirmed_rules.json --asset-id YOUR_ASSET_ID \
  --point-id YOUR_POINT_ID --confirm-identity --output outputs/checked
```

以上规则路径和编号需替换为已核实配置；`--confirm-identity`表示操作者确认物理设备身份，不能仅因同型号就确认。规则有工况时还须传匹配的`--operating-mode`。参考SHA仅绑定标定，不证明设备身份。设备有不同工况时，当前支持一条规则匹配指定工况，不自动选择多套工况限值。

## 结果约定

- `operating_status`: `within_range` / `below_range` / `above_range` / `unable_to_determine`。
- `range_assessment`: 原/换算值、单位、规则来源和scope、规则SHA、身份依据及无法判断的原因。
- `equipment_condition`: 始终`not_evaluated`。范围内不表示设备整体运行正常。
- 所有结果按候选读数解释。`scope=demo`明确表示演示规则，不能用作设备安全标准。

支持Pa/kPa/MPa/bar固定比例Decimal换算；div只能用于外圈格数，不能当作总位移mm。跨量纲、非有限数/溢出、缺配置/身份/工况、错误参考或识读失败均无法判断。表盘量程不等于允许工作范围。

退出码仍为原视觉语义：0表示检测/识读完成，不能用作“范围正常”信号；报警或Agent应读取上述字段，单独处理无法判断。

直接调用`run_onnx.run()`不会自动做范围判断；其他入口可在取得读数后调用`reading_range.attach(result, rules_path, asset_id=..., point_id=..., identity_confirmed=..., profile_sha256=..., operating_mode=...)`。不要从模型置信度推断物理身份。

## 验证

```bash
python -m unittest discover -s tests -p test_reading_range.py
python tools/verify_bundle.py
```

22项规则测试涵盖边界、单位、身份、工况、缺配置及异常输入。真实设备范围仍未启用；本次没有新增BPU或相机验收。
