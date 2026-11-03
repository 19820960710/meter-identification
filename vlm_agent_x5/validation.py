"""Reject unsupported output claims and keep immutable YOLO numeric fields."""
import json

def validate_model(raw):
    if raw is None:raise ValueError('未从ROS输出取得结构化VLM回答')
    obj=json.loads(raw)
    if not isinstance(obj,dict):raise ValueError('VLM回答不是JSON对象')
    for key in ('仪表类型','类型依据','可核对性','表盘状态'):
        if not isinstance(obj.get(key),str) or not obj[key].strip():raise ValueError('VLM缺少字段：'+key)
    if any(k in obj for k in ('读数','单位','正常范围','安全隐患')):
        raise ValueError('VLM试图写入YOLO数值或越界字段')
    return {k:obj[k].strip() for k in ('仪表类型','类型依据','可核对性','表盘状态')}

def assemble(yolo,vlm):
    raw=yolo['raw'];safe=yolo['reading_status']=='confirmed_candidate'
    return {'仪表类型':{'名称':vlm.get('仪表类型','无法核对'),'图像依据':vlm.get('类型依据','VLM未返回有效观察')},
            'YOLO读数和单位':{'原始读数':raw.get('value'),'原始单位':raw.get('unit'),
                '状态':'可供图像核对' if safe else '待复核','原因':None if safe else
                ('目标表盘无法唯一绑定' if yolo['binding']!='single_panel' else '缺少当前图片适用的刻度标定或有效读数')},
            '读数可核对性':vlm.get('可核对性','VLM未返回有效观察；无法核对'),
            '表盘状态':vlm.get('表盘状态','VLM未返回有效观察；无法核对')}

def validate_scene(raw):
    if raw is None:raise ValueError('ROS输出中没有场景JSON')
    obj=json.loads(raw)
    keys=('现场观察','安全隐患','场景合理性','建议','判断边界')
    if not isinstance(obj,dict) or set(obj)!=set(keys):raise ValueError('场景回答字段不完整或有额外字段')
    for key in keys:
        if not isinstance(obj[key],str) or not obj[key].strip() or len(obj[key])>120:
            raise ValueError('场景字段为空或过长：'+key)
    forbidden=('绝对安全','完全安全','确认泄漏','确认带电','立即停机','关闭阀门')
    if any(x in ' '.join(obj.values()) for x in forbidden):raise ValueError('场景回答包含无依据的安全或操作结论')
    return {key:obj[key].strip() for key in keys}
