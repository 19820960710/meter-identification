"""Candidate-value range assessment, separate from equipment health and vision."""
from decimal import Decimal,InvalidOperation
import hashlib,json,re,math
from pathlib import Path

UNITS={'Pa':('pressure',Decimal('1')),'kPa':('pressure',Decimal('1000')),'MPa':('pressure',Decimal('1000000')),'bar':('pressure',Decimal('100000')),'div':('outer_scale_divisions',Decimal('1')),'mm':('displacement',Decimal('.001')),'cm':('displacement',Decimal('.01')),'m':('displacement',Decimal('1'))}
REASONS={'reading_unavailable':'识读未通过，无法判断范围。','invalid_reading':'读数缺失或不是有限数值。','rule_missing':'未配置该仪表的允许范围。','invalid_registry':'范围配置文件格式无效。','invalid_rule':'仪表范围规则不完整或无效。','rule_disabled':'该仪表范围规则尚未启用。','identity_unconfirmed':'尚未确认仪表身份和巡检点。','point_mismatch':'巡检点与仪表配置不匹配。','profile_mismatch':'参考标定与该仪表配置不匹配。','mode_mismatch':'当前工况未提供或与规则不匹配。','unit_mismatch':'读数单位与规则量纲不兼容。','rule_file_unavailable':'范围配置无法读取。'}

def number(value):
    if isinstance(value,bool) or not isinstance(value,(int,float,str,Decimal)):raise ValueError('Not a number')
    try:d=Decimal(str(value))
    except InvalidOperation:raise ValueError('Not a number')
    if not d.is_finite() or not math.isfinite(float(d)):raise ValueError('Not finite')
    return d

def unavailable(reason,asset_id=None):
    return {'status':'unable_to_determine','reason_code':reason,'reason':REASONS[reason],'asset_id':asset_id,'reading_kind':'candidate','equipment_condition':'not_evaluated','converted_value':None,'rule':None}

def assess(reading,registry,*,asset_id=None,point_id=None,identity_confirmed=False,profile_sha256=None,operating_mode=None):
    def fail(reason):return unavailable(reason,asset_id)
    if reading.get('status')!='readable':return fail('reading_unavailable')
    try:value=number(reading.get('value'))
    except ValueError:return fail('invalid_reading')
    if not isinstance(registry,dict) or type(registry.get('schema_version')) is not int or registry['schema_version']!=1 or not isinstance(registry.get('assets'),dict):return fail('invalid_registry')
    if not isinstance(asset_id,str) or not asset_id or asset_id not in registry['assets']:return fail('rule_missing')
    rule=registry['assets'][asset_id]
    try:
        if not isinstance(rule,dict):raise ValueError()
        for key in ['point_id','rule_source']:
            if not isinstance(rule.get(key),str) or not rule[key].strip():raise ValueError()
        if not isinstance(rule.get('profile_sha256'),str) or not re.fullmatch('[0-9a-f]{64}',rule['profile_sha256']):raise ValueError()
        if rule.get('scope') not in ['demo','device_spec']:raise ValueError()
        if any(type(rule.get(key)) is not bool for key in ['enabled','lower_inclusive','upper_inclusive']):raise ValueError()
        lo,hi=number(rule.get('lower')),number(rule.get('upper'))
        if lo>hi or (lo==hi and not (rule['lower_inclusive'] and rule['upper_inclusive'])):raise ValueError()
        if rule.get('unit') not in UNITS or UNITS[rule['unit']][0]!=rule.get('quantity'):raise ValueError()
        mode=rule.get('operating_mode')
        if mode is not None and (not isinstance(mode,str) or not mode.strip()):raise ValueError()
    except (ValueError,TypeError,KeyError):return fail('invalid_rule')
    if not rule['enabled']:return fail('rule_disabled')
    if identity_confirmed is not True:return fail('identity_unconfirmed')
    if point_id!=rule['point_id']:return fail('point_mismatch')
    if profile_sha256!=rule['profile_sha256']:return fail('profile_mismatch')
    if mode is not None and operating_mode!=mode:return fail('mode_mismatch')
    unit=reading.get('unit')
    if not isinstance(unit,str) or unit not in UNITS or UNITS[unit][0]!=rule['quantity']:return fail('unit_mismatch')
    converted=value*UNITS[unit][1]/UNITS[rule['unit']][1]
    if not math.isfinite(float(converted)):return fail('invalid_reading')
    status='within_range'
    if converted<lo or (converted==lo and not rule['lower_inclusive']):status='below_range'
    elif converted>hi or (converted==hi and not rule['upper_inclusive']):status='above_range'
    output=unavailable('rule_missing',asset_id)
    output.update(status=status,reason_code=None,reason=None,point_id=point_id,identity_basis='explicit_operator_confirmation',original_value=float(value),original_unit=unit,converted_value=float(converted),converted_value_decimal=str(converted),unit=rule['unit'],operating_mode=operating_mode,rule={k:rule.get(k) for k in ['quantity','unit','lower','upper','lower_inclusive','upper_inclusive','rule_source','scope','operating_mode','profile_sha256']})
    return output

def attach(reading,config_path,**identity):
    """Attach only assessment fields, leaving vision values/geometry untouched."""
    try:
        raw=Path(config_path).read_bytes();registry=json.loads(raw.decode('utf-8-sig'))
        assessment=assess(reading,registry,**identity);assessment['rules_sha256']=hashlib.sha256(raw).hexdigest()
    except (OSError,UnicodeError,ValueError):assessment=unavailable('rule_file_unavailable',identity.get('asset_id'))
    result=dict(reading);result.update(operating_status=assessment['status'],range_assessment=assessment,equipment_condition='not_evaluated');return result

def display_lines(assessment):
    if not assessment:return ['范围判断：无法判断（缺少评估结果）。']
    if assessment['status']=='unable_to_determine':return ['候选范围判断：无法判断',assessment['reason']]
    rule=assessment['rule'];left='[' if rule['lower_inclusive'] else '(';right=']' if rule['upper_inclusive'] else ')'
    labels={'within_range':'范围内','below_range':'低于允许下限','above_range':'高于允许上限'}
    prefix='演示规则 · ' if rule['scope']=='demo' else ''
    return [prefix+'候选范围判断：'+labels[assessment['status']],f"仪表 {assessment['asset_id']}；允许范围 {left}{rule['lower']}, {rule['upper']}{right} {rule['unit']}", '范围判断不等于设备整体运行状态判断。']
