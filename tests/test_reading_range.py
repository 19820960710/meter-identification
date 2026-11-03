from pathlib import Path
import sys,copy,unittest
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tools'))
from reading_range import assess
SHA='a'*64
def rules():return {'schema_version':1,'assets':{'TEST':{'point_id':'POINT','profile_sha256':SHA,'quantity':'pressure','unit':'bar','lower':1,'upper':2,'lower_inclusive':True,'upper_inclusive':True,'enabled':True,'rule_source':'synthetic unit test only','scope':'demo','operating_mode':None}}}
def evaluate(value=1.5,unit='bar',status='readable',config=None,**kwargs):
    args=dict(asset_id='TEST',point_id='POINT',identity_confirmed=True,profile_sha256=SHA,operating_mode=None);args.update(kwargs)
    return assess({'status':status,'value':value,'unit':unit,'physical_accuracy_verified':False},rules() if config is None else config,**args)
class RangeTests(unittest.TestCase):
    def test_within(self):self.assertEqual(evaluate()['status'],'within_range')
    def test_below(self):self.assertEqual(evaluate(.9)['status'],'below_range')
    def test_above(self):self.assertEqual(evaluate(2.1)['status'],'above_range')
    def test_closed_bounds(self):
        for x in [1,2]:self.assertEqual(evaluate(x)['status'],'within_range')
    def test_open_bounds(self):
        r=rules();r['assets']['TEST'].update(lower_inclusive=False,upper_inclusive=False)
        self.assertEqual(evaluate(1,config=r)['status'],'below_range');self.assertEqual(evaluate(2,config=r)['status'],'above_range')
    def test_unit_conversion(self):
        for x,u in [(150,'kPa'),(.15,'MPa'),(150000,'Pa')]:self.assertEqual(evaluate(x,u)['converted_value'],1.5)
    def test_decimal_boundary(self):self.assertEqual(evaluate(.1,'MPa')['status'],'within_range')
    def test_failure_even_numeric_present(self):self.assertEqual(evaluate(status='needs_review')['reason_code'],'reading_unavailable')
    def test_missing_numeric(self):self.assertEqual(evaluate(None)['status'],'unable_to_determine')
    def test_nonfinite_and_bool(self):
        for x in [True,float('nan'),float('inf'),float('-inf'),'1e9999']:self.assertEqual(evaluate(x)['status'],'unable_to_determine')
    def test_conversion_overflow(self):self.assertEqual(evaluate('1e308','MPa')['status'],'unable_to_determine')
    def test_no_rule(self):self.assertEqual(evaluate(config={'schema_version':1,'assets':{}})['reason_code'],'rule_missing')
    def test_no_identity(self):self.assertEqual(evaluate(identity_confirmed=False)['reason_code'],'identity_unconfirmed')
    def test_wrong_point(self):self.assertEqual(evaluate(point_id='OTHER')['reason_code'],'point_mismatch')
    def test_wrong_profile(self):self.assertEqual(evaluate(profile_sha256='b'*64)['reason_code'],'profile_mismatch')
    def test_disabled(self):
        r=rules();r['assets']['TEST']['enabled']=False;self.assertEqual(evaluate(config=r)['reason_code'],'rule_disabled')
    def test_bad_rule(self):
        for field,value in [('lower',3),('upper',float('inf')),('lower_inclusive','yes'),('scope','unknown'),('rule_source','')]:
            r=rules();r['assets']['TEST'][field]=value;self.assertEqual(evaluate(config=r)['reason_code'],'invalid_rule')
    def test_empty_interval(self):
        r=rules();r['assets']['TEST'].update(lower=1,upper=1,upper_inclusive=False);self.assertEqual(evaluate(config=r)['reason_code'],'invalid_rule')
    def test_incompatible_units(self):self.assertEqual(evaluate(1.5,'div')['reason_code'],'unit_mismatch')
    def test_mode_required(self):
        r=rules();r['assets']['TEST']['operating_mode']='running';self.assertEqual(evaluate(config=r)['reason_code'],'mode_mismatch');self.assertEqual(evaluate(config=r,operating_mode='running')['status'],'within_range')
    def test_outer_divisions_not_mm(self):
        r=rules();r['assets']['TEST'].update(quantity='outer_scale_divisions',unit='div');self.assertEqual(evaluate(1.5,'div',config=r)['status'],'within_range');self.assertEqual(evaluate(1.5,'mm',config=r)['status'],'unable_to_determine')
    def test_no_mutation_and_candidate_label(self):
        r=rules();before=copy.deepcopy(r);out=evaluate(config=r);self.assertEqual(r,before);self.assertEqual(out['reading_kind'],'candidate');self.assertEqual(out['equipment_condition'],'not_evaluated')
if __name__=='__main__':unittest.main()
