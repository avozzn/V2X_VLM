import unittest
import numpy as np
def obj(x=0., length=4.):
    box=np.array([x,0.,0.,length,2.,2.,0.])
    return {'box':box,'raw_box':box.copy(),'spatial_box':box.copy(),'class':0,
            'stable_lwh':box[3:6].copy(),'raw_lwh':box[3:6].copy(),
            'size_stable_valid':True,'size_relative_mad':np.zeros(3),'motion_valid':True}

from tools.association.relaxed_fusion import associate_relaxed, process_relaxed

class RelaxedTest(unittest.TestCase):
    def test_full_distance_gate_keeps_clear_pair(self):
        self.assertEqual(associate_relaxed([obj()],[obj(1.2)],.05)[1],[(0,0)])
        self.assertEqual(associate_relaxed([obj()],[obj(2.1)],.05)[1],[])
    def test_short_age_unknown_motion_needs_geometry(self):
        r={**obj(.1),'motion_valid':False}
        self.assertEqual(associate_relaxed([obj()],[r],.05)[1],[])
        self.assertEqual(associate_relaxed([obj()],[r],.05,True)[1],[(0,0)])
        self.assertEqual(associate_relaxed([obj()],[r],.10001,True)[1],[])
        r={**obj(.4,length=.2),'motion_valid':False}
        self.assertEqual(associate_relaxed([obj(length=.2)],[r],.05,True)[1],[])
    def test_age_zero_and_missing_age(self):
        r={**obj(.1),'motion_valid':False}
        self.assertEqual(associate_relaxed([obj()],[r],0,True)[1],[(0,0)])
        self.assertEqual(associate_relaxed([obj()],[r],None,True)[1],[])
    def test_ambiguity_not_relaxed(self):
        self.assertEqual(associate_relaxed([obj()],[obj(.1),obj(-.1)],.05,True)[1],[])
    def test_raw_ego_dimensions_and_unmatched_preservation(self):
        v=obj();v['stable_lwh']=np.array([8,3,3])
        result=process_relaxed([v],[obj(.1),obj(10)],.05,'L2')
        self.assertEqual(len(result['outputs']),2)
        np.testing.assert_array_equal(result['outputs'][0]['box'],v['box'])
        self.assertEqual(result['outputs'][0]['source_bits'],[1,1])
    def test_empty_and_invalid_variant(self):
        self.assertEqual(associate_relaxed([],[],0,True)[1],[])
        with self.assertRaises(ValueError):process_relaxed([],[],0,'invalid')

if __name__=='__main__':unittest.main()
