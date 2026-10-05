import json
from pathlib import Path
import unittest
import numpy as np

from tools.association.stable_fusion import HistoryDataset, soft_associate, fuse_size, process, prepare_objects
from tools.association.evaluate_stable import association_counts, geometry, bootstrap_geometry


def obj(x=0., length=4., cls=0, yaw=0.):
    box=np.array([x,0.,0.,length,2.,2.,yaw])
    return {'box':box,'raw_box':box.copy(),'spatial_box':box.copy(),'class':cls,
            'stable_lwh':box[3:6].copy(),'raw_lwh':box[3:6].copy(),
            'size_stable_valid':True,'size_relative_mad':np.zeros(3),'motion_valid':True}


class StableFusionTest(unittest.TestCase):
    def test_history_median_ignores_future_other_scene_and_current_outlier(self):
        d=HistoryDataset.__new__(HistoryDataset)
        d.frames={'vehicle-side':{'current':{'sequence_id':'s','pointcloud_timestamp':'2000000'}}}
        d.side_timelines={('vehicle-side','s'):[(1600000,'p1'),(1800000,'p2'),(2000000,'current'),(2100000,'future')],
                          ('vehicle-side','wrong'):[(1900000,'wrong')]}
        visited=[]
        def rows(side,frame):
            visited.append(frame)
            return [{'track_id':'1','type':'Car','3d_location':dict(x=0,y=0,z=0),
                     '3d_dimensions':dict(l=40 if frame=='current' else 4,w=2,h=2),'rotation':0}]
        d.objects=rows;o={**obj(length=40),'track_id':'1'}
        s=d.dimensions('vehicle-side','current',o)
        self.assertTrue(s['size_stable_valid']);np.testing.assert_array_equal(s['stable_lwh'],[4,2,2])
        self.assertNotIn('future',visited);self.assertNotIn('wrong',visited)
        self.assertEqual(s['size_history_count'],3)

    def test_non_vehicle_preserves_pose_dependent_dimensions(self):
        d=HistoryDataset.__new__(HistoryDataset)
        d.frames={'vehicle-side':{'f':{'sequence_id':'s','pointcloud_timestamp':1}}}
        self.assertFalse(d.dimensions('vehicle-side','f',obj(cls=1))['size_stable_valid'])

    def test_zero_iou_soft_candidate_and_dummy(self):
        v=[obj(length=.2)];r=[obj(.4,length=.2)]
        candidates,accepted,_=soft_associate(v,r,.1)
        self.assertEqual(candidates,[(0,0)]);self.assertEqual(accepted,[(0,0)])
        self.assertEqual(soft_associate(v,[obj(10.)],.1)[0],[])
        # Expensive but within distance gate: dummy is cheaper.
        self.assertEqual(soft_associate(v,[obj(1.9,length=.2)],.1)[0],[])

    def test_axis_flip_dense_ambiguity_and_invalid_motion(self):
        self.assertEqual(soft_associate([obj()],[obj(.1,yaw=np.pi)],.1)[1],[(0,0)])
        self.assertEqual(soft_associate([obj()],[obj(.1),obj(-.1)],.1)[1],[])
        invalid={**obj(.1),'motion_valid':False}
        self.assertEqual(soft_associate([obj()],[invalid],.1)[1],[])
        self.assertEqual(soft_associate([obj()],[obj(.1)],1.1)[1],[])

    def test_size_blending_conflict_and_insufficient_history(self):
        v,r=obj(),obj(length=4.4)
        box,info=fuse_size(v,r,True);self.assertEqual(info['size_policy'],'blend_stable');self.assertAlmostEqual(box[3],4.2)
        for r in [obj(length=8),{**obj(length=4.4),'size_stable_valid':False}]:
            box,info=fuse_size(v,r,True);self.assertEqual(info['size_policy'],'select_ego');self.assertEqual(box[3],4)

    def test_same_pairing_and_unmatched_objects_survive(self):
        v=[obj()];r=[obj(.1),obj(10)]
        a,b=process(v,r,.1,'A3'),process(v,r,.1,'A4')
        self.assertEqual(a['accepted'],b['accepted']);self.assertEqual(len(a['outputs']),2)
        self.assertEqual(a['outputs'][0]['source_bits'],[1,1]);self.assertEqual(a['outputs'][1]['source_bits'],[0,1])

    def test_unknown_is_not_false_positive(self):
        c=association_counts([(0,0),(1,1),(2,2)],[(0,0),(1,3)])
        self.assertEqual(c['true_positive'],1);self.assertEqual(c['known_false_positive'],1);self.assertEqual(c['unknown_predictions'],1)

    def test_algorithm_runs_without_cooperative_labels(self):
        root=Path('/home/zzn/V2X_VLM/UniV2X/datasets/V2X-Seq-SPD-New')
        if not root.exists():self.skipTest('Native dataset unavailable')
        d=HistoryDataset(root)
        pair=json.loads((root/'cooperative/data_info.json').read_text())[10]
        # Guard every actual file read: cooperative object labels are forbidden.
        import builtins
        from unittest.mock import patch
        original=builtins.open;original_read=Path.read_text
        def guarded_open(file,*args,**kwargs):
            self.assertNotIn('/cooperative/label/',str(file));return original(file,*args,**kwargs)
        def guarded_read(path,*args,**kwargs):
            self.assertNotIn('/cooperative/label/',str(path));return original_read(path,*args,**kwargs)
        with patch('builtins.open',guarded_open),patch.object(Path,'read_text',guarded_read):
            v,r,selected,age=prepare_objects(d,pair)
            for variant in ['A0','A1','A2','A3','A4']:process(v,r,age,variant)


if __name__=='__main__':unittest.main()
