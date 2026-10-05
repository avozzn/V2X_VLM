import unittest
import numpy as np
from tools.association.core import extrapolate, transform_box, pair_features, match
from tools.association.run import resolve_reference, NativeDataset


class NativeAssociationTest(unittest.TestCase):
    def test_reference_identity_conflicts_and_missing(self):
        class Dataset:
            frames = {'vehicle-side':{'v':{'sequence_id':'a','pointcloud_timestamp':'100'}},
                      'infrastructure-side':{'i':{'sequence_id':'b','pointcloud_timestamp':'90'}}}
            def objects(self, side, frame):
                return self.rows[side]
        d = Dataset()
        def row(track, token):
            return {'track_id':track,'token':token,'type':'Car','3d_location':dict(x=0,y=0,z=0),
                    '3d_dimensions':dict(l=4,w=2,h=2),'rotation':0}
        d.rows={'vehicle-side':[row('1','vt')],'infrastructure-side':[row('2','it')]}
        pair={'vehicle_frame':'v','infrastructure_frame':'i','vehicle_sequence':'a','infrastructure_sequence':'b'}
        obj={'veh_frame_id':'v','inf_frame_id':'i','veh_track_id':'1','inf_track_id':'2',
             'veh_token':'vt','inf_token':'it','veh_pointcloud_timestamp':'100','inf_pointcloud_timestamp':'90'}
        self.assertEqual(resolve_reference(d,pair,obj)['status'],'valid')
        self.assertEqual(resolve_reference(d,pair,{**obj,'veh_token':'bad'})['status'],'unknown')
        self.assertEqual(resolve_reference(d,pair,{**obj,'inf_track_id':'-1'})['status'],'unknown')
        d.rows['vehicle-side'].append(row('1','other'))
        self.assertEqual(resolve_reference(d,pair,obj)['status'],'unknown')

    def test_history_is_past_and_same_sequence(self):
        d = NativeDataset.__new__(NativeDataset)
        d.frames={'infrastructure-side':{'current':{'sequence_id':'s','pointcloud_timestamp':'2000000'}}}
        d.timelines={'s':[(1000000,'past'),(2000000,'current'),(2100000,'future')],
                     'other':[(1999999,'wrong_scene')]}
        visited=[]
        def objects(side, frame):
            visited.append(frame); return [{'track_id':'1','type':'Car'}]
        d.objects=objects
        previous, _, status = d.latest_history('current',{'track_id':'1','type':'Car'})
        self.assertEqual((previous,status),('past','valid'))
        self.assertEqual(visited,['past'])

    def test_stationary_world_under_moving_ego(self):
        box=np.array([10.,0.,0.,4.,2.,2.,0.])
        result, velocity, valid=extrapolate(box,box.copy(),.5,1.)
        ego=np.eye(4); ego[0,3]=3.
        self.assertTrue(valid)
        np.testing.assert_allclose(velocity,0.)
        self.assertEqual(transform_box(result,np.linalg.inv(ego))[0],7.)

    def test_causal_constant_velocity_and_missing_history(self):
        old = np.array([0.,0.,0.,4.,2.,2.,0.])
        now = old.copy(); now[0] = 2.
        predicted, velocity, valid = extrapolate(now, old, .5, .75)
        self.assertTrue(valid)
        np.testing.assert_allclose(predicted[:2], [5.,0.])
        np.testing.assert_allclose(velocity, [4.,0.,0.])
        for history, dt, age in [(None,.5,.75),(old,0,.75),(old,1.1,.75),(old,.5,-.1)]:
            p, _, valid = extrapolate(now, history, dt, age)
            self.assertFalse(valid); np.testing.assert_array_equal(p,now)

    def test_rotated_transform_and_world_motion(self):
        box = np.array([1.,0.,0.,4.,2.,2.,0.])
        pose = np.array([[0.,-1,0,10],[1,0,0,20],[0,0,1,0],[0,0,0,1]])
        transformed = transform_box(box, pose)
        np.testing.assert_allclose(transformed[:3], [10,21,0])
        self.assertAlmostEqual(transformed[6], np.pi/2)
        np.testing.assert_allclose(transform_box(transformed,np.linalg.inv(pose)),box,atol=1e-12)

    def test_iou_height_and_rotation(self):
        def obj(box): return {'box':np.array(box,dtype=float),'class':0}
        v = [obj([0,0,0,4,2,2,0])]
        i = [obj([0,0,0,4,2,2,0]), obj([0,0,3,4,2,2,0]),obj([0,0,0,4,2,2,np.pi/2])]
        bev, vol, _, _ = pair_features(v,i)
        np.testing.assert_allclose(bev,[ [1,1,1/3] ])
        np.testing.assert_allclose(vol,[ [1,0,1/3] ])

    def test_unmatched_class_gates_and_dense_objects(self):
        def obj(x, cls=0): return {'box':np.array([x,0.,0.,4.,2.,2.,0.]),'class':cls}
        v=[obj(0),obj(1),obj(100)]
        i=[obj(.1),obj(1.1),obj(100,1)]
        for method in 'ABC':
            pairs=match(v,i,pair_features(v,i),method,(2,1,1),.1)
            self.assertEqual(pairs,[(0,0),(1,1)])
        self.assertEqual(match([],i,None,'B',(2,1,1),.1),[])


if __name__ == '__main__': unittest.main()
