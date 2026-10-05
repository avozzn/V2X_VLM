import tempfile
import unittest
from pathlib import Path
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from tools.object_vlm.distributed import local_indexes,sum_gradients,summarize_predictions


def worker(rank,path):
    torch.set_num_threads(1)
    dist.init_process_group('gloo',init_method='file://'+path,rank=rank,world_size=4)
    try:
        w=torch.nn.Parameter(torch.tensor(1.));u=torch.nn.Parameter(torch.tensor(.5));unused=torch.nn.Parameter(torch.tensor(2.))
        ref_w=torch.nn.Parameter(torch.tensor(1.));ref_u=torch.nn.Parameter(torch.tensor(.5))
        opt=torch.optim.SGD([w,u,unused],lr=.1);ref=torch.optim.SGD([ref_w,ref_u],lr=.1)
        for indexes in [[0,1,2,3],[4]]:
            opt.zero_grad();ref.zero_grad()
            def loss(index,a,b):
                return (a*(index+1)+(b if index==0 else 0)-2)**2
            sum(loss(i,ref_w,ref_u) for i in indexes).div(len(indexes)).backward()
            for i in local_indexes(indexes,rank,4):loss(i,w,u).div(len(indexes)).backward()
            sum_gradients([w,u,unused])
            torch.testing.assert_close(w.grad,ref_w.grad)
            if ref_u.grad is None:assert u.grad is None
            else:torch.testing.assert_close(u.grad,ref_u.grad)
            assert unused.grad is None
            opt.step();ref.step()
            torch.testing.assert_close(w,ref_w);torch.testing.assert_close(u,ref_u)
    finally:dist.destroy_process_group()


class DistributedTest(unittest.TestCase):
    def test_four_rank_gradient_mean_and_last_singleton(self):
        with tempfile.TemporaryDirectory() as tmp:
            mp.spawn(worker,args=(str(Path(tmp)/'rendezvous'),),nprocs=4,join=True)
    def test_929_samples_no_repeat_or_drop(self):
        seen=[]
        for offset in range(0,929,4):
            indexes=list(range(offset,min(offset+4,929)))
            for rank in range(4):seen.extend(local_indexes(indexes,rank,4))
        self.assertEqual(sorted(seen),list(range(929)));self.assertEqual(len(seen),929)
    def test_validation_restores_original_order_and_coverage(self):
        rows=[{'token':str(i),'target':{'future_xy':[[0,0]]*9,'future_mask':[True]*9}} for i in range(3)]
        predictions=[{'token':'2','trajectory':None},{'token':'0','trajectory':[[0,0]]*9},{'token':'1','trajectory':[[0,0]]*9}]
        def metrics(*args):return {'l2_1_2_3s':[0,0,0],'avg_l2_1_2_3s':0}
        report,ordered=summarize_predictions(rows,predictions,1,[10,20,30],1,metrics)
        self.assertEqual([x['token'] for x in ordered],['0','1','2']);self.assertEqual(report['coverage'],2/3)
        with self.assertRaises(ValueError):summarize_predictions(rows,predictions[:-1],1,[10],1,metrics)

if __name__=='__main__':unittest.main()
