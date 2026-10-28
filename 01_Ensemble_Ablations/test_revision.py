import unittest
from pathlib import Path
import numpy as np
import gi_ensemble as gi

class RevisionTests(unittest.TestCase):
    def test_exact_dataset_counts_and_shared_weights(self):
        cfg=gi.config(Path(__file__).with_name('config.json'))
        rng=np.random.default_rng(19)
        y=np.tile([0,1],30)
        train=[rng.normal(size=(60,512)) for _ in range(2)]
        for dataset,k in cfg['fusion']['fixed_feature_counts'].items():
            _,counts,_,_=gi.fit_branch_models(train,[x[:8] for x in train],[x[:8] for x in train],y,y[:8],[k],2)
            self.assertEqual(counts,[k,k])
        with self.assertRaises(ValueError):
            gi.fit_branch_models([train[0][:,:8]],[train[0][:8,:8]],[train[0][:8,:8]],y,y[:8],[256],2)
    def test_a4_does_not_copy_a0(self):
        rng=np.random.default_rng(4);y=np.tile([0,1],40)
        X=np.column_stack([y+rng.normal(0,.01,len(y)),rng.normal(0,100,len(y)),rng.normal(size=(len(y),4))])
        anova=gi.fit_branch_models([X],[X[:8]],[X[:8]],y,y[:8],[1],2)[0][0][0]
        variance=gi.fit_branch_models([X],[X[:8]],[X[:8]],y,y[:8],[1],2,select_features='variance')[0][0][0]
        self.assertNotEqual(anova.get_support(indices=True).tolist(),variance.get_support(indices=True).tolist())
    def test_fixed_weights_are_active(self):
        rng=np.random.default_rng(8);y=np.tile([0,1],20);X=rng.normal(size=(40,8))
        result=gi.branch_ensemble([X,X],[X[:8],X[:8]],[X[:8],X[:8]],y,y[:8],[4],2,.1,fixed_weights=[.3,.7])
        np.testing.assert_array_equal(result[2],[.3,.7])
        self.assertEqual(result[5],'fixed_config_weights')
if __name__=='__main__':unittest.main()
