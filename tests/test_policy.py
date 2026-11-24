import unittest
import numpy as np
from sure.policy import safe_actions, Kernel, Planner, EmpiricalBayes
from sure.experiment import label

class PolicyTests(unittest.TestCase):
    def test_unsafe_never_repeated(self):
        self.assertNotIn('retry',safe_actions(False,500,3)); self.assertNotIn('failover',safe_actions(False,500,3))
    def test_budget(self):
        self.assertEqual(safe_actions(True,0,3),('stop',)); self.assertNotIn('reconcile',safe_actions(True,50,0))
    def test_labels(self):
        row={'start_ms':100,'commit_ms':200}; self.assertEqual(label(row,50),'unstarted'); self.assertEqual(label(row,150),'processing'); self.assertEqual(label(row,250),'completed')
    def test_planner(self):
        k=Kernel().fit([{'state':'completed','action':'reconcile','keyed':False,'duration_ms':40,'next_state':'completed','observation':'confirmed'}]); p=Planner(k,2)
        self.assertEqual(p.choose(np.array([0,0,1]),False,100,1),'reconcile'); self.assertEqual(p.choose(np.array([0,0,1]),False,0,1),'stop')
if __name__=='__main__': unittest.main()
