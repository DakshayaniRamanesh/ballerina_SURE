import json
import unittest
from unittest.mock import patch
import numpy as np
from sure import independent_recovery as legacy
from sure.recovery_core import recover, SPECS, STATES
from sure.recovery_estimators import load_frozen, RandomForestAdapter
from sure.recovery_http import PaymentTransport
from bayesian.independent_evaluation import check_models, verify_freeze

class Clock:
    def __init__(self): self.t=0.
    def __call__(self): return self.t

class ScriptedClient:
    def __init__(self, clock, initial, responses):
        self.clock=clock; self.initial=initial; self.responses=responses; self.actions=[]
    def create_case(self, case): return 'fixture-test'
    def action(self,a,ident,keyed,remaining,cap=180):
        obs,dur=self.initial if not self.actions else self.responses.get(a,('error',25.))
        dur=min(dur,remaining,cap); self.clock.t+=dur; self.actions.append(a)
        return obs,dur
    def settled_truth(self,ident,n): return {'effects':1}

class RecoveryCoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        verify_freeze()
        cls.estimator,cls.kernel=load_frozen()
        cls.old_estimator,cls.old_kernel=check_models()

    def test_scripted_parity(self):
        count=0
        for policy in legacy.SPECS:
            for keyed in (False,True):
                for cap,deadline,budget in ((100,350,1),(150,750,2),(200,1250,4)):
                    for initial in ('confirmed','timeout','error','pending','unknown'):
                        for obs in ('confirmed','pending','error','unknown','timeout'):
                            c1,c2=Clock(),Clock()
                            responses={a:(obs,50.) for a in ('reconcile','retry','failover')}
                            responses['wait']=('waited',75.)
                            a=ScriptedClient(c1,(initial,cap),responses)
                            b=ScriptedClient(c2,(initial,cap),responses)
                            case=dict(case=0,family='scripted',keyed=keyed,cap=cap,deadline=deadline,budget=budget,process_ms=320)
                            with patch.object(legacy,'now',c1):
                                old=legacy.episode(a,self.old_kernel,self.old_estimator,policy,legacy.SPECS[policy],case)
                            new=recover(lambda action,remaining,cap=180:b.action(action,'fixture-test',keyed,remaining,cap),self.kernel,self.estimator,policy,keyed=keyed,cap=cap,deadline=deadline,budget=budget,clock=c2)
                            self.assertEqual(a.actions,b.actions)
                            for k in new.keys()-{'trace','client_confirmed_on_time'}:
                                self.assertEqual(new[k],old[k],(policy,case,initial,obs,k))
                            self.assertEqual(new['trace'],json.loads(old['trace']))
                            self.assertEqual(new['client_confirmed_on_time'],old['confirmed_on_time'])
                            count+=1
        print(f'Exact scripted parity: {count} policy episodes')

    def test_rf_schema_and_order(self):
        class Model:
            classes_=np.array(['completed','unstarted','processing'])
            feature_names_in_=np.array(['error','keyed','initial_timeout_ms'])
            def predict_proba(self,x):
                assert list(x.columns)==['error','keyed','initial_timeout_ms']
                return np.array([[.6,.1,.3]])
        adapter=RandomForestAdapter(Model())
        np.testing.assert_allclose(adapter.predict(dict(error='timeout',keyed=True,initial_timeout_ms=100,elapsed_ms=100)),[.1,.3,.6])

    def test_rf_invalid_schema(self):
        class Model:
            classes_=['unstarted','processing']
        with self.assertRaises(ValueError): RandomForestAdapter(Model())

    def test_v3_v5_planner_aliases(self):
        self.assertEqual(SPECS['v2'],SPECS['v3'])
        self.assertEqual(SPECS['v4'],SPECS['v5'])

    def test_invalid_inputs_before_dispatch(self):
        def fail(*args): self.fail('Must not dispatch')
        for kwargs in (dict(keyed='false',cap=100,deadline=350,budget=1),dict(keyed=False,cap=0,deadline=350,budget=1),dict(keyed=False,cap=100,deadline=350,budget=-1)):
            with self.assertRaises(ValueError): recover(fail,self.kernel,self.estimator,'v4',**kwargs)

    def test_http_wrong_confirmation_id(self):
        class Response:
            ok=True
            def json(self): return {'id':'wrong','status':'completed'}
        class Session:
            def post(self,*args,**kwargs): return Response()
        transport=PaymentTransport('http://localhost','right',True,Session())
        self.assertEqual(transport('retry',350,100)[0],'error')

if __name__=='__main__': unittest.main(verbosity=2)
