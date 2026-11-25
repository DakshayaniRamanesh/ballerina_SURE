"""HTTP adapter for the SURE payment fixture. Not a general production client."""
import time
import requests
from .recovery_core import HTTP_CAP, WAIT_MS

class PaymentTransport:
    def __init__(self, base, request_id, keyed, session=None):
        self.base=base.rstrip('/')
        self.request_id=request_id
        self.keyed=keyed
        self.session=session if session is not None else requests.Session()

    def __call__(self, action, remaining_ms, cap_ms=HTTP_CAP):
        start=time.monotonic()*1000
        if remaining_ms<=0:
            raise ValueError('Action dispatched after deadline')
        if action=='wait':
            time.sleep(min(WAIT_MS,remaining_ms)/1000)
            return 'waited',time.monotonic()*1000-start
        timeout=min(cap_ms,remaining_ms)/1000
        try:
            if action=='reconcile':
                res=self.session.get(f'{self.base}/primary/payments/{self.request_id}',timeout=timeout,allow_redirects=False)
            elif action in ('retry','failover'):
                endpoint='backup' if action=='failover' else 'primary'
                res=self.session.post(f'{self.base}/{endpoint}/payments',json={'id':self.request_id,'keyed':self.keyed},timeout=timeout,allow_redirects=False)
            else:
                raise ValueError('Unknown action')
            body=res.json()
            if not res.ok or not isinstance(body,dict) or body.get('id')!=self.request_id:
                obs='error'
            else:
                obs={'completed':'confirmed','pending':'pending','unknown':'unknown'}.get(body.get('status'),'error')
        except requests.Timeout:
            obs='timeout'
        except (requests.RequestException,ValueError):
            obs='error'
        return obs,time.monotonic()*1000-start
