"""Local reference service; same HTTP contract as the Ballerina fixture.
Only for experiments. Fault scheduling and truth are privileged admin operations.
"""
import argparse, json, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

lock = threading.Lock()
ledger = {}

def now(): return int(time.time()*1000)

class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_): pass
    def reply(self, obj, code=200):
        data=json.dumps(obj).encode()
        try:
            self.send_response(code); self.send_header('Content-Type','application/json')
            self.send_header('Content-Length',str(len(data))); self.end_headers(); self.wfile.write(data)
        except (BrokenPipeError, ConnectionResetError): pass
    def body(self): return json.loads(self.rfile.read(int(self.headers['Content-Length'])))
    def do_POST(self):
        p=self.path.strip('/').split('/')
        data=self.body(); ident=data['id']
        if p==['admin','scenario']:
            with lock:
                ledger[ident]={'mode':data['mode'],'keyed':data['keyed'],'attempts':0,'effects':0,'start_ms':0,'commit_ms':0}
            return self.reply({'ok':True})
        if len(p)!=2 or p[1]!='payments' or p[0] not in ('primary','backup'): return self.reply({'error':'path'},404)
        with lock:
            row=ledger.get(ident)
            if row is None: return self.reply({'error':'unknown id'},400)
            if row['keyed'] != data['keyed']: return self.reply({'error':'contract mismatch'},409)
            row['attempts']+=1; attempt=row['attempts']; mode=row['mode']
            if mode=='primary_down' and p[0]=='primary': return self.reply({'status':'error','id':ident},503)
            if row['keyed'] and row['effects']: return self.reply({'status':'completed','id':ident})
            if row['keyed'] and row['start_ms']: return self.reply({'status':'pending','id':ident})
            dropped=attempt==1 and mode=='drop'
            if not dropped and not row['start_ms']: row['start_ms']=now()
        if dropped:
            time.sleep(.32); return self.reply({'status':'unknown','id':ident},503)
        if attempt==1 and mode=='slow': time.sleep(.32)
        else: time.sleep(.025 if p[0]=='primary' else .035)
        with lock:
            row['effects']+=1
            if not row['commit_ms']: row['commit_ms']=now()
        if attempt==1 and mode in ('lost','status_slow','status_down'): time.sleep(.32)
        return self.reply({'status':'completed','id':ident})
    def do_GET(self):
        p=self.path.strip('/').split('/')
        if p==['health']: return self.reply({'status':'ok'})
        if p==['metrics']:
            with lock: totals={'logical_operations':len(ledger),'effects':sum(r['effects'] for r in ledger.values()),'attempts':sum(r['attempts'] for r in ledger.values())}
            return self.reply(totals)
        if len(p)==3 and p[:2]==['admin','truth']:
            with lock: row=ledger.get(p[2]); result=dict(row) if row else None
            return self.reply(result or {},200 if result else 404)
        if len(p)==3 and p[1]=='payments' and p[0] in ('primary','backup'):
            with lock:
                row=ledger.get(p[2]); mode=row['mode'] if row else ''
            if mode=='status_down': return self.reply({'status':'error','id':p[2]},503)
            time.sleep(.045 if mode!='status_slow' else .25)
            with lock:
                state='completed' if row and row['effects'] else 'pending' if row and row['start_ms'] else 'unknown'
            return self.reply({'id':p[2],'status':state})
        return self.reply({'error':'path'},404)

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--port',type=int,default=9090); args=p.parse_args()
    print(f'Reference service on http://127.0.0.1:{args.port}',flush=True)
    ThreadingHTTPServer(('127.0.0.1',args.port),Handler).serve_forever()
