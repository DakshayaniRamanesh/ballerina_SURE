"""Balanced scenario/action collection for the SURE starter testbed.
Run from project root: python -m bayesian.collect_balanced --repeats 8
Only uses local testbed admin APIs; never expose them in production.
"""
import argparse
import csv
from collections import Counter
from pathlib import Path
import random
import time

from sure.experiment import Client, MODES, epoch, features, label
from sure.policy import safe_actions

FIELDS = ['elapsed_ms', 'error', 'keyed', 'initial_timeout_ms', 'episode',
          'mode', 'state', 'action', 'observation', 'duration_ms', 'next_state']
AUDIT = ['episode', 'mode', 'keyed', 'initial_timeout_ms', 'planned_action',
         'initial_observation', 'initial_elapsed_ms', 'decision_epoch_ms',
         'action_end_epoch_ms', 'state', 'next_state', 'start_ms', 'commit_ms',
         'effects', 'attempts', 'recovery_sample']


def schedule(repeats, seed):
    cells = [(mode, keyed, cap, a) for mode in MODES
             for keyed in (False, True) for cap in (100, 150, 200)
             for a in safe_actions(keyed, 500, 3) if a != 'stop']
    jobs = cells * repeats
    random.Random(seed).shuffle(jobs)
    return jobs


def collect(args):
    if args.repeats < 1:
        raise ValueError('repeats must be positive')
    out = Path(args.out)
    audit = out.with_name(out.stem + '_episodes.csv')
    if out.exists() or audit.exists():
        raise FileExistsError('Output already exists; choose a new --out path.')
    out.parent.mkdir(parents=True, exist_ok=True)
    c = Client(args.base)
    if c.admin('/health').get('status') != 'ok':
        raise RuntimeError('Testbed health check failed')
    jobs = schedule(args.repeats, args.seed)
    counts = Counter()
    print(f'Planned initial episodes: {len(jobs)}', flush=True)
    with out.open('x', newline='', encoding='utf-8') as rf, audit.open('x', newline='', encoding='utf-8') as af:
        rw, aw = csv.DictWriter(rf, fieldnames=FIELDS), csv.DictWriter(af, fieldnames=AUDIT)
        rw.writeheader(); aw.writeheader()
        for i, (mode, keyed, cap, a) in enumerate(jobs, 1):
            ident = c.create(mode, keyed)
            obs, dt = c.action('retry', ident, keyed, cap, cap)
            t0 = epoch()
            recovered = obs != 'confirmed'
            t1, state, next_state = t0, '', ''
            if recovered:
                assert a in safe_actions(keyed, 500, 3)
                o, d = c.action(a, ident, keyed, 500)
                t1 = epoch()
                time.sleep(.36)
            truth = c.truth(ident)
            for k in ('start_ms', 'commit_ms', 'effects', 'attempts'):
                if k not in truth:
                    raise RuntimeError(f'Missing truth field {k} for {ident}')
            state, next_state = label(truth, t0), label(truth, t1)
            if recovered:
                rw.writerow(dict(features(obs, dt, keyed, cap), episode=ident,
                                 mode=mode, state=state, action=a, observation=o,
                                 duration_ms=d, next_state=next_state))
                counts[(state, a, keyed)] += 1
                rf.flush()
            aw.writerow(dict(episode=ident, mode=mode, keyed=keyed,
                             initial_timeout_ms=cap, planned_action=a,
                             initial_observation=obs, initial_elapsed_ms=dt,
                             decision_epoch_ms=t0, action_end_epoch_ms=t1,
                             state=state, next_state=next_state,
                             start_ms=truth['start_ms'], commit_ms=truth['commit_ms'],
                             effects=truth['effects'], attempts=truth['attempts'],
                             recovery_sample=int(recovered)))
            af.flush()
            if i % 25 == 0 or i == len(jobs):
                print(f'{i}/{len(jobs)} initial episodes; {sum(counts.values())} recovery samples', flush=True)
    print('\nState/action/keyed counts (zero counts included for eligible actions):')
    for keyed in (False, True):
        for state in ('unstarted', 'processing', 'completed'):
            for a in safe_actions(keyed, 500, 3):
                if a != 'stop':
                    print(f'{str(keyed):5} {state:10} {a:10} {counts[state,a,keyed]:4}')
    print(f'\nRecovery data: {out}\nAll initial episodes and truth audit: {audit}')
    print('Collection completed. Do not use privileged audit fields as runtime features.')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repeats', type=int, default=8)
    p.add_argument('--seed', type=int, default=2027)
    p.add_argument('--base', default='http://127.0.0.1:9090')
    p.add_argument('--out', default='data/recovery_balanced.csv')
    collect(p.parse_args())


if __name__ == '__main__':
    main()
