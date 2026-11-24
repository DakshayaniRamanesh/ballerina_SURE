"""SURE multi-step collection pilot. Run from project root.
Uses only the local synthetic fixture; admin truth is read after each trajectory.
No training or changes to existing datasets. All labels are provisional clock-based.
"""
import argparse
import csv
import json
import random
import time
from collections import Counter
from pathlib import Path

from sure.experiment import MODES, label
from bayesian.evaluate_bayesian_planner import RecoveryClient

WAIT_MS = 75
HTTP_CAP = 180
DEADLINE_MS = 1000
MAX_CALLS = 3
BOUNDARY_MS = 2
BASE = (
    ('wait', 'wait', 'wait', 'reconcile'),
    ('reconcile', 'wait', 'reconcile'),
    ('wait', 'reconcile', 'wait', 'reconcile'),
)
KEYED_EXTRA = (
    ('wait', 'wait', 'wait', 'failover'),
    ('failover', 'wait', 'retry'),
    ('retry', 'wait', 'reconcile'),
)
FIELDS = ('episode mode keyed initial_timeout_ms repeat sequence step initial_observation '
          'initial_elapsed_ms elapsed_before_ms previous_observation history_json action '
          'observation duration_ms elapsed_after_ms decision_epoch_ms action_end_epoch_ms '
          'state next_state start_ms commit_ms effects attempts boundary_uncertain usable').split()
EP_FIELDS = ('episode mode keyed initial_timeout_ms repeat sequence initial_observation '
             'initial_elapsed_ms final_observation steps calls stop_reason effects attempts '
             'start_ms commit_ms deadline_overrun').split()


def schedule(repeats, seed):
    jobs = [(m, k, cap, rep, seq) for rep in range(1, repeats + 1)
            for m in MODES for k in (False, True) for cap in (100, 150, 200)
            for seq in BASE + (KEYED_EXTRA if k else ())]
    random.Random(seed).shuffle(jobs)
    return jobs


def monotonic_ms():
    return time.perf_counter_ns() / 1e6


def epoch_ms():
    return time.time_ns() // 1_000_000


def near_boundary(truth, t):
    return any(truth[x] > 0 and abs(truth[x] - t) <= BOUNDARY_MS
               for x in ('start_ms', 'commit_ms'))


def collect_episode(c, job, clock=monotonic_ms, epoch=epoch_ms, sleep=time.sleep):
    mode, keyed, cap, rep, seq = job
    ident = c.create(mode, keyed)
    common = dict(episode=ident, mode=mode, keyed=keyed, initial_timeout_ms=cap,
                  repeat=rep, sequence='>'.join(seq))
    start = clock()
    obs, dt = c.action('retry', ident, keyed, DEADLINE_MS, cap)
    initial_end = epoch()
    common.update(initial_observation=obs, initial_elapsed_ms=dt)
    rows, history, calls = [], [], 0
    reason = 'initial_confirmed' if obs == 'confirmed' else 'sequence_finished'
    for step, action in enumerate(seq, 1):
        if obs == 'confirmed':
            break
        if action in ('retry', 'failover') and not keyed:
            raise RuntimeError('Unsafe scheduled action')
        if action != 'wait' and calls >= MAX_CALLS:
            reason = 'call_budget'; break
        before = clock() - start
        # Fixed-cap collection only; don't silently change the action timeout.
        reserve = 150 if action == 'wait' else 275
        if DEADLINE_MS - before < reserve:
            reason = 'deadline_admission'; break
        t0 = epoch()
        previous = obs
        obs, duration = c.action(action, ident, keyed, DEADLINE_MS - before, HTTP_CAP)
        t1 = epoch()
        after = clock() - start
        rows.append(dict(common, step=step, elapsed_before_ms=before,
                         previous_observation=previous, history_json=json.dumps(history),
                         action=action, observation=obs, duration_ms=duration,
                         elapsed_after_ms=after, decision_epoch_ms=t0, action_end_epoch_ms=t1))
        history.append(dict(action=action, observation=obs, duration_ms=duration))
        calls += int(action != 'wait')
        if obs == 'confirmed':
            reason = 'recovery_confirmed'; break
    elapsed = clock() - start
    # No privileged lookup during recovery; settle late fixture commits before audit.
    sleep(.50)
    truth = c.truth(ident)
    audit = {k: int(truth[k]) for k in ('start_ms', 'commit_ms', 'effects', 'attempts')}
    if audit['effects'] > 1:
        raise RuntimeError(f'Duplicate effect in {ident}; stop and investigate: {audit}')
    if audit['effects'] == 1 and not audit['commit_ms']:
        raise RuntimeError(f'Missing commit time in {ident}')
    if obs == 'confirmed':
        confirmation_end = rows[-1]['action_end_epoch_ms'] if rows else initial_end
        if audit['effects'] != 1 or not audit['commit_ms'] or audit['commit_ms'] > confirmation_end + BOUNDARY_MS:
            raise RuntimeError(f'Inconsistent confirmation in {ident}: {audit}')
    for row in rows:
        t0, t1 = row['decision_epoch_ms'], row['action_end_epoch_ms']
        if t1 < t0:
            raise RuntimeError('Wall clock moved backwards')
        uncertain = near_boundary(audit, t0) or near_boundary(audit, t1)
        row.update(audit, state=label(audit, t0), next_state=label(audit, t1),
                   boundary_uncertain=int(uncertain), usable=int(not uncertain))
        if not uncertain and row['observation'] == 'confirmed' and row['next_state'] != 'completed':
            raise RuntimeError('Unexplained confirmation label')
    ep = dict(common, final_observation=obs, steps=len(rows), calls=calls,
              stop_reason=reason, deadline_overrun=int(elapsed > DEADLINE_MS), **audit)
    return rows, ep


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repeats', type=int, default=1)
    p.add_argument('--seed', type=int, default=6107)
    p.add_argument('--base', default='http://127.0.0.1:9090')
    p.add_argument('--out', default='data/recovery_multistep_pilot.csv')
    args = p.parse_args()
    if args.repeats < 1: p.error('--repeats must be positive')
    out = Path(args.out)
    epout = out.with_name(out.stem + '_episodes.csv')
    if out.exists() or epout.exists(): raise FileExistsError('Choose a new --out; existing data preserved')
    c = RecoveryClient(args.base)
    if c.admin('/health').get('status') != 'ok': raise RuntimeError('Testbed unhealthy')
    out.parent.mkdir(parents=True, exist_ok=True)
    jobs = schedule(args.repeats, args.seed)
    counts, waits, totals = Counter(), Counter(), Counter()
    print(f'Planned episodes: {len(jobs)}', flush=True)
    with out.open('x', newline='', encoding='utf-8') as f, epout.open('x', newline='', encoding='utf-8') as e:
        w, ew = csv.DictWriter(f, fieldnames=FIELDS), csv.DictWriter(e, fieldnames=EP_FIELDS)
        w.writeheader(); ew.writeheader()
        for i, job in enumerate(jobs, 1):
            rows, ep = collect_episode(c, job)
            w.writerows(rows); ew.writerow(ep); f.flush(); e.flush()
            totals['raw_steps'] += len(rows); totals['overruns'] += ep['deadline_overrun']
            totals['initial_confirmed'] += int(ep['initial_observation'] == 'confirmed')
            for row in rows:
                totals['quarantined'] += int(not row['usable'])
                if row['usable']:
                    totals['usable'] += 1
                    counts[(row['step'], row['action'])] += 1
                    if row['action'] == 'wait':
                        waits[(row['state'], row['next_state'], row['keyed'])] += 1
            if i % 15 == 0 or i == len(jobs):
                print(f'{i}/{len(jobs)} episodes | {totals["raw_steps"]} recovery steps', flush=True)
    print('\nPilot totals:', dict(totals))
    print('\nUsable waiting transitions (by deduplication):')
    for k in (False, True):
        for s in ('unstarted', 'processing', 'completed'):
            for ns in ('unstarted', 'processing', 'completed'):
                if waits[s,ns,k] or (s=='processing' and ns=='completed'):
                    print(f'{k!s:5} {s:10} -> {ns:10}: {waits[s,ns,k]}')
    print('\nUsable step/action coverage:')
    for (step,a),n in sorted(counts.items()): print(step,a,n)
    print(f'\nSaved {out} and {epout}. Pilot only; do not train before auditing trajectories.')

if __name__ == '__main__':
    main()
