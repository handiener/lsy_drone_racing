"""Export all host recordings to CSV and build an integrity manifest; no hardware access."""
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
TRAJECTORIES = OUT / 'trajectories'
TRAJECTORIES.mkdir(exist_ok=True)


def tilt(quat):
    x, y, z, w = quat
    norm2 = x*x + y*y + z*z + w*w
    if norm2 == 0:
        return None
    return math.degrees(math.acos(max(-1, min(1, 1 - 2*(x*x+y*y)/norm2))))


inventory = []
pending = []
for path in sorted((ROOT / 'runs').glob('deploy_*')):
    if not (path / 'summary.json').exists():
        pending.append(path.name)
        continue
    events = [json.loads(line) for line in (path / 'events.jsonl').read_text().splitlines()]
    metadata = json.loads((path / 'metadata.json').read_text())
    summary = json.loads((path / 'summary.json').read_text())
    assert len(events) == summary['records_written'], path.name
    observations = [e for e in events if e['kind'] == 'observation']
    zero = observations[0]['monotonic_ns'] if observations else events[0]['monotonic_ns']
    seconds = lambda e: (e['monotonic_ns'] - zero) / 1e9
    race_end = next((e for e in events if e['kind'] == 'race_end'), None)
    cleanup = next((e for e in events if e['kind'] == 'cleanup_start'), None)
    poses = [e for e in events if e['kind'] == 'external_pose_sent']
    actions = {e['step']: e['action'] for e in events if e['kind'] == 'action'}
    mocap_csv = TRAJECTORIES / f'{path.name}_mocap.csv'
    with mocap_csv.open('w', newline='') as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(['monotonic_ns','race_time_s','host_phase','x_m','y_m','z_m','qx','qy','qz','qw','tilt_deg'])
        for e in poses:
            phase = 'preflight' if seconds(e) < 0 else 'cleanup' if cleanup and e['monotonic_ns'] >= cleanup['monotonic_ns'] else 'policy'
            writer.writerow([e['monotonic_ns'],seconds(e),phase,*e['pos'],*e['quat'],tilt(e['quat'])])
    policy_csv = TRAJECTORIES / f'{path.name}_policy.csv'
    with policy_csv.open('w', newline='') as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(['step','race_time_s','gates_passed','sorted_obstacle_numbers','x_m','y_m','z_m','vx_m_s','vy_m_s','vz_m_s','qx','qy','qz','qw','tilt_deg','command_json'])
        for e in observations:
            obs = e['obs']
            distances = [math.hypot(p[0]-obs['pos'][0], p[1]-obs['pos'][1]) for p in obs['obstacles_pos']]
            order = sorted(range(len(distances)), key=lambda k: distances[k])
            writer.writerow([e['step'],seconds(e),obs['n_gates_passed'],' '.join(str(k+1) for k in order),*obs['pos'],*obs['vel'],*obs['quat'],tilt(obs['quat']),json.dumps(actions.get(e['step']))])
    state = 'no_flight' if not observations else 'finished' if race_end and race_end['finished_track'] else 'ended_without_finish'
    inventory.append({'run':path.name,'status':state,'drone_ids':[d['id'] for d in metadata['config']['deploy']['drones']], 'real_track_objects':metadata['config']['deploy']['real_track_objects'], 'observations':len(observations),'mocap_samples':len(poses),'last_observed_gate_count':observations[-1]['obs']['n_gates_passed'] if observations else None,'race_end':race_end,'exceptions':[e for e in events if e['kind'] in ('exception','cleanup_exception')],'recording_summary':summary,'event_counts':dict(Counter(e['kind'] for e in events)),'exports':[str(mocap_csv.relative_to(ROOT)),str(policy_csv.relative_to(ROOT))]})
(OUT/'inventory.json').write_text(json.dumps(inventory,indent=2)+'\n')
rows=['| Run (UTC directory timestamp) | Status | Drone | Real objects | Policy samples | Mocap samples |', '| --- | --- | --- | --- | ---: | ---: |']
for item in inventory:
    rows.append(f"| [{item['run']}](../{item['run']}/metadata.json) | {item['status']} | {item['drone_ids']} | {item['real_track_objects']} | {item['observations']} | {item['mocap_samples']} |")
(OUT/'inventory.md').write_text('# Hardware recording inventory\n\n'+ '\n'.join(rows)+f'\n\nAll {len(inventory)} finalized raw directories are preserved, including the startup failure and interrupted flights.\nDetailed reports cover selected runs; inclusion in this inventory does not imply a full causal analysis.\n\n`complete` in a recorder summary means the recorder finalized without a writer error, not that the flight finished or that return succeeded. CSV phase `cleanup` identifies host cleanup and may include return or emergency stop.\n\nCSV time zero is the first policy observation where available. Original host timestamps and richer observations, features, commands, measured track and config remain in the raw JSON files.\n')
(OUT/'pending_recordings.json').write_text(json.dumps(pending,indent=2)+'\n')
# Hash every included evidence file except this manifest itself and Python caches.
files = []
for directory in [*(ROOT/'runs'/item['run'] for item in inventory), OUT]:
    files.extend(p for p in directory.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name != 'SHA256SUMS')
readme=ROOT/'runs/README.debug_real.md'
if readme.exists(): files.append(readme)
files=sorted(set(files))
(OUT/'SHA256SUMS').write_text(''.join(f'{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(ROOT)}\n' for p in files))
print(f'Validated {len(inventory)} hardware recordings; exported {2*len(inventory)} CSV files; hashed {len(files)} evidence files.')
