"""Offline analysis of the recorded deployments; never connects to hardware."""
import json
from pathlib import Path
from collections import Counter
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation

root = Path(__file__).resolve().parents[2]
out = Path(__file__).resolve().parent
runs = []
for path in ['deploy_20261117T100017.801841Z', 'deploy_20261117T100651.417694Z', 'deploy_20261117T100827.697140Z']:
    path = root / 'runs' / path
    events = [json.loads(line) for line in (path / 'events.jsonl').read_text().splitlines()]
    observations = [e for e in events if e['kind'] == 'observation']
    if not observations:
        runs.append({'run': path.name, 'status': 'no_flight', 'events': events})
        continue
    zero = observations[0]['monotonic_ns']
    time = lambda e: (e['monotonic_ns'] - zero) / 1e9
    finish = next(e for e in events if e['kind'] == 'race_end')
    finish_t = time(finish)
    poses = [e for e in events if e['kind'] == 'external_pose_sent']
    t = np.array([time(e) for e in poses])
    pos = np.array([e['pos'] for e in poses])
    commands = [e for e in events if e['kind'] == 'action']
    commands_by_step = {e['step']: e for e in commands}
    actions = np.array([commands_by_step[e['step']]['action'] for e in observations])
    obs_t = np.array([time(e) for e in observations])
    rotation = Rotation.from_quat([e['obs']['quat'] for e in observations]).as_matrix()
    cos_tilt = rotation[:, 2, 2]
    terminal = [e for e in events if e['kind'] == 'transition'][-1]['next_obs']
    terminal_tilt = np.degrees(np.arccos(np.clip(Rotation.from_quat(terminal['quat']).as_matrix()[2, 2], -1, 1)))
    reset = next(e for e in events if e['kind'] == 'reset')
    weight = reset['drone_parameters']['mass'] * 9.81
    race_mask = (t > 1) & (t <= finish_t)
    return_mask = (t > finish_t) & (t < finish_t + 3)
    race_i = np.flatnonzero(race_mask)[np.argmin(pos[race_mask, 2])]
    return_i = np.flatnonzero(return_mask)[np.argmin(pos[return_mask, 2])]
    loop = np.array([e['elapsed_s'] for e in events if e['kind'] == 'loop'])
    metadata = json.loads((path / 'metadata.json').read_text())
    summary = {
        'run': path.name, 'status': 'finished', 'finish_time_s': finish['flight_time_s'],
        'steps': finish['steps'], 'finish_pos_m': terminal['pos'],
        'finish_vel_m_s': terminal['vel'], 'finish_speed_m_s': float(np.linalg.norm(terminal['vel'])),
        'finish_tilt_deg': float(terminal_tilt),
        'race_min_height_after_1s_m': float(pos[race_i, 2]), 'race_min_height_time_s': float(t[race_i]),
        'return_min_height_m': float(pos[return_i, 2]),
        'return_min_height_after_finish_s': float(t[return_i] - finish_t),
        'return_min_height_pos_m': pos[return_i].tolist(),
        'max_return_y_first_3s_m': float(pos[return_mask, 1].max()),
        'loop_processing_max_ms': float(loop.max() * 1e3), 'loop_overruns': int(sum(loop > .02)),
        'inference_p95_ms': float(np.quantile([(e['inference_end_ns'] - e['inference_start_ns']) / 1e6 for e in commands], .95)),
        'mocap_update_max_gap_ms': float(np.diff(t).max() * 1e3),
        'estimated_vs_mocap_position_error_p95_m': float(np.quantile([np.linalg.norm(np.array(e['obs']['pos'])-e['mocap']['pos']) for e in observations], .95)),
        'first_braking_target': next(e for e in events if e['kind'] == 'go_to_sent'),
        'nominal_weight_N': weight, 'last_policy_thrust_N': float(actions[-1, 3]),
        'last_command_vertical_projection_N': float(actions[-1, 3] * cos_tilt[-1]),
        'recording': json.loads((path / 'summary.json').read_text()),
    }
    runs.append({'summary': summary, 'events': events, 't': t, 'pos': pos, 'obs_t': obs_t,
                 'actions': actions, 'cos_tilt': cos_tilt, 'weight': weight, 'reset': reset,
                 'finish_t': finish_t, 'config': metadata['config'], 'observations': observations})

flights = [r for r in runs if 'summary' in r]
fig, axes = plt.subplots(4, len(flights), figsize=(14, 16), squeeze=False)
for col, r in enumerate(flights):
    s = r['summary']; t = r['t']; pos = r['pos']; ft = r['finish_t']
    ax = axes[0, col]
    ax.plot(t, pos[:, 2], label='Mocap height')
    ax.axvline(ft, color='C3', label='Race finish / return handoff')
    ax.axhline(.05, color='gray', ls=':', label='Approx. resting height')
    landing = [e for e in r['events'] if e['kind'] == 'go_to_sent'][-1]
    zero = r['observations'][0]['monotonic_ns']
    land_t = (landing['monotonic_ns'] - zero) / 1e9
    ax.axvline(land_t, color='C2', ls='--', label='Planned landing command')
    ax.set(title=s['run'], xlabel='Seconds from first policy observation', ylabel='Mocap z (m)')
    ax.legend(fontsize=8)
    ax = axes[1, col]
    ax.plot(pos[t <= ft, 0], pos[t <= ft, 1], label='Race')
    ax.plot(pos[(t > ft) & (t < ft+3), 0], pos[(t > ft) & (t < ft+3), 1], color='C3', label='First 3 s of return')
    ax.scatter(*np.array(s['return_min_height_pos_m'])[:2], marker='x', s=70, color='black', label='Near-floor dip')
    gates = np.array(r['reset']['track']['gates_pos'])
    ax.scatter(gates[:, 0], gates[:, 1], marker='s', color='C2', label='Gate centers')
    for i, g in enumerate(gates): ax.annotate(str(i+1), g[:2])
    bounds = r['config']['env']['track']['safety_limits']
    lo, hi = bounds['pos_limit_low'], bounds['pos_limit_high']
    ax.plot([lo[0], hi[0], hi[0], lo[0], lo[0]], [lo[1], lo[1], hi[1], hi[1], lo[1]], 'k--', lw=1, label='Configured bounds')
    ax.set(xlabel='x (m)', ylabel='y (m)', aspect='equal'); ax.legend(fontsize=8)
    ax = axes[2, col]
    mask = (t >= ft-.5) & (t <= ft+3)
    ax.plot(t[mask]-ft, pos[mask, 2], label='Mocap height')
    ax.axvline(0, color='C3', label='Return handoff')
    ax.axhline(.05, color='gray', ls=':')
    ax.scatter(s['return_min_height_after_finish_s'], s['return_min_height_m'], color='black')
    ax.set(xlabel='Seconds relative to race finish', ylabel='Mocap z (m)', title='Unplanned descent before planned landing'); ax.legend(fontsize=8)
    ax = axes[3, col]
    ot = r['obs_t']; mask = ot >= ft-.7
    ax.plot(ot[mask]-ft, r['actions'][mask, 3], label='Commanded total thrust')
    ax.plot(ot[mask]-ft, r['actions'][mask, 3]*r['cos_tilt'][mask], label='Command × observed vertical projection')
    ax.axhline(r['weight'], color='black', ls='--', label='Configured weight (mg)')
    ax.set(xlabel='Seconds relative to race finish', ylabel='Force (N)', title='Final policy segment (not measured motor thrust)'); ax.legend(fontsize=8)
for ax in axes.flat: ax.grid(alpha=.25)
fig.tight_layout()
fig.savefig(out/'flight_analysis.png', dpi=160)
fig.savefig(out/'flight_analysis.pdf')
metrics = [r.get('summary', r) for r in runs]
(out/'metrics.json').write_text(json.dumps(metrics, indent=2)+'\n')
print(json.dumps([r['summary'] for r in flights], indent=2))
