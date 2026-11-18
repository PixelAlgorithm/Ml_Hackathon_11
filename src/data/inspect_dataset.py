"""Inspect every train/student_test trip and record measurements, not guesses."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from src.data.raw import load_trip, list_trips


def inspect(root: Path, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for split in ('train', 'student_test'):
        for folder in list_trips(root, split):
            t = load_trip(folder, split)
            gaps = np.diff(t.track[:, 2])
            coords = t.track[:, :2]
            route_pair = t.arcs[t.route]
            connected = float(np.mean(route_pair[:-1, 1] == route_pair[1:, 0])) if len(route_pair)>1 else None
            rows.append({
                'split': split, 'trip_id': t.trip_id, 'observations':len(t.track),
                'nodes':len(t.nodes), 'arcs':len(t.arcs), 'route_arcs':len(t.route),
                'missing_gps':int(np.sum(~np.isfinite(coords).all(axis=1))),
                'duplicate_consecutive_gps':int(np.sum(np.all(np.diff(coords,axis=0)==0,axis=1))),
                'nonincreasing_timestamps':int(np.sum(gaps<=0)),
                'gaps_over_2_s':int(np.sum(gaps>2)),
                'one_second_steps':int(np.sum(gaps==1)),
                'timestamp_steps':len(gaps),
                'largest_time_gap_s':float(max(gaps,default=0)),
                'lon_min':float(coords[:,0].min()), 'lon_max':float(coords[:,0].max()),
                'lat_min':float(coords[:,1].min()), 'lat_max':float(coords[:,1].max()),
                'route_directed_edge_continuity':connected,
            })
    inventory = pd.DataFrame(rows)
    inventory.to_csv(output_dir/'dataset_inventory.csv', index=False)
    totals = inventory.groupby('split',sort=True)[['observations','nodes','arcs','route_arcs','missing_gps','duplicate_consecutive_gps','nonincreasing_timestamps','gaps_over_2_s','one_second_steps','timestamp_steps']].sum()
    totals.insert(0,'trips',inventory.groupby('split').size())
    totals.to_csv(output_dir/'dataset_summary.csv')
    print(totals.to_string())
    print(f"\nOne-second timestamp share: {inventory.one_second_steps.sum()/inventory.timestamp_steps.sum():.2%}")
    print('Inventory:', output_dir/'dataset_inventory.csv')


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--data-root',required=True,type=Path)
    p.add_argument('--out',default=Path('results'),type=Path)
    args=p.parse_args();inspect(args.data_root,args.out)
