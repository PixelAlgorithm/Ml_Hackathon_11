"""Reading PathSnap raw, unheaded whitespace files. Each trip has local arc/node IDs."""
from pathlib import Path
from dataclasses import dataclass
import numpy as np

@dataclass
class Trip:
    trip_id: str
    split: str
    track: np.ndarray  # GPS rows [longitude, latitude, timestamp_seconds]
    nodes: np.ndarray  # coordinate rows [longitude, latitude]
    arcs: np.ndarray   # [from_node_id, to_node_id] -- row index = arc_id
    route: np.ndarray | None  # chronological route arc IDs; None for inference


def load_trip(folder: Path, split: str, read_route: bool = True) -> Trip:
    folder = Path(folder)
    ident = folder.name
    def read(extension, dtype=float):
        path = folder / f"{ident}.{extension}"
        if not path.exists():
            raise FileNotFoundError(path)
        return np.loadtxt(path, dtype=dtype, ndmin=2 if extension!='route' else 1)
    trip = Trip(ident, split, read('track'), read('nodes'), read('arcs', int), read('route', int).reshape(-1) if read_route else None)
    if trip.track.shape[1] != 3 or trip.nodes.shape[1] != 2 or trip.arcs.shape[1] != 2:
        raise ValueError(f"Unexpected file columns in {folder}")
    if not np.isfinite(trip.track).all() or not np.isfinite(trip.nodes).all():
        raise ValueError(f"Non-finite coordinates or time in {folder}")
    if (trip.arcs < 0).any() or (trip.arcs >= len(trip.nodes)).any():
        raise ValueError(f"Invalid node reference in {folder}")
    if read_route and ((trip.route < 0).any() or (trip.route >= len(trip.arcs)).any()):
        raise ValueError(f"Invalid route arc ID in {folder}")
    return trip


def list_trips(data_root: Path, split: str):
    base = Path(data_root)/split
    if not base.is_dir():
        raise FileNotFoundError(f"Expected split directory: {base}")
    return sorted(d for d in base.iterdir() if d.is_dir())


def to_local_xy(lonlat: np.ndarray, lon0: float, lat0: float):
    """Local equirectangular approximation in meters; origin chosen per trip.

    The trips span multiple countries. DO NOT use a single universal UTM zone.
    For regional trajectories, a per-trip projection is adequate for exploration;
    consider local azimuthal-equidistant/UTM for higher-precision work.
    """
    radius = 6_371_008.8
    xy = np.asarray(lonlat)
    return np.column_stack((
        np.deg2rad(xy[:, 0]-lon0)*radius*np.cos(np.deg2rad(lat0)),
        np.deg2rad(xy[:, 1]-lat0)*radius,
    ))
