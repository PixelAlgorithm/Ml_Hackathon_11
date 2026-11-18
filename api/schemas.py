from typing import Dict, List, Optional

from pydantic import BaseModel


class PredictRequest(BaseModel):
    trip_id: str


class TripInfo(BaseModel):
    trip_id: str


class RouteMetrics(BaseModel):
    route_precision: Optional[float] = None   # null when the trip has no reference route
    route_recall: Optional[float] = None
    route_f1: Optional[float] = None
    route_continuity: float
    n_discontinuities: int


class PredictResponse(BaseModel):
    trip_id: str
    scorer: str                                  # which emission model produced this
    n_points: int
    gps_latlon: List[List[float]]                # [[lat, lon], ...] observed fixes
    predicted_route_arcs: List[int]              # classifier + HMM route (arc ids)
    predicted_route_latlon: List[List[List[float]]]   # one [[lat,lon],[lat,lon]] per arc
    classifier_only_latlon: List[List[List[float]]]
    per_point_arc: List[int]                     # HMM arc per GPS point, -1 = no candidate
    metrics_hmm: RouteMetrics
    metrics_classifier_only: RouteMetrics
    hmm_breaks: int
    points_without_candidates: int
    timings_ms: Dict[str, float]
