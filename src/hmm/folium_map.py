"""Interactive Folium map: raw GPS, road network, reference route, predicted route(s).

Each layer has its own colour, a toggle in the layer control and a legend.
Coordinates in the data are (lon, lat); Folium wants (lat, lon) -- converted here.
Only the part of the network near the GPS trace is drawn (full networks can have
>100k arcs and would make the HTML unusable).
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Sequence

import folium
import numpy as np

from .graph import RoadGraph

LEGEND = """
<div style="position: fixed; bottom: 28px; left: 28px; z-index: 9999; background: white;
 padding: 8px 12px; border: 1px solid #999; border-radius: 6px; font-size: 12px; line-height: 1.5">
<b>PathSnap</b><br>
<span style="color:#1f77b4">&#9679;</span> Raw GPS fixes<br>
<span style="color:#999999">&#9473;</span> Road network (near trace)<br>
<span style="color:#2ca02c">&#9473;</span> Reference route<br>
<span style="color:#ff7f0e">&#9473;</span> {cls_label} (dashed, toggle in layers)<br>
<span style="color:#d62728">&#9473;</span> Predicted route ({cls_short} + HMM)
</div>"""


def arcs_to_latlon(graph: RoadGraph, arc_ids: Sequence[int]):
    """List of [(lat, lon), (lat, lon)] segments, one per arc."""
    ll = graph.nodes_lonlat
    return [[(ll[graph.arc_from[a], 1], ll[graph.arc_from[a], 0]),
             (ll[graph.arc_to[a], 1], ll[graph.arc_to[a], 0])] for a in arc_ids]


def make_route_map(
    graph: RoadGraph,
    track_lonlat: np.ndarray,
    predicted_arcs: Optional[Sequence[int]] = None,
    reference_arcs: Optional[Sequence[int]] = None,
    extra_routes: Optional[Dict[str, Sequence[int]]] = None,   # name -> arcs (e.g. classifier-only)
    out_html: Optional[str | Path] = None,
    network_margin_m: float = 300.0,
    max_network_arcs: int = 6000,
    title: str = "",
    classifier_label: str = "Classifier-only prediction",
) -> folium.Map:
    lon, lat = track_lonlat[:, 0], track_lonlat[:, 1]
    m = folium.Map(location=[float(lat.mean()), float(lon.mean())], zoom_start=13,
                   tiles="OpenStreetMap", control_scale=True)
    m.fit_bounds([[float(lat.min()), float(lon.min())], [float(lat.max()), float(lon.max())]])

    # network near the trace (bounding box + margin, in metres)
    xy = graph.to_xy(track_lonlat)
    lo, hi = xy.min(0) - network_margin_m, xy.max(0) + network_margin_m
    mid = (graph.nodes_xy[graph.arc_from] + graph.nodes_xy[graph.arc_to]) / 2
    near = np.where((mid >= lo).all(1) & (mid <= hi).all(1))[0]
    if len(near) > max_network_arcs:
        near = near[np.linspace(0, len(near) - 1, max_network_arcs).astype(int)]
    net = folium.FeatureGroup(name="Road network", show=True)
    folium.PolyLine(arcs_to_latlon(graph, near), color="#999999", weight=1, opacity=0.6).add_to(net)
    net.add_to(m)

    gps = folium.FeatureGroup(name="Raw GPS", show=True)
    for k in range(0, len(lat), max(1, len(lat) // 1500)):   # thin very long traces
        folium.CircleMarker([float(lat[k]), float(lon[k])], radius=2, color="#1f77b4",
                            fill=True, fill_opacity=0.7, weight=1).add_to(gps)
    gps.add_to(m)

    def add(name, arcs, color, weight, show=True, dash=None):
        fg = folium.FeatureGroup(name=name, show=show)
        folium.PolyLine(arcs_to_latlon(graph, arcs), color=color, weight=weight, opacity=0.85,
                        dash_array=dash, tooltip=name).add_to(fg)
        fg.add_to(m)

    if reference_arcs is not None:
        add("Reference route", reference_arcs, "#2ca02c", 6)
    for name, arcs in (extra_routes or {}).items():
        add(name, arcs, "#ff7f0e", 4, show=False, dash="6 6")
    if predicted_arcs is not None:
        add("Predicted route (HMM)", predicted_arcs, "#d62728", 3)

    folium.Marker([float(lat[0]), float(lon[0])], tooltip="Start",
                  icon=folium.Icon(color="green", icon="play")).add_to(m)
    folium.Marker([float(lat[-1]), float(lon[-1])], tooltip="End",
                  icon=folium.Icon(color="red", icon="stop")).add_to(m)
    if title:
        m.get_root().html.add_child(folium.Element(
            f'<h4 style="position:fixed;top:8px;left:60px;z-index:9999;background:white;'
            f'padding:4px 10px;border-radius:6px">{title}</h4>'))
    m.get_root().html.add_child(folium.Element(
        LEGEND.format(cls_label=classifier_label, cls_short=classifier_label.split(" ")[0])))
    folium.LayerControl(collapsed=False).add_to(m)
    if out_html:
        Path(out_html).parent.mkdir(parents=True, exist_ok=True)
        m.save(str(out_html))
    return m
