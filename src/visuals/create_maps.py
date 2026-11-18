"""Produce real EDA plots and a Folium road network + GPS + reference route map."""
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from src.data.raw import load_trip


def create(data_root,trip_id,out,split='train',max_roads=2000):
    import folium
    out.mkdir(parents=True,exist_ok=True)
    trip=load_trip(data_root/split/trip_id,split)
    # Note Folium takes LAT,LON while input uses LON,LAT!
    gps=[[float(lat),float(lon)] for lon,lat in trip.track[:,:2]]
    f=folium.Map(location=gps[len(gps)//2],zoom_start=13,tiles='OpenStreetMap',control_scale=True)
    roadlayer=folium.FeatureGroup(name='Road network (subset)',show=True)
    steps=max(1,len(trip.arcs)//max_roads)
    for arc_id in range(0,len(trip.arcs),steps):
        x,y=trip.nodes[trip.arcs[arc_id]]
        folium.PolyLine([(float(x[1]),float(x[0])),(float(y[1]),float(y[0]))],color='#AAB0B8',opacity=0.4,weight=1).add_to(roadlayer)
    roadlayer.add_to(f)
    raw=folium.FeatureGroup(name='Raw GPS',show=True)
    for i in range(0,len(gps),max(1,len(gps)//250)):
        folium.CircleMarker(gps[i],radius=2,color='blue',fill=True,fill_opacity=0.7).add_to(raw)
    folium.PolyLine(gps,color='blue',weight=2,opacity=0.45).add_to(raw)
    raw.add_to(f)
    coords=[]
    for j,arc_id in enumerate(trip.route):
        s,t=trip.nodes[trip.arcs[arc_id]]
        # The chosen illustrative trip has a connected reference path; other
        # trips can contain small route discontinuities and need careful handling.
        if j==0:coords.append([float(s[1]),float(s[0])])
        coords.append([float(t[1]),float(t[0])])
    ref=folium.FeatureGroup(name='Supplied reference route',show=True)
    folium.PolyLine(coords,color='green',weight=4,opacity=0.8).add_to(ref)
    ref.add_to(f)
    folium.LayerControl().add_to(f)
    folium.LatLngPopup().add_to(f)
    f.fit_bounds([gps[0],gps[-1]])
    path=out/f'{split}_{trip_id}_exploration.html';f.save(str(path))

    fig, ax=plt.subplots(figsize=(9,6))
    for arc_id in range(0,len(trip.arcs),steps):
        pair=trip.nodes[trip.arcs[arc_id]]
        ax.plot(pair[:,0],pair[:,1],color='lightgray',lw=.5,alpha=.7)
    for j,arc_id in enumerate(trip.route):
        pair=trip.nodes[trip.arcs[arc_id]]
        ax.plot(pair[:,0],pair[:,1],color='green',lw=1.2,alpha=.85,label='Reference route' if j==0 else None)
    ax.scatter(trip.track[::max(1,len(trip.track)//500),0],trip.track[::max(1,len(trip.track)//500),1],s=7,color='blue',alpha=.4,label='Raw GPS')
    ax.set(xlabel='Longitude (degrees)',ylabel='Latitude (degrees)',title=f'PathSnap train trip {trip_id}: GPS vs roads')
    ax.legend(loc='best');fig.tight_layout()
    img=out/f'{split}_{trip_id}_gps_roads.png';fig.savefig(img,dpi=170);plt.close(fig)
    print('Created',path,'and',img)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--split',choices=['train','student_test'],default='train')
    p.add_argument('--trip',default='00000005')
    p.add_argument('--out',type=Path,default=Path('results'))
    opts=p.parse_args();create(opts.data_root,opts.trip,opts.out,opts.split)
