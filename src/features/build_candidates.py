"""Road candidates and geometry-based features for PathSnap.

Every candidate is a road ARC ID specific to its trip. A supplied .route file is a
sequence of traversed arc IDs, NOT a GPS-aligned label table. Training labels
below are explicit *heuristic pseudo-labels*: nearest arc on supplied route.
Do not claim these are authoritative one-to-one GPS ground truth.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from src.data.raw import load_trip,list_trips,to_local_xy

FEATURE_COLUMNS = [
    'distance_m', 'log_distance_m', 'arc_length_m', 'heading_alignment',
    'speed_mps', 'log_dt_prev_s', 'progress_on_arc',
    'endpoint_distance_m', 'out_degree_from', 'in_degree_to',
]


def closest_on_segments(p, a, b):
    v=b-a
    len2=np.sum(v*v,axis=1)
    alpha=np.clip(np.einsum('ij,ij->i',p-a,v)/np.maximum(len2,1e-8),0,1)
    near=a+alpha[:,None]*v
    d=np.linalg.norm(near-p,axis=1)
    return d,alpha


def make_spatial_index(a,b,ids=None):
    # Three anchor points mitigate failing to retrieve long arcs by midpoint alone.
    if ids is None: ids=np.arange(len(a))
    xy=np.vstack([a,b,0.5*(a+b)])
    arc_ids=np.tile(np.asarray(ids,dtype=np.int32),3)
    return cKDTree(xy),arc_ids


def closest_candidates(p, tree, index_to_arc, arcs_a, arcs_b, top_k=8, anchors_k=60):
    k=min(anchors_k,len(index_to_arc))
    _, ix=tree.query(p,k=k)
    candidates=np.unique(index_to_arc[np.atleast_1d(ix)])
    d, frac=closest_on_segments(p,arcs_a[candidates],arcs_b[candidates])
    chosen=np.argsort(d)[:top_k]
    return candidates[chosen],d[chosen],frac[chosen]


def build_trip_candidates(trip, top_k=8, anchors_k=60, obs_indices=None):
    """Return one row per (observation, candidate arc) and a label audit table.

    IMPORTANT: Reference-route .route is used ONLY to derive heuristic training
    labels and audit statistics; it is never used to construct candidate features.
    At inference, call this with use_labels=False (below).
    """
    return _build_candidates(trip,top_k,anchors_k,use_labels=True,obs_indices=obs_indices)


def build_inference_candidates(trip,top_k=8,anchors_k=60,obs_indices=None):
    """Generate features with NO reading or use of reference route labels."""
    return _build_candidates(trip,top_k,anchors_k,use_labels=False,obs_indices=obs_indices)


def _build_candidates(trip,top_k,anchors_k,use_labels,obs_indices=None):
    lon0=float(np.median(trip.track[:,0])); lat0=float(np.median(trip.track[:,1]))
    gps=to_local_xy(trip.track[:,:2],lon0,lat0)
    nodes=to_local_xy(trip.nodes,lon0,lat0)
    a=nodes[trip.arcs[:,0]]; b=nodes[trip.arcs[:,1]]
    lengths=np.linalg.norm(b-a,axis=1)
    out_deg=np.bincount(trip.arcs[:,0],minlength=len(nodes))
    in_deg=np.bincount(trip.arcs[:,1],minlength=len(nodes))
    tree,idx_to_arc=make_spatial_index(a,b)
    rtree,ridx_to_arc=(None,None)
    if use_labels:
        if trip.route is None:
            raise ValueError("Training pseudo-labels require a supplied .route; inference must not use labels")
        route_ids=np.unique(trip.route)  # Inference NEVER touches this branch.
        rtree,ridx_to_arc=make_spatial_index(a[route_ids],b[route_ids],route_ids)
    frames=[]; audits=[]
    prev=np.diff(gps,axis=0)
    dt=np.diff(trip.track[:,2])
    if obs_indices is None:
        obs_indices=np.arange(len(gps))
    else:
        obs_indices=np.asarray(obs_indices,dtype=int)
    # Work in batches: multi-point spatial queries and feature arithmetic are far
    # faster than hundreds of thousands of individual Python KD-tree calls.
    for start in range(0,len(obs_indices),256):
        stop=min(start+256,len(obs_indices))
        obs=obs_indices[start:stop]
        points=gps[obs]
        batch=len(points)
        _, ix=tree.query(points,k=min(anchors_k,len(idx_to_arc)))
        lookup=idx_to_arc[np.atleast_2d(ix)]
        if lookup.shape[0]!=batch:
            lookup=lookup.reshape(batch,-1)
        av=a[lookup];bv=b[lookup]
        v=bv-av
        den=np.maximum(np.sum(v*v,axis=2),1e-8)
        alpha=np.clip(np.sum((points[:,None,:]-av)*v,axis=2)/den,0,1)
        nearest=av+alpha[:,:,None]*v
        distances=np.linalg.norm(points[:,None,:]-nearest,axis=2)
        ordering=np.argsort(distances,axis=1)
        # A single road arc is represented by 3 anchors, so remove duplicates.
        chosen_ids=np.empty((batch,top_k),dtype=np.int32)
        chosen_dist=np.empty((batch,top_k),dtype=float)
        chosen_frac=np.empty((batch,top_k),dtype=float)
        for row in range(batch):
            seen=set();count=0
            for col in ordering[row]:
                idx=int(lookup[row,col])
                if idx in seen:continue
                seen.add(idx)
                chosen_ids[row,count]=idx
                chosen_dist[row,count]=distances[row,col]
                chosen_frac[row,count]=alpha[row,col]
                count+=1
                if count==top_k:break
            if count<top_k:
                # Fallback on unusually sparse networks; do exact geometry search.
                d_all,alpha_all=closest_on_segments(points[row],a,b)
                for idx in np.argsort(d_all):
                    if idx in seen:continue
                    seen.add(int(idx));chosen_ids[row,count]=int(idx)
                    chosen_dist[row,count]=d_all[idx];chosen_frac[row,count]=alpha_all[idx]
                    count+=1
                    if count==top_k:break
            if count<top_k: raise ValueError('Fewer available arcs than requested top-k')

        prev_ix=np.maximum(obs-1,0)
        vectors=prev[prev_ix] if len(prev) else np.zeros((batch,2))
        prev_gap=dt[prev_ix] if len(dt) else np.ones(batch)
        lengths_motion=np.linalg.norm(vectors,axis=1)
        speed=np.minimum(lengths_motion/np.maximum(prev_gap,1e-6),1000)
        heading=vectors/np.maximum(lengths_motion[:,None],1e-8)
        vectors_arc=b[chosen_ids]-a[chosen_ids]
        lengths_arc=lengths[chosen_ids]
        align=np.sum(heading[:,None,:]*vectors_arc,axis=2)/np.maximum(lengths_arc,1e-8)
        endpoints=np.minimum(np.linalg.norm(points[:,None,:]-a[chosen_ids],axis=2),
                             np.linalg.norm(points[:,None,:]-b[chosen_ids],axis=2))

        data={'split':np.repeat(trip.split,batch*top_k),
              'trip_id':np.repeat(trip.trip_id,batch*top_k),
              'obs_idx':np.repeat(obs,top_k), 'arc_id':chosen_ids.ravel(),
              'distance_m':chosen_dist.ravel(),
              'log_distance_m':np.log1p(chosen_dist).ravel(),
              'arc_length_m':lengths_arc.ravel(),
              'heading_alignment':align.ravel(),
              'speed_mps':np.repeat(speed,top_k),
              'log_dt_prev_s':np.repeat(np.log1p(np.maximum(prev_gap,0)),top_k),
              'progress_on_arc':chosen_frac.ravel(),
              'endpoint_distance_m':endpoints.ravel(),
              'out_degree_from':out_deg[trip.arcs[chosen_ids,0]].ravel(),
              'in_degree_to':in_deg[trip.arcs[chosen_ids,1]].ravel()}
        if use_labels:
            _, ri=rtree.query(points,k=min(anchors_k,len(ridx_to_arc)))
            refs=ridx_to_arc[np.atleast_2d(ri)]
            if refs.shape[0]!=batch:refs=refs.reshape(batch,-1)
            rav=a[refs];rbv=b[refs];rv=rbv-rav
            rfrac=np.clip(np.sum((points[:,None,:]-rav)*rv,axis=2)/np.maximum(np.sum(rv*rv,axis=2),1e-8),0,1)
            rdist=np.linalg.norm(points[:,None,:]-(rav+rfrac[:,:,None]*rv),axis=2)
            best=np.argmin(rdist,axis=1)
            target=refs[np.arange(batch),best]
            coverage=np.any(chosen_ids==target[:,None],axis=1)
            data['label']=(chosen_ids==target[:,None]).astype(np.int8).ravel()
            audits.append(pd.DataFrame({'split':trip.split,'trip_id':trip.trip_id,'obs_idx':obs,
                'pseudo_reference_arc_id':target,'pseudo_reference_distance_m':rdist[np.arange(batch),best],
                'reference_in_candidates':coverage.astype(int)}))
        frames.append(pd.DataFrame(data))
    return pd.concat(frames,ignore_index=True), (pd.concat(audits,ignore_index=True) if use_labels else pd.DataFrame())


def run(root, out, split,top_k,anchors_k,limit_trips,max_observations_per_trip=None):
    out.mkdir(parents=True,exist_ok=True)
    folders=list_trips(root,split)
    if limit_trips is not None:folders=folders[:limit_trips]
    all_candidates=[];all_audit=[]
    for i,folder in enumerate(folders,1):
        trip=load_trip(folder,split)
        obs=None
        if max_observations_per_trip is not None and max_observations_per_trip<len(trip.track):
            obs=np.unique(np.linspace(0,len(trip.track)-1,max_observations_per_trip).astype(int))
        features,audit=build_trip_candidates(trip,top_k,anchors_k,obs_indices=obs)
        all_candidates.append(features);all_audit.append(audit)
        print(f'[{i}/{len(folders)}] {split} {trip.trip_id}: {len(trip.track)} observations, {len(features)} candidates, pseudo-label coverage {audit.reference_in_candidates.mean():.1%}',flush=True)
    features=pd.concat(all_candidates,ignore_index=True)
    audits=pd.concat(all_audit,ignore_index=True)
    features.to_csv(out/f'{split}_candidates.csv',index=False)
    audits.to_csv(out/f'{split}_reference_audit.csv',index=False)
    print('Wrote',out/f'{split}_candidates.csv',flush=True)
    print('Mean pseudo-label candidate coverage:',audits.reference_in_candidates.mean(),flush=True)
    print('CAUTION: labels are a nearest-route-arc heuristic, not GPS-aligned ground truth.')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--split',choices=['train','student_test'],default='train')
    p.add_argument('--out',type=Path,default=Path('results'))
    p.add_argument('--top-k',type=int,default=8)
    p.add_argument('--anchors-k',type=int,default=60)
    p.add_argument('--limit-trips',type=int)
    p.add_argument('--max-observations-per-trip',type=int,help="Faster exploratory baseline only; full run should omit this")
    q=p.parse_args();run(q.data_root,q.out,q.split,q.top_k,q.anchors_k,q.limit_trips,q.max_observations_per_trip)
