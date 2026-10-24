"""Conservative reference-to-frame diagnostic; never writes an active calibration."""
from pathlib import Path
import sys,math,hashlib
import cv2,numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
import meter_reader as meter

SETTINGS={'sift_features':1500,'ratio':.75,'minimum_matches':30,'minimum_fit_matches':20,
          'minimum_holdout_matches':10,'ransac_px':2.5,'ransac_confidence':.995,'ransac_iterations':2000,
          'minimum_inliers':12,'minimum_inlier_fraction':.6,'holdout_median_px':2.,'holdout_p90_px':4.,
          'minimum_source_hull_fraction':.25,'minimum_grid_cells':6,'anchor_hull_margin_diagonal':.05,
          'source_annulus_radius':[.65,1.20],'panel_iou':.5,'seed':42}

def project(points,h):
    return cv2.perspectiveTransform(np.asarray(points,float).reshape(1,-1,2),h)[0]

def iou(a,b):
    area=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
    union=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-area
    return area/union if union>0 else 0.


def quality_check(h,fit_source,fit_target,inliers,hold_source,hold_target,panel,target_size,proposals,anchors):
    failures=[];metrics={};inliers=np.asarray(inliers,bool).reshape(-1)
    src=np.asarray(fit_source,float);dst=np.asarray(fit_target,float)
    hs=np.asarray(hold_source,float);ht=np.asarray(hold_target,float)
    if len(src)<SETTINGS['minimum_fit_matches'] or len(hs)<SETTINGS['minimum_holdout_matches']:
        failures.append('match_split_counts')
    count=int(inliers.sum());fraction=count/max(1,len(src))
    metrics.update(fit_matches=len(src),holdout_matches=len(hs),inliers=count,inlier_fraction=fraction)
    if count<SETTINGS['minimum_inliers'] or fraction<SETTINGS['minimum_inlier_fraction']:failures.append('fit_consensus')
    err=np.linalg.norm(project(hs,h)-ht,axis=1)
    metrics.update(holdout_errors_px=err.tolist(),holdout_median_px=float(np.median(err)),holdout_p90_px=float(np.percentile(err,90)))
    inverse=np.linalg.inv(h);inv=np.linalg.norm(project(ht,inverse)-hs,axis=1)
    metrics['holdout_inverse_median_source_px']=float(np.median(inv))
    if (not np.isfinite(err).all() or metrics['holdout_median_px']>SETTINGS['holdout_median_px'] or
        metrics['holdout_p90_px']>SETTINGS['holdout_p90_px']):failures.append('heldout_reprojection')
    used=src[inliers];x1,y1,x2,y2=panel;area=(x2-x1)*(y2-y1)
    hull=cv2.convexHull(used.astype(np.float32)) if len(used)>=3 else None
    coverage=float(cv2.contourArea(hull))/area if hull is not None else 0.
    cells=np.clip(np.floor((used-[x1,y1])/[x2-x1,y2-y1]*3),0,2).astype(int)
    occupied=len({tuple(p) for p in cells})
    metrics.update(source_hull_fraction=coverage,source_grid_cells=occupied)
    if coverage<SETTINGS['minimum_source_hull_fraction'] or occupied<SETTINGS['minimum_grid_cells']:
        failures.append('inlier_coverage')
    distances=[float(cv2.pointPolygonTest(hull,(float(p[0]),float(p[1])),True)) for p in anchors] if hull is not None else [-1e9]*len(anchors)
    metrics['anchor_distance_to_inlier_hull_source_px']=distances
    if min(distances)<-math.hypot(x2-x1,y2-y1)*SETTINGS['anchor_hull_margin_diagonal']:
        failures.append('unsupported_scale_anchors')
    corners=np.array([[x1,y1],[x2,y1],[x2,y2],[x1,y2]],float);q=project(corners,h)
    denominator=np.column_stack([corners,np.ones(4)])@h[2]
    tw,th=target_size
    # Homogeneous normalization can turn an exact edge into width + 1e-13.
    # Numerical roundoff only; this is not a pixel-sized geometric margin.
    epsilon=np.finfo(float).eps*max(tw,th)*1000
    signed=float(np.sum(q[:,0]*np.roll(q[:,1],-1)-q[:,1]*np.roll(q[:,0],-1))/2)
    good=(np.isfinite(q).all() and np.all(denominator>1e-6) and signed>1 and
          cv2.isContourConvex(q.astype(np.float32)) and (q[:,0]>=-epsilon).all() and (q[:,0]<=tw+epsilon).all() and
          (q[:,1]>=-epsilon).all() and (q[:,1]<=th+epsilon).all())
    if not good:failures.append('projected_panel_geometry')
    bound=[float(q[:,0].min()),float(q[:,1].min()),float(q[:,0].max()),float(q[:,1].max())]
    overlaps=[iou(bound,b) for b in proposals];matched=[i for i,v in enumerate(overlaps) if v>=SETTINGS['panel_iou']]
    metrics.update(projected_panel_polygon=q.tolist(),projected_panel_box=bound,panel_ious=overlaps,matched_panel_indices=matched)
    if len(matched)!=1:failures.append('unique_coarse_panel_binding')
    return {'failures':failures,'metrics':metrics}

def register(reference_raw,target,calibration,panel,proposals):
    out={'status':'needs_review','reason':None,'numeric_value':None,'settings':dict(SETTINGS),
         'homography':None,'matches':[],'quality':None,'profile_written':False}
    try:
        meter.validate_calibration_image(reference_raw,calibration)
        reference=cv2.imdecode(np.frombuffer(reference_raw,np.uint8),1)
        if reference is None or calibration.get('image_size')!=[reference.shape[1],reference.shape[0]]:
            raise ValueError('Reference source dimensions mismatch.')
        if not isinstance(target,np.ndarray) or target.ndim!=3 or target.shape[2]!=3 or target.dtype!=np.uint8:
            raise ValueError('Invalid target image.')
        if (calibration.get('unit')!='div' or calibration.get('scale_mode')!='full_circle_uniform' or
            calibration.get('needle_role')!='outer_long_hand'):raise ValueError('Unsupported reference mode.')
        center,radius,_=meter.fit_calibration_ring(calibration['ticks'])
        meter.full_circle_calibration_check(center,calibration['ticks'],calibration['direction'])
        rh,rw=reference.shape[:2]
        if not(0<=panel[0]<panel[2]<=rw and 0<=panel[1]<panel[3]<=rh):raise ValueError('Invalid reference panel.')
        yy,xx=np.indices((rh,rw));rr=np.hypot(xx-center[0],yy-center[1])
        mask=((rr>=radius*.65)&(rr<=radius*1.20)&(xx>=panel[0])&(xx<=panel[2])&
              (yy>=panel[1])&(yy<=panel[3])).astype(np.uint8)*255
        cv2.setRNGSeed(SETTINGS['seed']);cv2.setNumThreads(1)
        sift=cv2.SIFT_create(nfeatures=SETTINGS['sift_features'])
        sk,sd=sift.detectAndCompute(cv2.cvtColor(reference,cv2.COLOR_BGR2GRAY),mask)
        tk,td=sift.detectAndCompute(cv2.cvtColor(target,cv2.COLOR_BGR2GRAY),None)
        out.update(reference_features=len(sk),target_features=len(tk),reference_mask_sha256=hashlib.sha256(mask.tobytes()).hexdigest())
        if sd is None or td is None or min(len(sd),len(td))<2:raise ValueError('Insufficient local features.')
        bf=cv2.BFMatcher(cv2.NORM_L2)
        def filtered(a,b):
            return {pair[0].queryIdx:pair[0].trainIdx for pair in bf.knnMatch(a,b,k=2)
                    if len(pair)==2 and pair[0].distance<SETTINGS['ratio']*pair[1].distance}
        forward=filtered(sd,td);backward=filtered(td,sd)
        pairs=[(i,j) for i,j in forward.items() if backward.get(j)==i]
        pairs.sort(key=lambda p:(sk[p[0]].pt[0],sk[p[0]].pt[1],p[0],p[1]))
        out['matches']=[{'reference_point':list(sk[i].pt),'target_point':list(tk[j].pt),
                         'split':'holdout' if k%3==0 else 'fit'} for k,(i,j) in enumerate(pairs)]
        out['mutual_matches']=len(pairs)
        if len(pairs)<SETTINGS['minimum_matches']:raise ValueError('Insufficient mutual ratio-filtered correspondences.')
        src=np.array([sk[i].pt for i,j in pairs],float);dst=np.array([tk[j].pt for i,j in pairs],float)
        hold=np.arange(len(pairs))%3==0
        if (~hold).sum()<20 or hold.sum()<10:raise ValueError('Insufficient fit/heldout matches.')
        h,inliers=cv2.findHomography(src[~hold],dst[~hold],cv2.RANSAC,SETTINGS['ransac_px'],
            maxIters=SETTINGS['ransac_iterations'],confidence=SETTINGS['ransac_confidence'])
        if h is None or not np.isfinite(h).all() or abs(h[2,2])<1e-9 or abs(np.linalg.det(h))<1e-10:
            raise ValueError('No nonsingular homography.')
        h=h/h[2,2];out['homography']=h.tolist();out['fit_inliers']=inliers.reshape(-1).astype(bool).tolist()
        quality=quality_check(h,src[~hold],dst[~hold],inliers,src[hold],dst[hold],panel,
            [target.shape[1],target.shape[0]],proposals,[t['point'] for t in calibration['ticks']])
        out['quality']=quality
        out['projected_ticks_for_diagnosis']=[{'value':t['value'],'point':p.tolist()}
            for t,p in zip(calibration['ticks'],project([t['point'] for t in calibration['ticks']],h))]
        if quality['failures']:out['reason']='registration_quality_rejected'
        else:out.update(status='registration_candidate',reason=None)
    except (ValueError,KeyError,TypeError,RuntimeError,np.linalg.LinAlgError,cv2.error) as error:
        out['reason']=str(error)
    return out
