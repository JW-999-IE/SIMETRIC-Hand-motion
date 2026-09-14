from __future__ import annotations
import argparse, json, math, re
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter, find_peaks, welch

EXCLUDED = {"P7","P8","P14"}
USABLE = {"validated","partial_start","partial_end","partial_both"}
PALM_LMS = ["WRIST","INDEX_MCP","MIDDLE_MCP","RING_MCP","PINKY_MCP"]
LM_INDEX = {
"WRIST":0,"THUMB_CMC":1,"THUMB_MCP":2,"THUMB_IP":3,"THUMB_TIP":4,
"INDEX_MCP":5,"INDEX_PIP":6,"INDEX_DIP":7,"INDEX_TIP":8,
"MIDDLE_MCP":9,"MIDDLE_PIP":10,"MIDDLE_DIP":11,"MIDDLE_TIP":12,
"RING_MCP":13,"RING_PIP":14,"RING_DIP":15,"RING_TIP":16,
"PINKY_MCP":17,"PINKY_PIP":18,"PINKY_DIP":19,"PINKY_TIP":20,
}

def p_simple(v):
    m=re.search(r"(\d+)",str(v)); return f"P{int(m.group(1))}" if m else str(v)

def p_norm(v):
    m=re.search(r"(\d+)",str(v)); return f"P{int(m.group(1)):02d}" if m else str(v)

def num(s): return pd.to_numeric(s,errors="coerce")

def read_parts(folder:Path)->pd.DataFrame:
    paths=sorted(folder.glob("part-*.parquet"))
    if not paths: paths=sorted(folder.glob("part-*.csv.gz"))
    if not paths: paths=sorted(folder.glob("part-*.csv"))
    if not paths: return pd.DataFrame()
    out=[]
    for p in paths:
        if p.suffix==".parquet": out.append(pd.read_parquet(p))
        elif p.name.endswith(".csv.gz"): out.append(pd.read_csv(p,compression="gzip"))
        else: out.append(pd.read_csv(p))
    return pd.concat(out,ignore_index=True) if out else pd.DataFrame()

def read_reconstructed(folder:Path)->pd.DataFrame:
    for p in [folder/"reconstructed.parquet",folder/"reconstructed.csv.gz",folder/"reconstructed.csv"]:
        if p.exists():
            return pd.read_parquet(p) if p.suffix==".parquet" else pd.read_csv(p)
    return pd.DataFrame()

def angle(a,b,c):
    ba=a-b; bc=c-b
    den=np.linalg.norm(ba)*np.linalg.norm(bc)
    if den<=0: return np.nan
    return float(np.degrees(np.arccos(np.clip(np.dot(ba,bc)/den,-1,1))))

def corrected_gopro_label(raw):
    s=str(raw).strip().lower()
    if s=="left": return "right"
    if s=="right": return "left"
    return ""

def reconstruct_gopro(df:pd.DataFrame)->pd.DataFrame:
    if df.empty: return df
    for c in ["frame","video_time_sec","hand_index","x_WRIST","y_WRIST","z_WRIST"]:
        if c in df.columns: df[c]=num(df[c])
    prev={}
    labels={}
    for frame,g in df.groupby("frame",sort=True):
        cand=[]
        for idx,r in g.iterrows():
            wrist=np.array([r.get("x_WRIST",np.nan),r.get("y_WRIST",np.nan),r.get("z_WRIST",0.0)],float)
            cand.append((idx,int(r.get("hand_index",0)),corrected_gopro_label(r.get("handedness","")),wrist,float(r.get("handedness_score",0.5))))
        def cost(c,target):
            _,_,lab,w,score=c
            lc=0 if lab==target else (score if lab else .5)
            d=0 if target not in prev or not np.isfinite(w).all() else np.linalg.norm(w-prev[target])
            return .30*lc+d
        if len(cand)==1:
            c=cand[0]; lab=c[2]
            target=lab if lab else min(["left","right"],key=lambda h:cost(c,h))
            labels[c[0]]=target
            if np.isfinite(c[3]).all(): prev[target]=c[3]
        elif len(cand)>=2:
            cand=sorted(cand,key=lambda c:-c[4])[:2]
            a,b=cand
            if cost(a,"left")+cost(b,"right") <= cost(a,"right")+cost(b,"left"):
                ass=[(a,"left"),(b,"right")]
            else: ass=[(a,"right"),(b,"left")]
            for c,target in ass:
                labels[c[0]]=target
                if np.isfinite(c[3]).all(): prev[target]=c[3]
    df=df.copy()
    df["anatomical_hand"]=[labels.get(i,"unresolved") for i in df.index]
    return df

def gopro_palm_track(df,hand,start,end):
    if df.empty:return pd.DataFrame()
    d=df[(df["anatomical_hand"]==hand)&num(df["video_time_sec"]).between(start,end)].copy()
    if d.empty:return pd.DataFrame()
    xs=[f"x_{lm}" for lm in PALM_LMS if f"x_{lm}" in d.columns]
    ys=[f"y_{lm}" for lm in PALM_LMS if f"y_{lm}" in d.columns]
    if not xs or not ys:return pd.DataFrame()
    x=d[xs].apply(num).mean(axis=1); y=d[ys].apply(num).mean(axis=1)
    out=pd.DataFrame({"time":num(d["video_time_sec"]),"x":x,"y":y,"frame":num(d["frame"])})
    return out.dropna().sort_values("time")

def cae_palm_track(df,hand,start,end):
    if df.empty:return pd.DataFrame()
    d=df.copy()
    d["timestamp_sec"]=num(d["timestamp_sec"]); d["landmark_index"]=num(d["landmark_index"])
    d=d[(d["anatomical_hand"].astype(str).str.lower()==hand)&d["timestamp_sec"].between(start,end)]
    if d.empty:return pd.DataFrame()
    ids=[LM_INDEX[x] for x in PALM_LMS]
    d=d[d["landmark_index"].isin(ids)]
    for c in ["x","y"]: d[c]=num(d[c])
    g=d.groupby(["frame_index","timestamp_sec"])[["x","y"]].mean().reset_index()
    return g.rename(columns={"timestamp_sec":"time","frame_index":"frame"}).dropna().sort_values("time")

def segments(track,max_gap=.25):
    if len(track)<3:return []
    t=track.time.to_numpy(float); cuts=np.where(np.diff(t)>max_gap)[0]+1
    return [x for x in np.split(track,np.array(cuts)) if len(x)>=5]

def smooth_xy(seg,window_sec=.20):
    t=seg.time.to_numpy(float); x=seg.x.to_numpy(float); y=seg.y.to_numpy(float)
    dt=np.median(np.diff(t))
    if not np.isfinite(dt) or dt<=0:return t,x,y,0
    w=max(5,int(round(window_sec/dt)))
    if w%2==0:w+=1
    if w>=len(t):w=len(t)-1 if len(t)%2==0 else len(t)
    if w<5:return t,x,y,1/dt
    x=savgol_filter(x,w,2,mode="interp"); y=savgol_filter(y,w,2,mode="interp")
    return t,x,y,1/dt

def sparc_metric(speed,fs,fc=10):
    speed=np.asarray(speed,float); speed=speed[np.isfinite(speed)]
    if len(speed)<16 or fs<=0:return np.nan
    speed=np.maximum(speed,0)
    n=2**int(np.ceil(np.log2(len(speed))))
    mag=np.abs(np.fft.rfft(speed-speed.mean(),n=n))
    freq=np.fft.rfftfreq(n,1/fs)
    mask=(freq>=0)&(freq<=min(fc,fs/2*.95))
    freq=freq[mask]; mag=mag[mask]
    if len(freq)<3 or np.max(mag)<=0:return np.nan
    mag=mag/np.max(mag); fn=freq/freq[-1]
    return -float(np.sum(np.sqrt(np.diff(fn)**2+np.diff(mag)**2)))

def ldlj_metric(t,x,y):
    if len(t)<8:return np.nan
    duration=t[-1]-t[0]
    if duration<=0:return np.nan
    vx=np.gradient(x,t); vy=np.gradient(y,t)
    ax=np.gradient(vx,t); ay=np.gradient(vy,t)
    jx=np.gradient(ax,t); jy=np.gradient(ay,t)
    jerk2=jx*jx+jy*jy
    path=np.sum(np.sqrt(np.diff(x)**2+np.diff(y)**2))
    if path<=0:return np.nan
    integral=np.trapz(jerk2,t)
    val=(duration**5/path**2)*integral
    return -float(np.log(max(val,1e-12)))

def motion_metrics(track,attempt_start,still_thr=.03,onset_thr=.05,min_still=.5,onset_hold=.25,max_gap=.25):
    if track.empty:return {},pd.DataFrame()
    track=track.drop_duplicates("time").sort_values("time")
    segs=segments(track,max_gap)
    if not segs:return {},pd.DataFrame()
    all_v=[];all_a=[];sparcs=[];ldljs=[];weights=[];peak_count=0;still_total=0;still_eps=0;longest=0
    traj=[]
    for seg in segs:
        t,x,y,fs=smooth_xy(seg)
        if fs<=0 or len(t)<5:continue
        vx=np.gradient(x,t);vy=np.gradient(y,t);v=np.sqrt(vx*vx+vy*vy)
        ax=np.gradient(vx,t);ay=np.gradient(vy,t);a=np.sqrt(ax*ax+ay*ay)
        all_v.extend(v);all_a.extend(a)
        s=sparc_metric(v,fs); l=ldlj_metric(t,x,y)
        dur=t[-1]-t[0];weights.append(dur);sparcs.append(s);ldljs.append(l)
        peaks,_=find_peaks(v,prominence=max(np.nanmedian(v)*.25,1e-4),distance=max(1,int(fs*.15)));peak_count+=len(peaks)
        low=v<still_thr
        starts=[]; st=None
        for i,b in enumerate(low):
            if b and st is None:st=i
            if (not b or i==len(low)-1) and st is not None:
                en=i if not b else i+1; dd=t[min(en-1,len(t)-1)]-t[st]
                if dd>=min_still: starts.append(dd)
                st=None
        still_total+=sum(starts);still_eps+=len(starts);longest=max([longest]+starts)
        traj.append(pd.DataFrame({"time":t,"x":x,"y":y,"speed":v,"accel":a}))
    if not traj:return {},pd.DataFrame()
    full=pd.concat(traj,ignore_index=True).sort_values("time")
    x0,y0=full.iloc[0][["x","y"]];x1,y1=full.iloc[-1][["x","y"]]
    disp=float(np.hypot(x1-x0,y1-y0))
    path=0
    for seg in traj:path+=float(np.sum(np.hypot(np.diff(seg.x),np.diff(seg.y))))
    total_dur=max(full.time.max()-full.time.min(),1e-9)
    weights=np.array(weights,float)
    def wavg(vals):
        vals=np.array(vals,float);m=np.isfinite(vals)&np.isfinite(weights)
        return float(np.average(vals[m],weights=weights[m])) if m.any() else np.nan
    v=np.array(all_v,float);a=np.array(all_a,float)
    # movement onset: first sustained > threshold in any continuous segment
    onset=np.nan
    for seg in traj:
        above=seg.speed.to_numpy(float)>onset_thr
        fs=1/np.median(np.diff(seg.time)) if len(seg)>2 else 0
        need=max(1,int(round(onset_hold*fs))) if fs>0 else 1
        run=0
        for i,b in enumerate(above):
            run=run+1 if b else 0
            if run>=need:
                onset=float(seg.time.iloc[i-need+1]-attempt_start);break
        if np.isfinite(onset):break
    # tremor on longest segment
    longest_seg=max(traj,key=lambda q:q.time.iloc[-1]-q.time.iloc[0])
    tremor_rms=np.nan;tremor_hz=np.nan
    if len(longest_seg)>=32:
        t=longest_seg.time.to_numpy(float);fs=1/np.median(np.diff(t))
        if fs>=25:
            x=longest_seg.x.to_numpy(float);y=longest_seg.y.to_numpy(float)
            w=max(5,int(round(.5*fs)));w+=1-w%2
            if w<len(x):
                tx=savgol_filter(x,w,2,mode="interp");ty=savgol_filter(y,w,2,mode="interp")
                rx=x-tx;ry=y-ty;tremor_rms=float(np.sqrt(np.mean(rx*rx+ry*ry)))
                f,px=welch(rx,fs=fs,nperseg=min(len(rx),256));_,py=welch(ry,fs=fs,nperseg=min(len(ry),256))
                band=(f>=4)&(f<=min(12,fs/2*.9))
                if band.any(): tremor_hz=float(f[band][np.argmax((px+py)[band])])
    return {
        "attempt_duration_sec":total_dur,"start_x_norm":x0,"start_y_norm":y0,"end_x_norm":x1,"end_y_norm":y1,
        "displacement_norm":disp,"path_length_norm":path,"path_efficiency":disp/path if path>0 else np.nan,
        "mean_speed_norm_s":float(np.nanmean(v)),"peak_speed_norm_s":float(np.nanmax(v)),
        "mean_accel_norm_s2":float(np.nanmean(a)),"rms_accel_norm_s2":float(np.sqrt(np.nanmean(a*a))),
        "peak_accel_norm_s2":float(np.nanmax(a)),"sparc":wavg(sparcs),"log_dimensionless_jerk":wavg(ldljs),
        "velocity_peak_count":peak_count,"stillness_total_sec":still_total,"stillness_fraction":still_total/total_dur,
        "stillness_episode_count":still_eps,"longest_stillness_sec":longest,
        "tremor_rms_norm":tremor_rms,"tremor_dominant_hz":tremor_hz,"movement_onset_delay_sec":onset,
    },full

def resample100(full):
    if full.empty:return pd.DataFrame()
    t=full.time.to_numpy(float);x=full.x.to_numpy(float);y=full.y.to_numpy(float)
    if t[-1]<=t[0]:return pd.DataFrame()
    tn=(t-t[0])/(t[-1]-t[0]);target=np.linspace(0,1,100)
    return pd.DataFrame({"trajectory_point":np.arange(1,101),"normalized_time_0_1":target,
                         "x_norm":np.interp(target,tn,x),"y_norm":np.interp(target,tn,y)})

def world_points_gopro(row):
    pts={}
    for lm in LM_INDEX:
        cols=[f"x_world_{lm}",f"y_world_{lm}",f"z_world_{lm}"]
        if all(c in row.index for c in cols):
            v=np.array([row[c] for c in cols],float)
            if np.isfinite(v).all():pts[LM_INDEX[lm]]=v
    return pts

def grip_candidate(pts):
    req=[4,8,12,16,20,5,17,6,10,14,18,3,2]
    if not all(i in pts for i in req):return "UNCERTAIN",np.nan
    scale=np.linalg.norm(pts[5]-pts[17])
    if scale<=0:return "UNCERTAIN",np.nan
    ti=np.linalg.norm(pts[4]-pts[8])/scale
    ap=np.linalg.norm(pts[4]-pts[20])/scale
    flex=np.nanmean([angle(pts[a],pts[b],pts[c]) for a,b,c in [(5,6,8),(9,10,12),(13,14,16),(17,18,20)]])
    th=angle(pts[2],pts[3],pts[4])
    if ap>1.8 and flex>155:return "NOT_GRIPPING_CANDIDATE",.55
    if np.isfinite(th) and th<115 and np.linalg.norm(pts[4]-pts[5])/scale<1.1:return "TC_CANDIDATE",.55
    if ti<.45:return "IPG_CANDIDATE",.60
    if ti<1.1 and flex<155:return "EPG_CANDIDATE",.50
    return "UNCERTAIN",.40

def grip_segments_gopro(df,hand,start,end,min_seg=.3):
    d=df[(df.anatomical_hand==hand)&num(df.video_time_sec).between(start,end)].copy()
    rec=[]
    for _,r in d.iterrows():
        try:pts=world_points_gopro(r)
        except:continue
        lab,conf=grip_candidate(pts)
        rec.append((float(r.video_time_sec),lab,conf))
    return collapse_grips(rec,min_seg)

def grip_segments_cae(df,hand,start,end,min_seg=.3):
    d=df.copy();d["timestamp_sec"]=num(d.timestamp_sec);d["landmark_index"]=num(d.landmark_index)
    d=d[(d.anatomical_hand.astype(str).str.lower()==hand)&d.timestamp_sec.between(start,end)]
    rec=[]
    for t,g in d.groupby("timestamp_sec"):
        pts={}
        for _,r in g.iterrows():
            idx=int(r.landmark_index);v=np.array([r.world_x,r.world_y,r.world_z],float)
            if np.isfinite(v).all():pts[idx]=v
        lab,conf=grip_candidate(pts);rec.append((float(t),lab,conf))
    return collapse_grips(rec,min_seg)

def collapse_grips(rec,min_seg):
    if not rec:return pd.DataFrame()
    rec=sorted(rec);times=np.array([r[0] for r in rec]);labs=[r[1] for r in rec];confs=np.array([r[2] for r in rec],float)
    # 7-sample rolling mode
    sm=[]
    for i in range(len(labs)):
        a=max(0,i-3);b=min(len(labs),i+4)
        vals=pd.Series(labs[a:b]);sm.append(vals.mode().iloc[0] if len(vals.mode()) else labs[i])
    segs=[];st=0
    for i in range(1,len(sm)+1):
        if i==len(sm) or sm[i]!=sm[st]:
            dur=times[i-1]-times[st]
            if dur>=min_seg:
                segs.append({"segment_start_sec":times[st],"segment_end_sec":times[i-1],
                             "segment_duration_sec":dur,"grip_candidate":sm[st],
                             "candidate_confidence":float(np.nanmean(confs[st:i]))})
            st=i
    return pd.DataFrame(segs)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--root",required=True)
    ap.add_argument("--mapping",required=True)
    ap.add_argument("--output-dir",required=True)
    ap.add_argument("--stillness-threshold",type=float,default=.03)
    ap.add_argument("--onset-threshold",type=float,default=.05)
    ap.add_argument("--max-gap-sec",type=float,default=.25)
    args=ap.parse_args()
    root=Path(args.root);out=Path(args.output_dir);out.mkdir(parents=True,exist_ok=True)
    mp=pd.read_csv(args.mapping,dtype=str).fillna("")
    mp=mp[~mp.participant.map(p_simple).isin(EXCLUDED)]
    mp=mp[mp.mapping_status.isin(USABLE)].copy()
    metrics=[];traj_rows=[];grip_rows=[]
    # process grouped by source/video once
    for (source,vid),gmap in mp.groupby(["chosen_source","selected_video_id"]):
        if not vid:continue
        if str(source).startswith("gopro_"):
            data=reconstruct_gopro(read_parts(root/"landmarks"/vid))
            source_kind="gopro"
        elif source=="cae_hand":
            data=read_reconstructed(root/"cae"/"analysis"/"reconstructed"/vid)
            source_kind="cae"
        else:continue
        if data.empty:
            for _,m in gmap.iterrows():
                metrics.append({"attempt_id":m.attempt_id,"participant":p_norm(m.participant),"hand":m.hand,
                                "motion_source":source,"video_id":vid,"qc_status":"FAIL_NO_SOURCE_DATA"})
            continue
        for _,m in gmap.iterrows():
            start=float(m.local_start_sec);end=float(m.local_end_sec);hand=str(m.hand).lower()
            track=gopro_palm_track(data,hand,start,end) if source_kind=="gopro" else cae_palm_track(data,hand,start,end)
            met,full=motion_metrics(track,start,args.stillness_threshold,args.onset_threshold,max_gap=args.max_gap_sec)
            coverage=len(track)/(max((end-start),1e-9)*(1/np.median(np.diff(track.time)) if len(track)>2 else 1)) if len(track)>2 else np.nan
            row={"attempt_id":m.attempt_id,"participant":p_norm(m.participant),"attempt_sequence":m.attempt_sequence,
                 "round":m["round"],"device":m.device,"hand":hand,"motion_source":source,"video_id":vid,
                 "mapping_status":m.mapping_status,"coverage":m.coverage,"local_start_sec":start,"local_end_sec":end,
                 "tracking_coverage":min(float(coverage),1.0) if np.isfinite(coverage) else np.nan,
                 "qc_status":"PASS" if len(track)>=10 else "WARN_LOW_TRACK","review_note":m.review_note}
            row.update(met)
            gs=grip_segments_gopro(data,hand,start,end) if source_kind=="gopro" else grip_segments_cae(data,hand,start,end)
            if not gs.empty:
                row["initial_grip_candidate"]=gs.iloc[0].grip_candidate
                row["final_grip_candidate"]=gs.iloc[-1].grip_candidate
                row["dominant_grip_candidate"]=gs.groupby("grip_candidate").segment_duration_sec.sum().idxmax()
                row["grip_change_count"]=max(len(gs)-1,0)
                for j,s in gs.iterrows():
                    rr={"attempt_id":m.attempt_id,"participant":p_norm(m.participant),"hand":hand,
                        "motion_source":source,"video_id":vid,"segment_number":j+1,**s.to_dict(),
                        "validation_status":"screening_only_manual_validation_required"}
                    grip_rows.append(rr)
            row["grip_validation_status"]="screening_only_manual_validation_required"
            metrics.append(row)
            r=resample100(full)
            if not r.empty:
                r.insert(0,"video_id",vid);r.insert(0,"motion_source",source);r.insert(0,"hand",hand)
                r.insert(0,"participant",p_norm(m.participant));r.insert(0,"attempt_id",m.attempt_id)
                traj_rows.append(r)
    md=pd.DataFrame(metrics);td=pd.concat(traj_rows,ignore_index=True) if traj_rows else pd.DataFrame()
    gd=pd.DataFrame(grip_rows)
    md.to_csv(out/"attempt_hand_metrics_source_aware.csv",index=False)
    td.to_csv(out/"trajectory_100pt.csv",index=False)
    gd.to_csv(out/"grip_segments_candidates.csv",index=False)
    params={"stillness_threshold_norm_s":args.stillness_threshold,"onset_threshold_norm_s":args.onset_threshold,
            "max_tracking_gap_sec":args.max_gap_sec,"grip_status":"screening_only_manual_validation_required",
            "tremor_status":"exploratory_tracking_jitter_sensitive"}
    (out/"analysis_parameters.json").write_text(json.dumps(params,indent=2),encoding="utf-8")
    print(f"Wrote {len(md)} attempt-hand rows, {len(td)} trajectory rows, {len(gd)} grip segments to {out}")
if __name__=="__main__":main()
