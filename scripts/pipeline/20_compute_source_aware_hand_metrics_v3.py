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

def motion_metrics(track,attempt_start,mapped_duration,still_thr=.03,onset_thr=.05,min_still=.5,onset_hold=.25,max_gap=.25):
    if track.empty:return {},pd.DataFrame()
    track=track.drop_duplicates("time").sort_values("time")
    segs=segments(track,max_gap)
    if not segs:return {},pd.DataFrame()

    all_v=[];all_a=[];sparcs=[];ldljs=[];weights=[]
    peak_count=0;still_total=0;still_eps=0;longest=0
    traj=[]
    continuous_duration=0.0

    for seg in segs:
        t,x,y,fs=smooth_xy(seg)
        if fs<=0 or len(t)<5:continue
        vx=np.gradient(x,t);vy=np.gradient(y,t);v=np.sqrt(vx*vx+vy*vy)
        ax=np.gradient(vx,t);ay=np.gradient(vy,t);a=np.sqrt(ax*ax+ay*ay)
        all_v.extend(v);all_a.extend(a)
        s=sparc_metric(v,fs); l=ldlj_metric(t,x,y)
        dur=max(float(t[-1]-t[0]),0.0)
        continuous_duration+=dur
        weights.append(dur);sparcs.append(s);ldljs.append(l)

        peaks,_=find_peaks(
            v,
            prominence=max(np.nanmedian(v)*.25,1e-4),
            distance=max(1,int(fs*.15))
        )
        peak_count+=len(peaks)

        low=v<still_thr
        starts=[]; st=None
        for i,b in enumerate(low):
            if b and st is None:st=i
            if (not b or i==len(low)-1) and st is not None:
                en=i if not b else i+1
                dd=t[min(en-1,len(t)-1)]-t[st]
                if dd>=min_still: starts.append(dd)
                st=None
        still_total+=sum(starts)
        still_eps+=len(starts)
        longest=max([longest]+starts)

        traj.append(pd.DataFrame({
            "time":t,"x":x,"y":y,"speed":v,"accel":a
        }))

    if not traj:return {},pd.DataFrame()

    full=pd.concat(traj,ignore_index=True).sort_values("time")
    x0,y0=full.iloc[0][["x","y"]]
    x1,y1=full.iloc[-1][["x","y"]]
    disp=float(np.hypot(x1-x0,y1-y0))

    path=0.0
    for seg in traj:
        path+=float(np.sum(np.hypot(np.diff(seg.x),np.diff(seg.y))))

    observed_span=max(float(full.time.max()-full.time.min()),0.0)
    weights=np.array(weights,float)

    def wavg(vals):
        vals=np.array(vals,float)
        m=np.isfinite(vals)&np.isfinite(weights)&(weights>0)
        return float(np.average(vals[m],weights=weights[m])) if m.any() else np.nan

    v=np.array(all_v,float)
    a=np.array(all_a,float)

    onset=np.nan
    for seg in traj:
        above=seg.speed.to_numpy(float)>onset_thr
        fs=1/np.median(np.diff(seg.time)) if len(seg)>2 else 0
        need=max(1,int(round(onset_hold*fs))) if fs>0 else 1
        run=0
        for i,b in enumerate(above):
            run=run+1 if b else 0
            if run>=need:
                onset=float(seg.time.iloc[i-need+1]-attempt_start)
                break
        if np.isfinite(onset):break

    longest_seg=max(traj,key=lambda q:q.time.iloc[-1]-q.time.iloc[0])
    tremor_rms=np.nan;tremor_hz=np.nan
    if len(longest_seg)>=32:
        t=longest_seg.time.to_numpy(float)
        fs=1/np.median(np.diff(t))
        if fs>=25:
            x=longest_seg.x.to_numpy(float);y=longest_seg.y.to_numpy(float)
            w=max(5,int(round(.5*fs)));w+=1-w%2
            if w<len(x):
                tx=savgol_filter(x,w,2,mode="interp")
                ty=savgol_filter(y,w,2,mode="interp")
                rx=x-tx;ry=y-ty
                tremor_rms=float(np.sqrt(np.mean(rx*rx+ry*ry)))
                f,px=welch(rx,fs=fs,nperseg=min(len(rx),256))
                _,py=welch(ry,fs=fs,nperseg=min(len(ry),256))
                band=(f>=4)&(f<=min(12,fs/2*.9))
                if band.any():
                    tremor_hz=float(f[band][np.argmax((px+py)[band])])

    raw_eff=disp/path if path>0 else np.nan
    still_fraction_observed=(
        still_total/continuous_duration if continuous_duration>0 else np.nan
    )

    return {
        # attempt duration is the validated mapped window, not tracking span
        "attempt_duration_sec":mapped_duration,
        "mapped_attempt_duration_sec":mapped_duration,
        "observed_track_span_sec":observed_span,
        "observed_continuous_tracking_sec":continuous_duration,
        "start_x_norm":x0,"start_y_norm":y0,
        "end_x_norm":x1,"end_y_norm":y1,
        "displacement_norm":disp,
        "path_length_norm":path,
        "path_efficiency_raw":raw_eff,
        # final path_efficiency is assigned after QC/continuity checks
        "path_efficiency":np.nan,
        "mean_speed_norm_s":float(np.nanmean(v)),
        "peak_speed_norm_s":float(np.nanmax(v)),
        "mean_accel_norm_s2":float(np.nanmean(a)),
        "rms_accel_norm_s2":float(np.sqrt(np.nanmean(a*a))),
        "peak_accel_norm_s2":float(np.nanmax(a)),
        "sparc":wavg(sparcs),
        "log_dimensionless_jerk":wavg(ldljs),
        "velocity_peak_count":peak_count,
        "stillness_total_sec":still_total,
        "stillness_fraction":still_fraction_observed,
        "stillness_fraction_observed_tracking":still_fraction_observed,
        "stillness_fraction_mapped_attempt":(
            still_total/mapped_duration if mapped_duration>0 else np.nan
        ),
        "stillness_episode_count":still_eps,
        "longest_stillness_sec":longest,
        "tremor_rms_norm":tremor_rms,
        "tremor_dominant_hz":tremor_hz,
        "first_sustained_hand_movement_delay_sec":onset,
    },full

def tracking_diagnostics(track,window_start,window_end,max_gap=.25):
    out={
        "tracking_coverage":np.nan,
        "tracking_sample_count":0,
        "tracking_median_dt_sec":np.nan,
        "continuous_segment_count":0,
        "largest_internal_gap_sec":np.nan,
        "leading_missing_sec":np.nan,
        "trailing_missing_sec":np.nan,
        "observed_span_fraction":np.nan,
    }
    if track.empty or window_end<=window_start:
        return out

    d=track.drop_duplicates("time").sort_values("time")
    t=d.time.to_numpy(float)
    out["tracking_sample_count"]=len(t)

    if len(t)>=2:
        diffs=np.diff(t)
        pos=diffs[diffs>0]
        if len(pos):
            dt=float(np.median(pos))
            out["tracking_median_dt_sec"]=dt
            out["tracking_coverage"]=min(
                len(t)*dt/max(window_end-window_start,1e-9),1.0
            )
            out["largest_internal_gap_sec"]=float(np.max(diffs))
        out["continuous_segment_count"]=1+int(np.sum(diffs>max_gap))
    elif len(t)==1:
        out["continuous_segment_count"]=1

    out["leading_missing_sec"]=max(float(t[0]-window_start),0.0)
    out["trailing_missing_sec"]=max(float(window_end-t[-1]),0.0)
    out["observed_span_fraction"]=min(
        max(float(t[-1]-t[0]),0.0)/max(window_end-window_start,1e-9),
        1.0
    )
    return out


def gap_aware_resample100(full,window_start,window_end,max_gap=.25):
    cols=[
        "trajectory_point","normalized_time_0_1","time_sec",
        "x_norm","y_norm","speed_norm_s","accel_norm_s2","tracking_valid"
    ]
    if full.empty or window_end<=window_start:
        return pd.DataFrame(columns=cols)

    target=np.linspace(window_start,window_end,100)
    result=pd.DataFrame({
        "trajectory_point":np.arange(1,101),
        "normalized_time_0_1":np.linspace(0,1,100),
        "time_sec":target,
        "x_norm":np.nan,
        "y_norm":np.nan,
        "speed_norm_s":np.nan,
        "accel_norm_s2":np.nan,
        "tracking_valid":False,
    })

    d=full.drop_duplicates("time").sort_values("time")
    segs=segments(d,max_gap)
    for seg in segs:
        if len(seg)<2:continue
        lo=float(seg.time.iloc[0]); hi=float(seg.time.iloc[-1])
        mask=(target>=lo)&(target<=hi)
        if not mask.any():continue
        tt=target[mask]
        result.loc[mask,"x_norm"]=np.interp(tt,seg.time,seg.x)
        result.loc[mask,"y_norm"]=np.interp(tt,seg.time,seg.y)
        result.loc[mask,"speed_norm_s"]=np.interp(tt,seg.time,seg.speed)
        result.loc[mask,"accel_norm_s2"]=np.interp(tt,seg.time,seg.accel)
        result.loc[mask,"tracking_valid"]=True
    return result

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

def read_gopro_cached(root, vid, cache):
    key=("gopro",vid)
    if key not in cache:
        cache[key]=reconstruct_gopro(read_parts(root/"landmarks"/vid))
    return cache[key]

def read_cae_cached(root, vid, cache):
    key=("cae",vid)
    if key not in cache:
        cache[key]=read_reconstructed(root/"cae"/"analysis"/"reconstructed"/vid)
    return cache[key]

def gopro_duration(root, vid, data):
    meta=root/"landmarks"/vid/"metadata.json"
    if meta.exists():
        try:
            obj=json.loads(meta.read_text(encoding="utf-8"))
            fps=float(obj.get("fps",0))
            frames=float(obj.get("frame_count",0))
            if fps>0 and frames>0:
                return frames/fps
        except Exception:
            pass
    if not data.empty and "video_time_sec" in data.columns:
        t=num(data["video_time_sec"]).dropna().sort_values()
        if len(t):
            dt=float(np.median(np.diff(t))) if len(t)>1 else 0.0
            return float(t.iloc[-1]+max(dt,0.0))
    return np.nan

def shift_track(track, delta):
    if track.empty:return track
    out=track.copy()
    out["time"]=num(out["time"])+delta
    return out

def shift_grip_segments(gs, delta):
    if gs.empty:return gs
    out=gs.copy()
    out["segment_start_sec"]=num(out["segment_start_sec"])+delta
    out["segment_end_sec"]=num(out["segment_end_sec"])+delta
    return out

def cross_file_gopro(root, vids, hand, start, end, cache):
    if len(vids)!=2:
        return pd.DataFrame(),pd.DataFrame(),np.nan,"FAIL_CROSS_FILE_ID_COUNT"

    v1,v2=vids
    d1=read_gopro_cached(root,v1,cache)
    d2=read_gopro_cached(root,v2,cache)
    if d1.empty or d2.empty:
        return pd.DataFrame(),pd.DataFrame(),np.nan,"FAIL_NO_SOURCE_DATA"

    dur1=gopro_duration(root,v1,d1)
    if not np.isfinite(dur1) or dur1<=start:
        return pd.DataFrame(),pd.DataFrame(),np.nan,"FAIL_BAD_FIRST_VIDEO_DURATION"

    # Put both source-local pieces on a single attempt-relative time axis.
    t1=gopro_palm_track(d1,hand,start,dur1)
    t2=gopro_palm_track(d2,hand,0,end)
    t1=shift_track(t1,-start)
    prefix=dur1-start
    t2=shift_track(t2,prefix)
    track=pd.concat([t1,t2],ignore_index=True).sort_values("time") if len(t1) or len(t2) else pd.DataFrame()

    g1=grip_segments_gopro(d1,hand,start,dur1)
    g2=grip_segments_gopro(d2,hand,0,end)
    g1=shift_grip_segments(g1,-start)
    g2=shift_grip_segments(g2,prefix)
    grips=pd.concat([g1,g2],ignore_index=True) if len(g1) or len(g2) else pd.DataFrame()

    window_duration=prefix+end
    qc="PASS" if len(track)>=10 else "WARN_LOW_TRACK"
    return track,grips,window_duration,qc

def append_grip_summary(row, gs, grip_rows, m, source, vid_label, hand):
    if gs.empty:
        row["grip_validation_status"]="screening_only_manual_validation_required"
        return

    gs=gs.sort_values("segment_start_sec").reset_index(drop=True)
    row["initial_grip_candidate"]=gs.iloc[0].grip_candidate
    row["final_grip_candidate"]=gs.iloc[-1].grip_candidate
    row["dominant_grip_candidate"]=gs.groupby("grip_candidate").segment_duration_sec.sum().idxmax()
    row["grip_change_count"]=max(len(gs)-1,0)

    for j,seg in gs.iterrows():
        rr={
            "attempt_id":m.attempt_id,
            "participant":p_norm(m.participant),
            "hand":hand,
            "motion_source":source,
            "video_id":vid_label,
            "segment_number":j+1,
            **seg.to_dict(),
            "validation_status":"screening_only_manual_validation_required",
        }
        grip_rows.append(rr)

    row["grip_validation_status"]="screening_only_manual_validation_required"

def main():
    ap=argparse.ArgumentParser(
        description=(
            "SIMETRIC source-aware motion metrics v3: explicit mapped duration, "
            "coverage/continuity QC, gap-aware trajectories, conservative path efficiency, "
            "and correctly named first sustained hand-movement onset."
        )
    )
    ap.add_argument("--root",required=True)
    ap.add_argument("--mapping",required=True)
    ap.add_argument("--output-dir",required=True)
    ap.add_argument("--stillness-threshold",type=float,default=.03)
    ap.add_argument("--onset-threshold",type=float,default=.05)
    ap.add_argument("--max-gap-sec",type=float,default=.25)
    ap.add_argument("--primary-min-coverage",type=float,default=.80)
    ap.add_argument("--sensitivity-min-coverage",type=float,default=.50)
    args=ap.parse_args()

    root=Path(args.root)
    out=Path(args.output_dir)
    out.mkdir(parents=True,exist_ok=True)

    mp=pd.read_csv(args.mapping,dtype=str).fillna("")
    mp=mp[~mp.participant.map(p_simple).isin(EXCLUDED)]
    mp=mp[mp.mapping_status.isin(USABLE)].copy()

    metrics=[];traj_rows=[];grip_rows=[];cache={}

    for _,m in mp.iterrows():
        source=str(m.chosen_source)
        hand=str(m.hand).lower()
        vids=[x for x in str(m.selected_video_id).split(";") if x]
        vid_label=";".join(vids)
        start=float(m.local_start_sec)
        end=float(m.local_end_sec)
        is_cross=(
            str(m.get("cross_file","")).lower() in {"true","1"}
            or len(vids)>1
        )

        base={
            "attempt_id":m.attempt_id,
            "participant":p_norm(m.participant),
            "attempt_sequence":m.attempt_sequence,
            "round":m["round"],
            "device":m.device,
            "hand":hand,
            "motion_source":source,
            "video_id":vid_label,
            "mapping_status":m.mapping_status,
            "coverage":m.coverage,
            "local_start_sec":start,
            "local_end_sec":end,
            "cross_file":is_cross,
            "tracking_coverage":np.nan,
            "qc_status":"",
            "qc_tier":"",
            "analysis_eligible_primary":False,
            "analysis_eligible_sensitivity":False,
            "trajectory_analysis_eligible":False,
            "review_note":m.review_note,
        }

        if not vids:
            base["qc_status"]="FAIL_NO_VIDEO_ID"
            base["qc_tier"]="FAIL"
            metrics.append(base)
            continue

        if source.startswith("gopro_"):
            if is_cross:
                track,gs,window_duration,qc=cross_file_gopro(
                    root,vids,hand,start,end,cache
                )
                window_start=0.0
                window_end=float(window_duration)
                attempt_start=0.0
                base["qc_status"]=qc
            else:
                data=read_gopro_cached(root,vids[0],cache)
                if data.empty:
                    base["qc_status"]="FAIL_NO_SOURCE_DATA"
                    base["qc_tier"]="FAIL"
                    metrics.append(base)
                    continue
                track=gopro_palm_track(data,hand,start,end)
                gs=grip_segments_gopro(data,hand,start,end)
                window_duration=max(end-start,1e-9)
                window_start=start;window_end=end;attempt_start=start
                base["qc_status"]="PASS" if len(track)>=10 else "WARN_LOW_TRACK"

        elif source=="cae_hand":
            if is_cross or len(vids)!=1:
                base["qc_status"]="FAIL_UNSUPPORTED_CAE_MULTIFILE"
                base["qc_tier"]="FAIL"
                metrics.append(base)
                continue
            data=read_cae_cached(root,vids[0],cache)
            if data.empty:
                base["qc_status"]="FAIL_NO_SOURCE_DATA"
                base["qc_tier"]="FAIL"
                metrics.append(base)
                continue
            track=cae_palm_track(data,hand,start,end)
            gs=grip_segments_cae(data,hand,start,end)
            window_duration=max(end-start,1e-9)
            window_start=start;window_end=end;attempt_start=start
            base["qc_status"]="PASS" if len(track)>=10 else "WARN_LOW_TRACK"
        else:
            base["qc_status"]="FAIL_UNKNOWN_SOURCE"
            base["qc_tier"]="FAIL"
            metrics.append(base)
            continue

        diag=tracking_diagnostics(
            track,window_start,window_end,args.max_gap_sec
        )
        base.update(diag)

        cov=diag["tracking_coverage"]
        span_frac=diag["observed_span_fraction"]

        if not np.isfinite(cov) or len(track)<10:
            base["qc_tier"]="FAIL_OR_INSUFFICIENT_TRACK"
        elif cov < .25:
            base["qc_tier"]="VERY_LOW_COVERAGE_LT25"
        elif cov < .50:
            base["qc_tier"]="LOW_COVERAGE_25_50"
        elif cov < args.primary_min_coverage:
            base["qc_tier"]="MODERATE_COVERAGE_50_PRIMARY"
        else:
            base["qc_tier"]="HIGH_COVERAGE_PRIMARY"

        met,full=motion_metrics(
            track,
            attempt_start,
            window_duration,
            args.stillness_threshold,
            args.onset_threshold,
            max_gap=args.max_gap_sec,
        )
        base.update(met)

        primary_ok=(
            base["qc_status"]=="PASS"
            and np.isfinite(cov)
            and cov>=args.primary_min_coverage
            and np.isfinite(span_frac)
            and span_frac>=args.primary_min_coverage
        )
        sensitivity_ok=(
            base["qc_status"] in {"PASS","WARN_LOW_TRACK"}
            and np.isfinite(cov)
            and cov>=args.sensitivity_min_coverage
            and np.isfinite(span_frac)
            and span_frac>=args.sensitivity_min_coverage
        )

        base["analysis_eligible_primary"]=bool(primary_ok)
        base["analysis_eligible_sensitivity"]=bool(sensitivity_ok)

        raw_eff=base.get("path_efficiency_raw",np.nan)
        # Whole-attempt efficiency is only defensible for the high-coverage
        # primary set; otherwise preserve raw value separately and expose NA.
        if primary_ok and np.isfinite(raw_eff) and raw_eff <= 1.000001:
            base["path_efficiency"]=raw_eff
        else:
            base["path_efficiency"]=np.nan
            if np.isfinite(raw_eff) and raw_eff>1.000001:
                base["path_efficiency_qc"]="INVALID_GT_1"
            elif not primary_ok:
                base["path_efficiency_qc"]="NOT_REPORTED_OUTSIDE_PRIMARY_QC"
            else:
                base["path_efficiency_qc"]="MISSING"
        if "path_efficiency_qc" not in base:
            base["path_efficiency_qc"]="PASS"

        r=gap_aware_resample100(
            full,window_start,window_end,args.max_gap_sec
        )
        valid_fraction=(
            float(r["tracking_valid"].mean()) if len(r) else 0.0
        )
        base["trajectory_valid_fraction_100pt"]=valid_fraction
        base["trajectory_analysis_eligible"]=bool(
            primary_ok and valid_fraction>=args.primary_min_coverage
        )

        append_grip_summary(
            base,gs,grip_rows,m,source,vid_label,hand
        )
        metrics.append(base)

        if len(r):
            r.insert(0,"video_id",vid_label)
            r.insert(0,"motion_source",source)
            r.insert(0,"hand",hand)
            r.insert(0,"participant",p_norm(m.participant))
            r.insert(0,"attempt_id",m.attempt_id)
            r["cross_file"]=is_cross
            r["analysis_eligible_primary"]=bool(primary_ok)
            traj_rows.append(r)

    md=pd.DataFrame(metrics)
    td=pd.concat(traj_rows,ignore_index=True) if traj_rows else pd.DataFrame()
    gd=pd.DataFrame(grip_rows)

    md.to_csv(out/"attempt_hand_metrics_source_aware_v3.csv",index=False)
    td.to_csv(out/"trajectory_100pt_gap_aware_v3.csv",index=False)
    gd.to_csv(out/"grip_segments_candidates_v3.csv",index=False)

    params={
        "stillness_threshold_norm_s":args.stillness_threshold,
        "onset_threshold_norm_s":args.onset_threshold,
        "onset_variable":"first_sustained_hand_movement_delay_sec",
        "onset_interpretation":"first sustained hand motion above threshold; not insertion-specific",
        "max_tracking_gap_sec":args.max_gap_sec,
        "primary_min_tracking_coverage":args.primary_min_coverage,
        "sensitivity_min_tracking_coverage":args.sensitivity_min_coverage,
        "primary_set_requires_observed_span_fraction":args.primary_min_coverage,
        "trajectory_resampling":"100 points across mapped attempt; interpolation only inside continuous tracking segments; long gaps remain missing",
        "path_efficiency":"reported only for primary-QC rows and only when <=1",
        "stillness_fraction":"fraction of observed continuous tracking time; mapped-attempt denominator also retained separately",
        "grip_status":"screening_only_manual_validation_required",
        "tremor_status":"exploratory_tracking_jitter_sensitive",
        "cross_file_support":"tail of first GoPro landmark stream concatenated with head of second on attempt-relative timeline",
    }
    (out/"analysis_parameters_v3.json").write_text(
        json.dumps(params,indent=2),encoding="utf-8"
    )

    print("=== SOURCE-AWARE METRICS V3 COMPLETE ===")
    print("Metric rows:",len(md))
    print("Trajectory rows:",len(td))
    print("Grip segments:",len(gd))
    print()
    print("QC status:")
    print(md["qc_status"].value_counts(dropna=False).to_string())
    print()
    print("QC tiers:")
    print(md["qc_tier"].value_counts(dropna=False).to_string())
    print()
    print("Primary-analysis eligible:",int(md["analysis_eligible_primary"].astype(bool).sum()))
    print("Sensitivity-analysis eligible:",int(md["analysis_eligible_sensitivity"].astype(bool).sum()))
    print("Trajectory primary eligible:",int(md["trajectory_analysis_eligible"].astype(bool).sum()))
    print("Invalid raw path efficiency >1:",int((pd.to_numeric(md["path_efficiency_raw"],errors="coerce")>1.000001).sum()))
    print()
    print("Wrote:",out/"attempt_hand_metrics_source_aware_v3.csv")
    print("Wrote:",out/"trajectory_100pt_gap_aware_v3.csv")
    print("Wrote:",out/"grip_segments_candidates_v3.csv")
    print("Wrote:",out/"analysis_parameters_v3.json")

if __name__=="__main__":
    main()
