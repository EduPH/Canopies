# -*- coding: utf-8 -*-
"""
canopies.html_viz
=================

Self-contained interactive HTML views of a canopy, for small embedded
complexes. Each exporter writes ONE file with two 3-D plotly panels driven by
a shared time slider / play button:

  * left  -- the complex embedded in the plane/space, showing the sublevel set
             K_s(t) = {sigma : f_sigma(t) <= s}; simplices appear and disappear
             as their piecewise-linear values cross the level over time.
  * right -- the canopy: each trajectory as a curve in (birth, death, time)
             space (the vineyard-style picture), structural (non-Hausdorff)
             events marked in red, and a moving marker per trajectory tracking
             the current time in lock-step with the left panel.

`export_dual_sweep_html` additionally sweeps the level s, so both the
parameter t and the filtration level can be explored.

No plotly *python* package is needed: the HTML embeds its data as JSON and
loads plotly.js from a CDN, so the file opens in any browser. This module is
pure standard library.

Public API
----------
    export_filtration_html(cx, plf, canopy, positions, out_html, level, ...)
    export_dual_sweep_html(cx, plf, canopy, positions, out_html, ...)
"""

import json
from fractions import Fraction

INF = float("inf")
_PLOTLY_CDN = "https://cdn.plot.ly/plotly-2.35.2.min.js"


def export_filtration_html(cx, plf, canopy, positions, out_html, level,
                           frames=61, dims=(1,), title="canopy filtration"):
    """Interactive HTML for a small embedded complex (vertices, edges, AND
    triangles). Left panel: the sublevel set K_s(t) = {sigma : f_sigma(t) <= s}
    at the fixed level s=`level`, so simplices APPEAR and DISAPPEAR as their
    piecewise-linear values cross the level over time. Right panel: the canopy
    trajectories (restricted to `dims`) in (birth, death, time), structural
    events in red, and a now-marker per trajectory -- both panels share one
    time slider.

    `positions` maps each vertex integer to a 2D or 3D coordinate.
    """
    from .core import eta_of_pair

    # vertex bookkeeping ----------------------------------------------------
    verts = [cx.simplices[i][0] for i in range(cx.n) if cx.dim[i] == 0]
    vindex = {v: r for r, v in enumerate(verts)}
    def xyz(v):
        p = positions[v]
        return (float(p[0]), float(p[1]), float(p[2]) if len(p) > 2 else 0.0)
    VX = [xyz(v)[0] for v in verts]
    VY = [xyz(v)[1] for v in verts]
    VZ = [xyz(v)[2] for v in verts]
    vlabel = [cx.name(cx.index[(v,)]) for v in verts]

    edge_simps = [(i, cx.simplices[i]) for i in range(cx.n) if cx.dim[i] == 1]
    tri_simps = [(i, cx.simplices[i]) for i in range(cx.n) if cx.dim[i] == 2]
    tri_palette = ["#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#17becf"]

    times = [k / (frames - 1) for k in range(frames)]
    fracs = [Fraction(k, frames - 1) for k in range(frames)]

    # per-frame sublevel membership ----------------------------------------
    edges_xyz, tris_ijk = [], []
    for t in fracs:
        fv = plf.values(t)
        ex, ey, ez = [], [], []
        for (i, s) in edge_simps:
            if fv[i] <= level:
                a, b = s
                ex += [xyz(a)[0], xyz(b)[0], None]
                ey += [xyz(a)[1], xyz(b)[1], None]
                ez += [xyz(a)[2], xyz(b)[2], None]
        edges_xyz.append((ex, ey, ez))
        ii, jj, kk, fc = [], [], [], []
        for ti, (i, s) in enumerate(tri_simps):
            if fv[i] <= level:
                a, b, c = s
                ii.append(vindex[a]); jj.append(vindex[b]); kk.append(vindex[c])
                fc.append(tri_palette[ti % len(tri_palette)])
        tris_ijk.append((ii, jj, kk, fc))

    # canopy trajectories (restricted to dims) ------------------------------
    dims = set(dims)
    trajs = []
    fmax = -INF
    for tr in canopy.trajectories:
        s0 = tr.starts[0][1][0]
        if cx.dim[s0] not in dims:
            continue
        bs, ds = [], []
        for t in fracs:
            fv = plf.values(t)
            b, d = eta_of_pair(fv, tr.pair_at(t))
            bs.append(float(b))
            ds.append(None if d == INF else float(d))
            fmax = max(fmax, float(b), float(b) if d == INF else float(d))
        trajs.append({"label": cx.pair_name(tr.starts[0][1]) + " → " + cx.pair_name(tr.starts[-1][1]),
                      "b": bs, "d": ds})
    events = []
    for e in canopy.structural_events():
        fv = plf.values(e.t)
        for pr in set(e.removed) | set(e.added):
            if cx.dim[pr[0]] not in dims:
                continue
            b, d = eta_of_pair(fv, pr)
            events.append({"t": float(e.t), "b": float(b),
                           "d": (None if d == INF else float(d)),
                           "forced": bool(e.forced)})
    cap = fmax + 0.15 * (abs(fmax) + 1.0)

    data = {"title": title, "level": float(level),
            "VX": VX, "VY": VY, "VZ": VZ, "vlabel": vlabel,
            "times": times, "edges_xyz": edges_xyz, "tris_ijk": tris_ijk,
            "trajs": trajs, "events": events, "cap": cap}
    html = _FILT_TEMPLATE.replace("__PLOTLY_CDN__", _PLOTLY_CDN)
    html = html.replace("__TITLE__", title).replace("__DATA_JSON__", json.dumps(data))
    with open(out_html, "w") as f:
        f.write(html)
    return out_html


_FILT_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>__TITLE__</title>
<script src="__PLOTLY_CDN__"></script>
<style>
  body { font-family: system-ui, sans-serif; margin:0; background:#fff; color:#111; }
  #head { padding:10px 16px 0; }
  #head h2 { margin:0 0 4px; font-size:18px; }
  #head p  { margin:0; font-size:13px; color:#555; }
  #plot { width:100%; height:82vh; }
</style>
</head>
<body>
<div id="head">
  <h2>__TITLE__</h2>
  <p>Left: the sublevel complex K<sub>s</sub>(t) at level s = <b id="lvl"></b> —
     edges and triangles appear/disappear as their filtration values cross the
     level over time (holes open and close). Right: canopy trajectories in
     (birth, death, time); red = structural (non-Hausdorff) events. Shared
     Play / time slider.</p>
</div>
<div id="plot"></div>
<script>
const D = __DATA_JSON__;
document.getElementById("lvl").textContent = D.level;
const F = D.times.length;

// ---- left: vertices (static) ---------------------------------------------
const vTrace = {type:"scatter3d", mode:"markers+text", x:D.VX,y:D.VY,z:D.VZ,
  marker:{size:5, color:"#222"}, text:D.vlabel, textposition:"top center",
  hoverinfo:"text", scene:"scene", showlegend:false};

// ---- left: edges (animated) ----------------------------------------------
const e0 = D.edges_xyz[0];
const eTrace = {type:"scatter3d", mode:"lines", x:e0[0],y:e0[1],z:e0[2],
  line:{color:"#1f77b4", width:6}, hoverinfo:"none", scene:"scene", showlegend:false};

// ---- left: triangles (animated mesh) -------------------------------------
const t0 = D.tris_ijk[0];
const triTrace = {type:"mesh3d", x:D.VX,y:D.VY,z:D.VZ, i:t0[0],j:t0[1],k:t0[2],
  facecolor:t0[3], opacity:0.45, hoverinfo:"none", scene:"scene", showlegend:false};

// ---- right: canopy trajectories (static) ---------------------------------
const cap = D.cap;
const palette = ["#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd","#8c564b","#e377c2"];
const trajTraces = [];
D.trajs.forEach((tr,i)=>{
  const dd = tr.d.map(v=> v===null?cap:v);
  trajTraces.push({type:"scatter3d", mode:"lines", x:tr.b,y:dd,z:D.times,
    line:{color:palette[i%palette.length], width:5}, name:tr.label,
    hovertext:tr.label, hoverinfo:"text", scene:"scene2", showlegend:true});
});

// ---- right: structural events (static) -----------------------------------
const evX=[],evY=[],evZ=[],evT=[];
for (const e of D.events){ evX.push(e.b); evY.push(e.d===null?cap:e.d); evZ.push(e.t);
  evT.push("structural t="+e.t.toFixed(3)+(e.forced?" (forced)":" (choice)")); }
const evTrace = {type:"scatter3d", mode:"markers", x:evX,y:evY,z:evZ,
  marker:{size:7, color:"red", symbol:"diamond"}, text:evT, hoverinfo:"text",
  scene:"scene2", showlegend:false};

// ---- right: now-markers (animated) ---------------------------------------
function nowPts(f){ const x=[],y=[],z=[];
  D.trajs.forEach(tr=>{ x.push(tr.b[f]); y.push(tr.d[f]===null?cap:tr.d[f]); z.push(D.times[f]); });
  return {x,y,z}; }
const n0 = nowPts(0);
const nowTrace = {type:"scatter3d", mode:"markers", x:n0.x,y:n0.y,z:n0.z,
  marker:{size:5, color:"black"}, hoverinfo:"none", scene:"scene2", showlegend:false};

const traces = [vTrace, eTrace, triTrace, ...trajTraces, evTrace, nowTrace];
const IDX_EDGE=1, IDX_TRI=2, IDX_NOW=traces.length-1;

const frames=[];
for (let f=0; f<F; f++){
  const ex=D.edges_xyz[f], ti=D.tris_ijk[f], np=nowPts(f);
  frames.push({name:String(f),
    data:[ {x:ex[0], y:ex[1], z:ex[2]},
           {i:ti[0], j:ti[1], k:ti[2], facecolor:ti[3]},
           {x:np.x, y:np.y, z:np.z} ],
    traces:[IDX_EDGE, IDX_TRI, IDX_NOW]});
}
const steps = D.times.map((t,f)=>({label:t.toFixed(2), method:"animate",
  args:[[String(f)],{mode:"immediate", frame:{duration:0,redraw:true}, transition:{duration:0}}]}));

const layout = {
  margin:{l:0,r:0,t:10,b:0},
  scene:{domain:{x:[0,0.46],y:[0,1]}, aspectmode:"data",
         xaxis:{title:""}, yaxis:{title:""}, zaxis:{title:"", range:[-0.5,0.5]},
         camera:{eye:{x:0.1,y:-0.2,z:1.8}}},
  scene2:{domain:{x:[0.54,1],y:[0,1]},
          xaxis:{title:"birth"}, yaxis:{title:"death"}, zaxis:{title:"time"}},
  legend:{x:0.54, y:1, font:{size:10}},
  updatemenus:[{type:"buttons", showactive:false, x:0.05, y:0, xanchor:"right", yanchor:"top",
    buttons:[
      {label:"▶ Play", method:"animate", args:[null,{fromcurrent:true, frame:{duration:140,redraw:true}, transition:{duration:0}}]},
      {label:"❚❚ Pause", method:"animate", args:[[null],{mode:"immediate", frame:{duration:0,redraw:false}, transition:{duration:0}}]}
    ]}],
  sliders:[{active:0, x:0.1, len:0.85, y:0, pad:{t:30},
    currentvalue:{prefix:"t = ", font:{size:14}}, steps:steps}]
};
Plotly.newPlot("plot", traces, layout, {responsive:true})
  .then(()=>Plotly.addFrames("plot", frames));
</script>
</body>
</html>
"""


def export_dual_sweep_html(cx, plf, canopy, positions, out_html,
                           frames=61, dims=(1,), s_default=2.5, s_steps=80,
                           title="canopy dual sweep"):
    """Interactive HTML with BOTH sweeps: a time slider t and a level slider s.

    Left panel: the sublevel complex K_s(t) = {sigma : f_sigma(t) <= s}. Move s
    to watch simplices switch on (the persistence sweep at fixed t); move t to
    deform the filtration. Right panel: the canopy trajectories in
    (birth, death, time), a translucent 'death = s' slice plane, structural
    events (red), now-markers per trajectory colored by whether that feature is
    currently OPEN as a hole (birth <= s < death). Both sliders update live.
    """
    from .core import eta_of_pair

    verts = [cx.simplices[i][0] for i in range(cx.n) if cx.dim[i] == 0]
    vindex = {v: r for r, v in enumerate(verts)}

    def xyz(v):
        p = positions[v]
        return (float(p[0]), float(p[1]), float(p[2]) if len(p) > 2 else 0.0)
    VX = [xyz(v)[0] for v in verts]
    VY = [xyz(v)[1] for v in verts]
    VZ = [xyz(v)[2] for v in verts]
    vlabel = [cx.name(cx.index[(v,)]) for v in verts]
    vid = [cx.index[(v,)] for v in verts]

    edges = [(cx.simplices[i], i) for i in range(cx.n) if cx.dim[i] == 1]
    tris = [(cx.simplices[i], i) for i in range(cx.n) if cx.dim[i] == 2]
    edge_pairs = [[vindex[s[0]], vindex[s[1]]] for (s, _i) in edges]
    tri_tris = [[vindex[s[0]], vindex[s[1]], vindex[s[2]]] for (s, _i) in tris]
    tri_palette = ["#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#17becf"]
    tri_colors = [tri_palette[i % len(tri_palette)] for i in range(len(tris))]

    times = [k / (frames - 1) for k in range(frames)]
    fracs = [Fraction(k, frames - 1) for k in range(frames)]

    vert_vals, edge_vals, tri_vals = [], [], []
    vmin, vmax = INF, -INF
    for t in fracs:
        fv = plf.values(t)
        vv = [float(fv[i]) for i in vid]
        ev = [float(fv[i]) for (_s, i) in edges]
        tv = [float(fv[i]) for (_s, i) in tris]
        vert_vals.append(vv); edge_vals.append(ev); tri_vals.append(tv)
        allv = vv + ev + tv
        vmin = min(vmin, min(allv)); vmax = max(vmax, max(allv))

    dims = set(dims)
    trajs = []
    bmin, bmax = INF, -INF
    for tr in canopy.trajectories:
        if cx.dim[tr.starts[0][1][0]] not in dims:
            continue
        bs, ds = [], []
        for t in fracs:
            fv = plf.values(t)
            b, d = eta_of_pair(fv, tr.pair_at(t))
            bs.append(float(b))
            ds.append(None if d == INF else float(d))
            bmin = min(bmin, float(b)); bmax = max(bmax, float(b))
        trajs.append({"label": cx.pair_name(tr.starts[0][1]) + " → " + cx.pair_name(tr.starts[-1][1]),
                      "b": bs, "d": ds})
    events = []
    for e in canopy.structural_events():
        fv = plf.values(e.t)
        for pr in set(e.removed) | set(e.added):
            if cx.dim[pr[0]] not in dims:
                continue
            b, d = eta_of_pair(fv, pr)
            events.append({"t": float(e.t), "b": float(b),
                           "d": (None if d == INF else float(d)), "forced": bool(e.forced)})
    cap = bmax + 0.15 * (abs(bmax) + 1.0)
    for tr in trajs:                                  # cap infinite deaths for display
        pass

    data = {"title": title,
            "VX": VX, "VY": VY, "VZ": VZ, "vlabel": vlabel,
            "edges": edge_pairs, "tris": tri_tris, "tri_colors": tri_colors,
            "times": times, "vert_vals": vert_vals, "edge_vals": edge_vals,
            "tri_vals": tri_vals, "trajs": trajs, "events": events,
            "cap": cap, "s_min": vmin, "s_max": vmax, "s_default": float(s_default),
            "s_steps": int(s_steps), "b_min": bmin, "b_max": bmax}
    html = _DUAL_TEMPLATE.replace("__PLOTLY_CDN__", _PLOTLY_CDN)
    html = html.replace("__TITLE__", title).replace("__DATA_JSON__", json.dumps(data))
    with open(out_html, "w") as f:
        f.write(html)
    return out_html


_DUAL_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>__TITLE__</title>
<script src="__PLOTLY_CDN__"></script>
<style>
  body { font-family: system-ui, sans-serif; margin:0; background:#fff; color:#111; }
  #head { padding:10px 16px 0; }
  #head h2 { margin:0 0 4px; font-size:18px; }
  #head p { margin:0; font-size:13px; color:#555; }
  #plot { width:100%; height:72vh; }
  #ctl { padding:8px 16px 16px; display:grid; grid-template-columns:auto 1fr auto; gap:6px 12px; align-items:center; max-width:900px; }
  #ctl label { font-size:13px; color:#333; }
  #ctl .val { font-variant-numeric:tabular-nums; font-size:13px; width:70px; }
  input[type=range] { width:100%; }
  button { font-size:13px; padding:4px 10px; }
</style>
</head>
<body>
<div id="head">
  <h2>__TITLE__</h2>
  <p>Left: sublevel complex K<sub>s</sub>(t). Move <b>s</b> to sweep the filtration
     threshold (simplices switch on as s rises); move <b>t</b> to deform it. A hole
     is open only while s sits between a cycle's birth (its edges) and death (its
     filling triangle) — at s = max the whole complex is filled, so no holes (that
     is correct). Right: canopy in (birth, death, time) with the <b>death = s</b>
     slice plane; now-markers turn red when that feature is open as a hole
     (birth ≤ s &lt; death), grey when closed.</p>
</div>
<div id="plot"></div>
<div id="ctl">
  <button id="play">▶ Play t</button>
  <input id="tr" type="range" min="0" max="0" step="1" value="0"/>
  <span class="val" id="tval">t=0.00</span>
  <label>level s</label>
  <input id="sr" type="range" min="0" max="1" step="0.001" value="0"/>
  <span class="val" id="sval">s=0.00</span>
</div>
<script>
const D = __DATA_JSON__;
const F = D.times.length;
const cap = D.cap;

// ---------- static geometry ------------------------------------------------
const palette = ["#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd","#8c564b","#e377c2"];

function presentVerts(tk, s){
  const x=[],y=[],z=[],txt=[];
  for (let vi=0; vi<D.VX.length; vi++) if (D.vert_vals[tk][vi] <= s){
    x.push(D.VX[vi]); y.push(D.VY[vi]); z.push(D.VZ[vi]); txt.push(D.vlabel[vi]); }
  return {x,y,z,txt};
}
function presentEdges(tk, s){
  const x=[],y=[],z=[];
  D.edges.forEach((e,ei)=>{ if (D.edge_vals[tk][ei] <= s){
    x.push(D.VX[e[0]],D.VX[e[1]],null); y.push(D.VY[e[0]],D.VY[e[1]],null); z.push(D.VZ[e[0]],D.VZ[e[1]],null);} });
  return {x,y,z};
}
function presentTris(tk, s){
  const i=[],j=[],k=[],fc=[];
  D.tris.forEach((t,ti)=>{ if (D.tri_vals[tk][ti] <= s){
    i.push(t[0]); j.push(t[1]); k.push(t[2]); fc.push(D.tri_colors[ti]); } });
  return {i,j,k,fc};
}
function nowPts(tk, s){
  const x=[],y=[],z=[],col=[];
  D.trajs.forEach(tr=>{ const b=tr.b[tk], d=(tr.d[tk]===null?Infinity:tr.d[tk]);
    x.push(b); y.push(d===Infinity?cap:d); z.push(D.times[tk]);
    col.push((b<=s && s<d) ? "red" : "#888"); });
  return {x,y,z,col};
}

// ---------- static right-panel pieces (canopy curves + events) -------------
const bpad=0.1*(Math.abs(D.b_max-D.b_min)+1), b0=D.b_min-bpad, b1=D.b_max+bpad, t1=D.times[F-1];
const trajTraces=D.trajs.map((tr,i)=>{ const dd=tr.d.map(v=>v===null?cap:v);
  return {type:"scatter3d",mode:"lines",x:tr.b,y:dd,z:D.times,
    line:{color:palette[i%palette.length],width:5},name:tr.label,hovertext:tr.label,
    hoverinfo:"text",scene:"scene2",showlegend:true,uid:"traj"+i}; });
const evX=[],evY=[],evZ=[],evT=[];
for (const e of D.events){ evX.push(e.b); evY.push(e.d===null?cap:e.d); evZ.push(e.t);
  evT.push("structural t="+e.t.toFixed(3)+(e.forced?" (forced)":" (choice)")); }
const evTrace={type:"scatter3d",mode:"markers",x:evX,y:evY,z:evZ,
  marker:{size:7,color:"red",symbol:"diamond"},text:evT,hoverinfo:"text",
  scene:"scene2",showlegend:false,uid:"ev"};

// Build the FULL trace list for a given (tk, s). We rebuild via Plotly.react
// (rather than restyle) so the triangle mesh, edges and slice plane all update
// reliably; uid + uirevision keep the camera and identities stable.
function buildTraces(tk, s){
  const pv=presentVerts(tk,s), pe=presentEdges(tk,s), pt=presentTris(tk,s), nw=nowPts(tk,s);
  const vTrace={type:"scatter3d",mode:"markers+text",x:pv.x,y:pv.y,z:pv.z,text:pv.txt,
    textposition:"top center",marker:{size:5,color:"#222"},hoverinfo:"text",
    scene:"scene",showlegend:false,uid:"v"};
  const eTrace={type:"scatter3d",mode:"lines",x:pe.x,y:pe.y,z:pe.z,
    line:{color:"#1f77b4",width:6},hoverinfo:"none",scene:"scene",showlegend:false,uid:"e"};
  const triTrace={type:"mesh3d",x:D.VX,y:D.VY,z:D.VZ,i:pt.i,j:pt.j,k:pt.k,facecolor:pt.fc,
    opacity:0.45,flatshading:true,hoverinfo:"none",scene:"scene",showlegend:false,uid:"tri"};
  const planeTrace={type:"mesh3d",x:[b0,b1,b1,b0],y:[s,s,s,s],z:[0,0,t1,t1],
    i:[0,0],j:[1,2],k:[2,3],color:"#888",opacity:0.18,hoverinfo:"none",
    scene:"scene2",showlegend:false,uid:"plane"};
  const nowTrace={type:"scatter3d",mode:"markers",x:nw.x,y:nw.y,z:nw.z,
    marker:{size:6,color:nw.col},hoverinfo:"none",scene:"scene2",showlegend:false,uid:"now"};
  return [vTrace,eTrace,triTrace,...trajTraces,evTrace,planeTrace,nowTrace];
}

const layout={ margin:{l:0,r:0,t:10,b:0}, uirevision:"keep",
  scene:{domain:{x:[0,0.46],y:[0,1]},aspectmode:"data",uirevision:"keep",
    xaxis:{title:""},yaxis:{title:""},zaxis:{title:"",range:[-0.5,0.5]},camera:{eye:{x:0.1,y:-0.2,z:1.8}}},
  scene2:{domain:{x:[0.54,1],y:[0,1]},uirevision:"keep",
    xaxis:{title:"birth"},yaxis:{title:"death"},zaxis:{title:"time"}},
  legend:{x:0.54,y:1,font:{size:10}} };

// ---------- controls -------------------------------------------------------
const tr=document.getElementById("tr"), sr=document.getElementById("sr");
const tval=document.getElementById("tval"), sval=document.getElementById("sval");
tr.max=F-1; sr.min=D.s_min; sr.max=D.s_max; sr.step=(D.s_max-D.s_min)/D.s_steps; sr.value=D.s_default;

function render(){
  const tk=+tr.value, s=+sr.value;
  tval.textContent="t="+D.times[tk].toFixed(2); sval.textContent="s="+s.toFixed(2);
  Plotly.react("plot",buildTraces(tk,s),layout,{responsive:true});
}
tr.addEventListener("input",render);
sr.addEventListener("input",render);

const plotDiv=document.getElementById("plot");
Plotly.newPlot(plotDiv,buildTraces(0,D.s_default),layout,{responsive:true});
render();

// hover a vine -> darken/thicken it, dim the rest (restyle, no full rebuild)
const TRAJ_BASE=3, NT=D.trajs.length;
let HL=null;
function applyHighlight(idx){ const ids=[],cols=[],wids=[];
  for (let i=0;i<NT;i++){ ids.push(TRAJ_BASE+i);
    cols.push(i===idx ? palette[i%palette.length] : "rgba(150,150,160,0.12)");
    wids.push(i===idx ? 8 : 1); }
  Plotly.restyle(plotDiv,{"line.color":cols,"line.width":wids},ids); }
function clearHighlight(){ const ids=[],cols=[],wids=[];
  for (let i=0;i<NT;i++){ ids.push(TRAJ_BASE+i); cols.push(palette[i%palette.length]); wids.push(5); }
  Plotly.restyle(plotDiv,{"line.color":cols,"line.width":wids},ids); }
plotDiv.on("plotly_hover", ev=>{ const cn=ev.points[0].curveNumber;
  if (cn>=TRAJ_BASE && cn<TRAJ_BASE+NT && HL!==cn-TRAJ_BASE){ HL=cn-TRAJ_BASE; applyHighlight(HL); } });
plotDiv.on("plotly_unhover", ()=>{ if (HL!==null){ HL=null; clearHighlight(); } });

let timer=null;
document.getElementById("play").addEventListener("click",function(){
  if (timer){ clearInterval(timer); timer=null; this.textContent="▶ Play t"; return; }
  this.textContent="❚❚ Pause";
  timer=setInterval(()=>{ tr.value=(+tr.value+1)%F; render(); },140);
});
</script>
</body>
</html>
"""
