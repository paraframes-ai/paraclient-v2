#!/usr/bin/env python3
"""Generate a self-contained civics review console (single HTML file) from the
candidate JSONL. The reviewer approves/edits/rejects each item; the page exports
a reviewed JSONL that plugs into build_civics_dataset.py --promote."""
import json
import sys

import civics_schema as cs

rows = [json.loads(l) for l in open("data/civics_candidates.jsonl")]
data = []
for r in rows:
    msgs = {m["role"]: m["content"] for m in r["messages"]}
    data.append({
        "source": r["source"],
        "strand": r["strand"],
        "grade_band": r["grade_band"],
        "time_varying": bool(r["time_varying"]),
        "license": r["license"],
        "attribution": r.get("attribution", ""),
        "q": msgs.get("user", ""),
        "a": msgs.get("assistant", ""),
    })

TEMPLATE = r"""<title>Civics Review Console</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
:root{
  --bg:#F4F6F9; --panel:#FFFFFF; --panel-2:#FAFBFD; --ink:#161A22; --muted:#5B6573;
  --line:#E3E7ED; --line-2:#EDF0F5;
  --accent:#1F3A5F; --accent-soft:#E8EEF6; --accent-ink:#1F3A5F;
  --ok:#2E7D5B; --ok-soft:#E5F1EB; --no:#B23B3B; --no-soft:#F8E8E8;
  --warn:#9A6A10; --warn-soft:#F6EBD3;
  --font-ui:system-ui,-apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
  --font-doc:"Iowan Old Style","Palatino Linotype",Palatino,Georgia,"Times New Roman",serif;
  --font-mono:ui-monospace,"SF Mono","Cascadia Code",Menlo,Consolas,monospace;
  --radius:12px; --shadow:0 1px 2px rgba(20,30,50,.04),0 8px 24px rgba(20,30,50,.06);
}
@media (prefers-color-scheme:dark){
  :root{
    --bg:#0F131A; --panel:#181D27; --panel-2:#141922; --ink:#E7EBF1; --muted:#98A2B1;
    --line:#29313D; --line-2:#222933;
    --accent:#7BA7D9; --accent-soft:#1B2636; --accent-ink:#B9D0EC;
    --ok:#58C596; --ok-soft:#17281F; --no:#E38585; --no-soft:#2A1A1A;
    --warn:#D9A951; --warn-soft:#2A2413;
    --shadow:0 1px 2px rgba(0,0,0,.3),0 10px 30px rgba(0,0,0,.35);
  }
}
:root[data-theme="light"]{
  --bg:#F4F6F9; --panel:#FFFFFF; --panel-2:#FAFBFD; --ink:#161A22; --muted:#5B6573;
  --line:#E3E7ED; --line-2:#EDF0F5;
  --accent:#1F3A5F; --accent-soft:#E8EEF6; --accent-ink:#1F3A5F;
  --ok:#2E7D5B; --ok-soft:#E5F1EB; --no:#B23B3B; --no-soft:#F8E8E8;
  --warn:#9A6A10; --warn-soft:#F6EBD3;
  --shadow:0 1px 2px rgba(20,30,50,.04),0 8px 24px rgba(20,30,50,.06);
}
:root[data-theme="dark"]{
  --bg:#0F131A; --panel:#181D27; --panel-2:#141922; --ink:#E7EBF1; --muted:#98A2B1;
  --line:#29313D; --line-2:#222933;
  --accent:#7BA7D9; --accent-soft:#1B2636; --accent-ink:#B9D0EC;
  --ok:#58C596; --ok-soft:#17281F; --no:#E38585; --no-soft:#2A1A1A;
  --warn:#D9A951; --warn-soft:#2A2413;
  --shadow:0 1px 2px rgba(0,0,0,.3),0 10px 30px rgba(0,0,0,.35);
}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font-family:var(--font-ui);
  font-size:15px;line-height:1.5;-webkit-font-smoothing:antialiased}
button{font-family:inherit}
:focus-visible{outline:2px solid var(--accent);outline-offset:2px;border-radius:6px}

/* ---- top bar ---- */
header{position:sticky;top:0;z-index:10;background:var(--panel);border-bottom:1px solid var(--line);
  padding:12px 20px;display:flex;align-items:center;gap:18px;flex-wrap:wrap}
.brand{display:flex;align-items:baseline;gap:10px}
.brand h1{font-size:16px;margin:0;font-weight:650;letter-spacing:-.01em}
.brand .sub{font-size:12px;color:var(--muted);font-family:var(--font-mono)}
.meter{flex:1;min-width:180px;display:flex;flex-direction:column;gap:5px}
.meter .track{height:7px;background:var(--line-2);border-radius:99px;overflow:hidden;display:flex}
.meter .track i{display:block;height:100%}
.meter .track .a{background:var(--ok)} .meter .track .r{background:var(--no)}
.meter .lbl{font-size:11.5px;color:var(--muted);font-variant-numeric:tabular-nums;
  display:flex;gap:12px;font-family:var(--font-mono)}
.counts{display:flex;gap:6px}
.pill{font-size:11.5px;font-weight:600;padding:3px 9px;border-radius:99px;font-variant-numeric:tabular-nums;
  border:1px solid transparent;white-space:nowrap}
.pill.ok{background:var(--ok-soft);color:var(--ok)} .pill.no{background:var(--no-soft);color:var(--no)}
.pill.pend{background:var(--line-2);color:var(--muted)}
.who{display:flex;align-items:center;gap:6px}
.who input{background:var(--panel-2);border:1px solid var(--line);color:var(--ink);border-radius:8px;
  padding:6px 9px;font-size:13px;width:130px}
.btn{background:var(--panel-2);border:1px solid var(--line);color:var(--ink);border-radius:8px;
  padding:7px 13px;font-size:13px;font-weight:600;cursor:pointer;transition:background .12s,border-color .12s}
.btn:hover{background:var(--accent-soft);border-color:var(--accent)}
.btn.primary{background:var(--accent);border-color:var(--accent);color:#fff}
:root[data-theme="dark"] .btn.primary,@media(prefers-color-scheme:dark){}
.btn.primary:hover{filter:brightness(1.06)}

/* ---- layout ---- */
.wrap{display:grid;grid-template-columns:248px minmax(0,1fr);gap:0;min-height:calc(100vh - 58px)}
aside{border-right:1px solid var(--line);background:var(--panel-2);overflow:auto;max-height:calc(100vh - 58px);
  position:sticky;top:58px}
.filters{padding:12px;display:flex;gap:6px;flex-wrap:wrap;border-bottom:1px solid var(--line)}
.filters button{font-size:11.5px;padding:4px 9px;border-radius:99px;border:1px solid var(--line);
  background:var(--panel);color:var(--muted);cursor:pointer;font-weight:600}
.filters button[aria-pressed="true"]{background:var(--accent-soft);color:var(--accent-ink);border-color:var(--accent)}
.list{padding:6px}
.row{display:flex;align-items:center;gap:9px;padding:7px 9px;border-radius:8px;cursor:pointer;
  font-size:12.5px;color:var(--muted)}
.row:hover{background:var(--line-2)}
.row.active{background:var(--accent-soft);color:var(--accent-ink)}
.row .dot{width:8px;height:8px;border-radius:99px;background:var(--line);flex:none}
.row.ok .dot{background:var(--ok)} .row.no .dot{background:var(--no)}
.row .src{font-family:var(--font-mono);font-size:11px;flex:none;width:104px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.row .tv{margin-left:auto;color:var(--warn);font-size:11px;flex:none}

/* ---- card ---- */
main{padding:26px clamp(16px,4vw,44px);display:flex;justify-content:center}
.card{width:100%;max-width:720px;background:var(--panel);border:1px solid var(--line);
  border-radius:var(--radius);box-shadow:var(--shadow);overflow:hidden}
.card .top{padding:16px 22px;border-bottom:1px solid var(--line-2);display:flex;align-items:center;gap:9px;flex-wrap:wrap}
.idx{font-family:var(--font-mono);font-size:12.5px;color:var(--muted);font-variant-numeric:tabular-nums}
.chip{font-size:11px;font-weight:650;letter-spacing:.03em;text-transform:uppercase;padding:3px 8px;
  border-radius:6px;background:var(--line-2);color:var(--muted)}
.chip.src{font-family:var(--font-mono);text-transform:none;letter-spacing:0;background:var(--accent-soft);color:var(--accent-ink)}
.chip.tv{background:var(--warn-soft);color:var(--warn)}
.status-tag{margin-left:auto;font-size:12px;font-weight:700}
.status-tag.ok{color:var(--ok)} .status-tag.no{color:var(--no)} .status-tag.pend{color:var(--muted)}
.body{padding:22px}
.q{font-family:var(--font-doc);font-size:23px;line-height:1.32;letter-spacing:-.01em;
  text-wrap:balance;margin:0 0 20px}
.warnbox{display:flex;gap:10px;background:var(--warn-soft);color:var(--warn);border-radius:10px;
  padding:11px 13px;font-size:13px;margin:0 0 18px;line-height:1.4}
.warnbox b{font-weight:700}
label.fld{display:block;font-size:11.5px;font-weight:650;letter-spacing:.04em;text-transform:uppercase;
  color:var(--muted);margin:0 0 6px}
textarea{width:100%;background:var(--panel-2);border:1px solid var(--line);color:var(--ink);
  border-radius:10px;padding:12px 14px;font-family:var(--font-ui);font-size:15px;line-height:1.5;resize:vertical}
textarea.ans{min-height:92px} textarea.note{min-height:52px;font-size:13.5px}
.meta{display:flex;gap:16px;flex-wrap:wrap;margin:16px 0 4px}
.meta .m{display:flex;flex-direction:column;gap:4px}
.meta label{font-size:10.5px;text-transform:uppercase;letter-spacing:.04em;color:var(--muted);font-weight:650}
.meta select{background:var(--panel-2);border:1px solid var(--line);color:var(--ink);border-radius:8px;
  padding:6px 8px;font-size:13px;font-family:var(--font-ui)}
.tvtoggle{display:flex;align-items:center;gap:7px;font-size:13px;color:var(--muted);cursor:pointer;user-select:none}
.spacer{height:14px}

/* ---- action bar ---- */
.actions{position:sticky;bottom:0;background:var(--panel);border-top:1px solid var(--line);
  padding:12px 22px;display:flex;align-items:center;gap:10px;flex-wrap:wrap}
.actions .nav{display:flex;gap:8px}
.abtn{flex:none;border:1px solid var(--line);background:var(--panel-2);color:var(--ink);
  border-radius:10px;padding:10px 16px;font-size:14px;font-weight:650;cursor:pointer;display:flex;gap:8px;align-items:center}
.abtn kbd{font-family:var(--font-mono);font-size:11px;background:var(--line-2);border-radius:4px;padding:1px 5px;color:var(--muted)}
.abtn.approve{border-color:var(--ok);color:var(--ok)} .abtn.approve:hover{background:var(--ok-soft)}
.abtn.reject{border-color:var(--no);color:var(--no)} .abtn.reject:hover{background:var(--no-soft)}
.abtn.skip:hover{background:var(--line-2)}
.grow{flex:1}
.hint{font-size:11.5px;color:var(--muted);font-family:var(--font-mono)}
@media (max-width:820px){ .wrap{grid-template-columns:1fr} aside{display:none} }
@media (prefers-reduced-motion:reduce){*{transition:none!important}}
</style>

<header>
  <div class="brand"><h1>Civics Review</h1><span class="sub">K-12 · pre-training gate</span></div>
  <div class="meter">
    <div class="track"><i class="a" id="mA"></i><i class="r" id="mR"></i></div>
    <div class="lbl"><span id="mReviewed">0/0 reviewed</span><span id="mPct">0%</span></div>
  </div>
  <div class="counts">
    <span class="pill ok" id="cOk">0 approved</span>
    <span class="pill no" id="cNo">0 rejected</span>
    <span class="pill pend" id="cPend">0 pending</span>
  </div>
  <div class="who"><label class="hint" for="who">reviewer</label>
    <input id="who" placeholder="your name" autocomplete="off"></div>
  <button class="btn" id="resetBtn" title="Clear all decisions in this browser">Reset</button>
  <button class="btn primary" id="exportBtn">Export JSONL</button>
</header>

<div class="wrap">
  <aside>
    <div class="filters" id="filters">
      <button data-f="all" aria-pressed="true">All</button>
      <button data-f="pending">Pending</button>
      <button data-f="approved">Approved</button>
      <button data-f="rejected">Rejected</button>
      <button data-f="tv">Time-varying</button>
    </div>
    <div class="list" id="list"></div>
  </aside>
  <div>
    <main><div class="card" id="card"></div></main>
    <div class="actions">
      <div class="nav">
        <button class="abtn skip" id="prevBtn">‹ Prev <kbd>J</kbd></button>
        <button class="abtn skip" id="nextBtn">Next › <kbd>K</kbd></button>
      </div>
      <div class="grow"></div>
      <button class="abtn reject" id="rejectBtn">Reject <kbd>R</kbd></button>
      <button class="abtn skip" id="skipBtn">Skip <kbd>S</kbd></button>
      <button class="abtn approve" id="approveBtn">Approve <kbd>A</kbd></button>
    </div>
  </div>
</div>

<script>
const CANDIDATES = __DATA__;
const SYS = __SYS__;
const KEY = "civics-review-v1";
const STRANDS = ["civic_life","foundations","constitution","world_affairs","citizen_roles"];
const BANDS = ["elementary","middle","high"];

let state = load();
let filter = "all";
let cur = 0;

function load(){
  try{ return JSON.parse(localStorage.getItem(KEY)) || {}; }catch(e){ return {}; }
}
function save(){ localStorage.setItem(KEY, JSON.stringify(state)); }
function rec(i){
  const c = CANDIDATES[i];
  if(!state[c.source+"#"+c.grade_band]) state[c.source+"#"+c.grade_band] =
    {status:"pending", a:c.a, note:"", strand:c.strand, band:c.grade_band, tv:c.time_varying};
  return state[c.source+"#"+c.grade_band];
}
function visible(){
  return CANDIDATES.map((c,i)=>i).filter(i=>{
    const r = rec(i);
    if(filter==="all") return true;
    if(filter==="tv") return CANDIDATES[i].time_varying;
    return r.status===filter;
  });
}
const $ = s=>document.querySelector(s);

function renderList(){
  const vis = visible();
  const el = $("#list"); el.innerHTML="";
  vis.forEach(i=>{
    const c=CANDIDATES[i], r=rec(i);
    const row=document.createElement("div");
    row.className="row "+(r.status==="approved"?"ok":r.status==="rejected"?"no":"")+(i===cur?" active":"");
    row.innerHTML=`<span class="dot"></span><span class="src">${c.source}</span>`+
      `<span>${c.grade_band[0].toUpperCase()}</span>`+(c.time_varying?`<span class="tv">◷</span>`:"");
    row.onclick=()=>{cur=i;renderAll();};
    el.appendChild(row);
  });
}
function renderCard(){
  const c=CANDIDATES[cur], r=rec(cur);
  const st=r.status;
  const stTag = st==="approved"?`<span class="status-tag ok">Approved</span>`:
    st==="rejected"?`<span class="status-tag no">Rejected</span>`:
    `<span class="status-tag pend">Pending</span>`;
  $("#card").innerHTML = `
    <div class="top">
      <span class="idx">${cur+1} / ${CANDIDATES.length}</span>
      <span class="chip src">${c.source}</span>
      <span class="chip">${(r.strand||c.strand).replace("_"," ")}</span>
      <span class="chip">${r.band||c.grade_band}</span>
      <span class="chip">public domain</span>
      ${c.time_varying?`<span class="chip tv">◷ time-varying</span>`:""}
      ${stTag}
    </div>
    <div class="body">
      <p class="q">${esc(c.q)}</p>
      ${c.time_varying?`<div class="warnbox"><span>◷</span><span><b>Time-varying answer.</b>
        Teach the concept and point to an official lookup — never bake in the current fact.
        Edit the drafted answer so it does that.</span></div>`:""}
      <label class="fld" for="ans">Drafted answer — edit as needed</label>
      <textarea id="ans" class="ans">${esc(r.a)}</textarea>
      <div class="meta">
        <div class="m"><label for="selStrand">Strand</label>
          <select id="selStrand">${STRANDS.map(s=>`<option ${((r.strand||c.strand)===s)?"selected":""}>${s}</option>`).join("")}</select></div>
        <div class="m"><label for="selBand">Grade band</label>
          <select id="selBand">${BANDS.map(b=>`<option ${((r.band||c.grade_band)===b)?"selected":""}>${b}</option>`).join("")}</select></div>
        <div class="m"><label>Flag</label>
          <label class="tvtoggle"><input type="checkbox" id="tvChk" ${r.tv?"checked":""}> time-varying</label></div>
      </div>
      <div class="spacer"></div>
      <label class="fld" for="note">Reviewer note (optional)</label>
      <textarea id="note" class="note" placeholder="e.g. reworded for a 6th-grade reading level">${esc(r.note)}</textarea>
    </div>`;
  $("#ans").addEventListener("input",e=>{rec(cur).a=e.target.value;save();});
  $("#note").addEventListener("input",e=>{rec(cur).note=e.target.value;save();});
  $("#selStrand").addEventListener("change",e=>{rec(cur).strand=e.target.value;save();renderList();});
  $("#selBand").addEventListener("change",e=>{rec(cur).band=e.target.value;save();renderList();});
  $("#tvChk").addEventListener("change",e=>{rec(cur).tv=e.target.checked;save();});
}
function renderStats(){
  let ok=0,no=0;
  CANDIDATES.forEach((c,i)=>{const s=rec(i).status; if(s==="approved")ok++; else if(s==="rejected")no++;});
  const n=CANDIDATES.length, done=ok+no, pend=n-done;
  $("#mA").style.width=(ok/n*100)+"%"; $("#mR").style.width=(no/n*100)+"%";
  $("#mReviewed").textContent=`${done}/${n} reviewed`;
  $("#mPct").textContent=Math.round(done/n*100)+"%";
  $("#cOk").textContent=ok+" approved"; $("#cNo").textContent=no+" rejected"; $("#cPend").textContent=pend+" pending";
}
function renderAll(){ renderList(); renderCard(); renderStats();
  const active=document.querySelector(".row.active"); if(active) active.scrollIntoView({block:"nearest"}); }
function esc(s){return (s||"").replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;");}

function decide(status){ rec(cur).status=status; save();
  const vis=visible(); const pos=vis.indexOf(cur);
  if(pos>-1 && pos<vis.length-1){ cur=vis[pos+1]; } else { const nxt=CANDIDATES.findIndex((c,i)=>rec(i).status==="pending"); if(nxt>-1)cur=nxt; }
  renderAll();
}
function step(d){ const vis=visible(); let pos=vis.indexOf(cur);
  if(pos<0){cur=vis[0]??cur;} else {pos=Math.max(0,Math.min(vis.length-1,pos+d)); cur=vis[pos];} renderAll(); }

$("#approveBtn").onclick=()=>decide("approved");
$("#rejectBtn").onclick=()=>decide("rejected");
$("#skipBtn").onclick=()=>decide("pending");
$("#prevBtn").onclick=()=>step(-1);
$("#nextBtn").onclick=()=>step(1);
document.querySelectorAll("#filters button").forEach(b=>b.onclick=()=>{
  filter=b.dataset.f; document.querySelectorAll("#filters button").forEach(x=>x.setAttribute("aria-pressed", x===b));
  const vis=visible(); if(vis.length&&!vis.includes(cur))cur=vis[0]; renderAll();
});
$("#who").value = state.__who || "";
$("#who").addEventListener("input",e=>{state.__who=e.target.value;save();});
$("#resetBtn").onclick=()=>{ if(confirm("Clear ALL review decisions stored in this browser?")){ localStorage.removeItem(KEY); state=load(); renderAll(); } };

$("#exportBtn").onclick=()=>{
  const who=state.__who||"";
  const lines=CANDIDATES.map((c,i)=>{
    const r=rec(i);
    return JSON.stringify({
      subject:"civics", grade_band:r.band||c.grade_band, strand:r.strand||c.strand,
      time_varying:!!r.tv, source:c.source, license:c.license, attribution:c.attribution||"",
      review_status:r.status, reviewer:who, review_notes:r.note||"",
      messages:[{role:"system",content:SYS},{role:"user",content:c.q},{role:"assistant",content:r.a}]
    });
  });
  const blob=new Blob([lines.join("\n")+"\n"],{type:"application/x-ndjson"});
  const a=document.createElement("a"); a.href=URL.createObjectURL(blob);
  a.download="reviewed_civics_candidates.jsonl"; a.click(); URL.revokeObjectURL(a.href);
};

document.addEventListener("keydown",e=>{
  const t=e.target.tagName;
  if(t==="TEXTAREA"||t==="INPUT"||t==="SELECT") return;
  const k=e.key.toLowerCase();
  if(k==="a"){decide("approved");e.preventDefault();}
  else if(k==="r"){decide("rejected");e.preventDefault();}
  else if(k==="s"){decide("pending");e.preventDefault();}
  else if(k==="j"||k==="arrowleft"){step(-1);e.preventDefault();}
  else if(k==="k"||k==="arrowright"){step(1);e.preventDefault();}
  else if(k==="e"){const a=$("#ans");if(a){a.focus();e.preventDefault();}}
});

renderAll();
</script>
"""

html = TEMPLATE.replace("__DATA__", json.dumps(data)).replace("__SYS__", json.dumps(cs.CIVICS_SYS))
out = sys.argv[1] if len(sys.argv) > 1 else "/home/ashwinvbalakrishnan1/.claude/jobs/876fd82c/tmp/civics_review.html"
with open(out, "w") as f:
    f.write(html)
print(f"wrote {out} with {len(data)} candidates, {sum(d['time_varying'] for d in data)} time-varying")
