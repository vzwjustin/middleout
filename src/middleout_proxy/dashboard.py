from __future__ import annotations

_DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>MiddleOut</title>
<style>
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
:root{
  --bg:#0b0d12;--surface:#13171f;--surface-2:#171c25;--border:#222936;--border-2:#2a3242;
  --text:#e6edf3;--sub:#8a94a6;--muted:#525a6b;
  --blue:#3b82f6;--blue-soft:rgba(59,130,246,.16);--blue-line:rgba(59,130,246,.45);
  --green:#3fb950;--red:#f85149;--amber:#f59e0b;--cyan:#22d3ee;--violet:#a78bfa;
  --sans:system-ui,-apple-system,'Segoe UI',Inter,sans-serif;
  --mono:ui-monospace,'SF Mono','JetBrains Mono','Cascadia Code',Menlo,monospace;
  --ease:cubic-bezier(.2,.7,.2,1);
}
html,body{background:var(--bg)}
body{color:var(--text);font-family:var(--sans);font-size:16px;line-height:1.6;min-height:100vh;padding:40px 40px 56px;max-width:1400px;margin:0 auto;-webkit-font-smoothing:antialiased}

/* --- header --- */
.hdr{position:relative;display:flex;align-items:center;justify-content:space-between;padding:18px 22px 20px;margin:-12px -22px 36px;border-radius:14px;background:linear-gradient(180deg,rgba(59,130,246,.06) 0%,rgba(59,130,246,0) 70%);border:1px solid var(--border);overflow:hidden}
.hdr::after{content:"";position:absolute;left:0;right:0;bottom:-1px;height:1px;background:linear-gradient(90deg,transparent 0%,var(--blue-line) 20%,var(--blue-line) 80%,transparent 100%);opacity:.7;animation:slide 8s var(--ease) infinite}
@keyframes slide{0%,100%{transform:translateX(-12%)}50%{transform:translateX(12%)}}
.brand{display:flex;align-items:center;gap:10px;font-size:17px;font-weight:600;letter-spacing:-.01em}
.glyph{font-family:var(--mono);font-size:14px;font-weight:700;color:var(--blue);background:var(--blue-soft);padding:3px 8px;border-radius:6px;border:1px solid var(--blue-line);line-height:1}
.brand small{font-size:12px;font-weight:500;color:var(--sub);letter-spacing:.04em;text-transform:uppercase}
.hdr-right{display:flex;align-items:center;gap:14px}
.up-chip{display:flex;align-items:center;gap:6px;font-family:var(--mono);font-size:12px;color:var(--sub);padding:4px 8px;border:1px solid var(--border-2);border-radius:6px;background:var(--surface)}
.up-chip::before{content:"";width:5px;height:5px;border-radius:50%;background:var(--muted)}
.pill{display:flex;align-items:center;gap:8px;font-size:13px;color:var(--sub);font-variant-numeric:tabular-nums}
.dot{position:relative;width:8px;height:8px;border-radius:50%;background:var(--green);box-shadow:0 0 6px rgba(63,185,80,.55)}
.dot.err{background:var(--red);box-shadow:0 0 6px rgba(248,81,73,.55)}
.dot.fetching::after{content:"";position:absolute;inset:-4px;border-radius:50%;border:1.5px solid currentColor;color:var(--blue);opacity:.6;animation:pulse 1.1s var(--ease) infinite}
@keyframes pulse{0%{transform:scale(.6);opacity:.7}100%{transform:scale(1.8);opacity:0}}

/* --- sections --- */
section{margin-bottom:28px}
.sh{display:flex;align-items:center;gap:8px;font-size:12px;text-transform:uppercase;letter-spacing:.1em;color:var(--muted);margin-bottom:12px;font-weight:600}
.sh::after{content:"";flex:1;height:1px;background:linear-gradient(90deg,var(--border) 0%,transparent 100%)}

/* --- metrics grid --- */
.grid{display:grid;grid-template-columns:repeat(3,1fr);gap:1px;background:var(--border);border:1px solid var(--border);border-radius:12px;overflow:hidden}
.cell{position:relative;background:var(--surface);padding:18px 20px;transition:background .2s var(--ease)}
.cell:hover{background:var(--surface-2)}
.cell[data-tip]{cursor:help}
.cl{font-size:12px;color:var(--sub);margin-bottom:8px;letter-spacing:.02em;display:flex;align-items:center;gap:5px}
.cl::after{content:"i";display:inline-flex;align-items:center;justify-content:center;width:12px;height:12px;border-radius:50%;border:1px solid var(--muted);color:var(--muted);font-size:9px;font-family:var(--mono);font-style:italic;opacity:.6}
.cv{font-family:var(--mono);font-size:28px;font-weight:600;line-height:1;letter-spacing:-.01em;transition:opacity .22s var(--ease),color .25s var(--ease)}
.cv.fading{opacity:.25}
.cv.red{color:var(--red)}

/* --- compression overview card --- */
.card{position:relative;background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:20px 22px;transition:border-color .2s var(--ease),transform .2s var(--ease)}
.brow{display:flex;justify-content:space-between;align-items:flex-start;gap:20px;margin-bottom:14px}
.brow > div{min-width:0}
.bpct{font-family:var(--mono);font-size:22px;font-weight:600;letter-spacing:-.01em}
.bpct.small{font-size:16px}
.track{position:relative;height:4px;background:var(--border);border-radius:3px;margin-bottom:14px;overflow:hidden}
.fill{height:100%;background:linear-gradient(90deg,var(--blue) 0%,#60a5fa 100%);border-radius:3px;transition:width .6s var(--ease);width:0%}
.bsub{display:flex;justify-content:space-between;align-items:center;gap:18px;font-size:12px;color:var(--sub);flex-wrap:wrap}
.bsub b{font-family:var(--mono);color:var(--text);font-weight:500}
.spark{display:flex;align-items:flex-end;gap:2px;height:18px}
.spark .bar{width:3px;background:var(--blue-line);border-radius:1px;transition:height .35s var(--ease),background .25s var(--ease)}
.spark .bar.lit{background:var(--blue)}

/* --- engines --- */
.engines{display:grid;grid-template-columns:1fr 1fr;gap:12px}
.eng{position:relative;background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:16px 18px;transition:border-color .2s var(--ease),transform .2s var(--ease),background .2s var(--ease);overflow:hidden}
.eng::before{content:"";position:absolute;left:0;top:0;bottom:0;width:2px;background:var(--accent,var(--blue));opacity:.7}
.eng:hover{transform:translateY(-1px);border-color:var(--border-2);background:var(--surface-2)}
.eng:hover::before{opacity:1;box-shadow:0 0 12px var(--accent,var(--blue))}
.eng[data-accent="blue"]{--accent:var(--blue)}
.eng[data-accent="cyan"]{--accent:var(--cyan)}
.eng[data-accent="amber"]{--accent:var(--amber)}
.eng[data-accent="violet"]{--accent:var(--violet)}
.eng.off{opacity:.78}
.erow{display:flex;align-items:center;justify-content:space-between;gap:10px;margin-bottom:4px}
.etitle{display:flex;align-items:center;gap:8px;font-size:15px;font-weight:600}
.etitle .edot{width:6px;height:6px;border-radius:50%;background:var(--accent,var(--blue));box-shadow:0 0 6px var(--accent,var(--blue))}
.edesc{font-size:13px;color:var(--sub);line-height:1.5;margin-top:2px;margin-bottom:12px}
.levels{display:flex;gap:2px;padding:2px;background:var(--border);border-radius:8px;border:1px solid var(--border-2)}
.lvl{flex:1;padding:5px 6px;font-size:13px;font-family:var(--mono);font-weight:500;color:var(--sub);text-align:center;border-radius:6px;cursor:pointer;text-transform:lowercase;letter-spacing:.02em;transition:background .18s var(--ease),color .18s var(--ease);user-select:none;background:transparent;border:none}
.lvl:hover{color:var(--text)}
.lvl.active{background:var(--accent,var(--blue));color:#fff;box-shadow:0 1px 0 rgba(0,0,0,.25)}
.eng.off .levels{opacity:.45;pointer-events:none}
.eng.off .lvl.active{background:var(--muted)}

/* --- toggle --- */
.tog{position:relative;width:34px;height:20px;border-radius:11px;background:var(--border);border:1px solid var(--border-2);transition:background .25s var(--ease),border-color .25s var(--ease);cursor:pointer;display:inline-block;flex-shrink:0}
.tog.on{background:var(--accent,var(--blue));border-color:var(--accent,var(--blue));box-shadow:0 0 0 3px rgba(59,130,246,.12)}
.eng[data-accent="cyan"] .tog.on{box-shadow:0 0 0 3px rgba(34,211,238,.13)}
.eng[data-accent="amber"] .tog.on{box-shadow:0 0 0 3px rgba(245,158,11,.13)}
.eng[data-accent="violet"] .tog.on{box-shadow:0 0 0 3px rgba(167,139,250,.14)}
.thumb{position:absolute;top:2px;left:2px;width:14px;height:14px;border-radius:50%;background:#fff;transition:transform .25s var(--ease),width .15s var(--ease)}
.tog.on .thumb{transform:translateX(14px)}
.tog.bounce .thumb{animation:bounce .35s var(--ease)}
@keyframes bounce{0%{transform:translateX(14px) scale(1)}40%{transform:translateX(14px) scale(1.15)}100%{transform:translateX(14px) scale(1)}}

/* --- config table --- */
table{width:100%;border-collapse:collapse}
tr{border-bottom:1px solid var(--border)}
tr:last-child{border-bottom:none}
td{padding:12px 0;font-size:13px;vertical-align:middle}
td:first-child{color:var(--sub)}
td:last-child{text-align:right}
.badge{display:inline-flex;align-items:center;gap:5px;padding:3px 9px;border-radius:6px;font-size:11px;font-family:var(--mono);background:var(--surface-2);color:var(--text);border:1px solid var(--border-2);cursor:help}
.badge.subtle{color:var(--sub)}

/* --- footer --- */
footer{margin-top:36px;padding-top:16px;border-top:1px solid var(--border);display:flex;justify-content:space-between;align-items:center;font-size:11.5px;color:var(--muted)}
footer .v{font-family:var(--mono)}
footer .v b{color:var(--sub);font-weight:500}
footer .ts{font-family:var(--mono)}

/* --- tooltip system --- */
[data-tip]{position:relative}
[data-tip]::after{
  content:attr(data-tip);position:absolute;left:50%;top:calc(100% + 8px);transform:translateX(-50%) translateY(-4px);
  background:#0a0d13;color:var(--text);font-size:14px;font-weight:400;line-height:1.5;
  padding:8px 11px;border-radius:7px;border:1px solid var(--border-2);
  white-space:pre-line;max-width:260px;width:max-content;text-align:left;
  box-shadow:0 8px 24px rgba(0,0,0,.45),0 2px 6px rgba(0,0,0,.3);
  opacity:0;pointer-events:none;visibility:hidden;z-index:30;
  transition:opacity .15s var(--ease),transform .15s var(--ease),visibility 0s linear .28s;
}
[data-tip]:hover::after{opacity:1;visibility:visible;transform:translateX(-50%) translateY(0);transition-delay:.28s,.28s,0s}
[data-tip][data-tip-pos="top"]::after{top:auto;bottom:calc(100% + 8px);transform:translateX(-50%) translateY(4px)}
[data-tip][data-tip-pos="top"]:hover::after{transform:translateX(-50%) translateY(0)}
[data-tip][data-tip-pos="left"]::after{top:50%;left:auto;right:calc(100% + 8px);transform:translateY(-50%) translateX(4px)}
[data-tip][data-tip-pos="left"]:hover::after{transform:translateY(-50%) translateX(0)}

@media(min-width:1100px){
  .grid{grid-template-columns:repeat(4,1fr)}
}
@media(max-width:640px){
  body{padding:24px 16px}
  .grid{grid-template-columns:1fr 1fr}
  .engines{grid-template-columns:1fr}
  .hdr{flex-direction:column;align-items:flex-start;gap:10px}
}
@media(prefers-reduced-motion:reduce){
  *,*::before,*::after{animation-duration:.001ms!important;transition-duration:.001ms!important}
  .hdr::after{animation:none;opacity:.4}
}
.badge.warn{background:rgba(245,158,11,.12);color:var(--amber);border-color:rgba(245,158,11,.3)}

/* --- observability extension --- */
.dual{display:grid;grid-template-columns:2fr 1fr;gap:14px}
@media(max-width:900px){.dual{grid-template-columns:1fr}}
.gauge{position:relative;width:100%;height:8px;background:var(--border);border-radius:4px;overflow:hidden}
.gauge .gf{height:100%;background:linear-gradient(90deg,var(--green) 0%,var(--amber) 70%,var(--red) 100%);transition:width .4s var(--ease);width:0%}
.chart-wrap{position:relative;width:100%;height:170px}
.chart-wrap canvas{display:block;width:100%;height:100%}
.legend{display:flex;gap:14px;font-size:12px;color:var(--sub);margin-top:8px;flex-wrap:wrap}
.legend .sw{display:inline-block;width:10px;height:10px;border-radius:2px;vertical-align:middle;margin-right:6px}
.eng-bars{display:flex;flex-direction:column;gap:8px;min-height:120px}
.eb{display:grid;grid-template-columns:120px 1fr auto;gap:10px;align-items:center;font-size:13px}
.eb .ebn{font-family:var(--mono);color:var(--sub);overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.eb .ebt{font-family:var(--mono);color:var(--text);font-variant-numeric:tabular-nums}
.eb .ebbar{position:relative;height:6px;background:var(--border);border-radius:3px;overflow:hidden}
.eb .ebbarfill{height:100%;background:var(--blue);border-radius:3px;transition:width .35s var(--ease);width:0%}
.eb[data-engine^="caveman"] .ebbarfill{background:var(--amber)}
.eb[data-engine^="rtk"] .ebbarfill{background:var(--violet)}
.eb[data-engine^="jl"] .ebbarfill{background:var(--cyan)}
.eb[data-engine="middle-out"] .ebbarfill{background:var(--blue)}
.eb[data-engine$="-response"] .ebbarfill{background:linear-gradient(90deg,var(--green),var(--cyan))}
.tbl-wrap{overflow-x:auto;border:1px solid var(--border);border-radius:12px;background:var(--surface)}
.tbl{width:100%;border-collapse:collapse;font-size:13px}
.tbl thead th{position:sticky;top:0;background:var(--surface-2);font-weight:600;color:var(--sub);text-align:left;padding:9px 12px;font-size:11.5px;text-transform:uppercase;letter-spacing:.06em;border-bottom:1px solid var(--border)}
.tbl td{padding:9px 12px;border-bottom:1px solid var(--border);vertical-align:top;font-family:var(--mono)}
.tbl tr:last-child td{border-bottom:none}
.tbl tr.rrow{cursor:pointer;transition:background .15s var(--ease)}
.tbl tr.rrow:hover{background:var(--surface-2)}
.tbl tr.detail td{background:#0e1219;padding:0;border-top:1px solid var(--border-2)}
.tbl tr.detail .det-inner{padding:10px 14px;color:var(--sub);font-size:12px;display:flex;gap:18px;flex-wrap:wrap}
.tbl tr.detail .det-eng{display:inline-flex;align-items:center;gap:6px;background:var(--surface);border:1px solid var(--border-2);padding:3px 8px;border-radius:6px;font-family:var(--mono);font-size:12px;color:var(--text)}
.tbl tr.detail .det-eng b{color:var(--blue);font-weight:600}
.tbl .nodata{text-align:center;color:var(--muted);padding:20px;font-family:var(--sans);font-size:13px;font-style:italic}
.tbl .st-2xx{color:var(--green)}
.tbl .st-4xx{color:var(--amber)}
.tbl .st-5xx{color:var(--red)}
.tbl .st-na{color:var(--muted)}
.lat-tiles{display:grid;grid-template-columns:1fr 1fr;gap:1px;background:var(--border);border:1px solid var(--border);border-radius:12px;overflow:hidden}
.lat-tiles .cell{padding:14px 16px}
.lat-tiles .cv{font-size:22px}

/* --- brain (Phase 2/3/4) --- */
.brain-grid{display:grid;grid-template-columns:1fr 1fr;gap:14px}
@media(max-width:900px){.brain-grid{grid-template-columns:1fr}}
.kpi-row{display:grid;grid-template-columns:repeat(3,1fr);gap:1px;background:var(--border);border:1px solid var(--border);border-radius:12px;overflow:hidden;margin-bottom:14px}
.kpi-row .cell{padding:14px 16px}
.kpi-row .cv{font-size:22px}
.chip{display:inline-flex;align-items:center;gap:6px;padding:3px 10px;border-radius:999px;font-size:11.5px;font-family:var(--mono);background:var(--surface-2);color:var(--sub);border:1px solid var(--border-2);margin:2px 4px 2px 0}
.chip.on{color:var(--green);border-color:rgba(63,185,80,.32);background:rgba(63,185,80,.08)}
.chip.warn{color:var(--amber);border-color:rgba(245,158,11,.32);background:rgba(245,158,11,.08)}
.chip.off{color:var(--muted)}
.cost-bars{display:flex;flex-direction:column;gap:8px;min-height:80px;margin-top:8px}
.cb{display:grid;grid-template-columns:160px 1fr auto;gap:10px;align-items:center;font-size:13px}
.cb .cbn{font-family:var(--mono);color:var(--sub);overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.cb .cbbar{position:relative;height:6px;background:var(--border);border-radius:3px;overflow:hidden}
.cb .cbbarfill{height:100%;background:linear-gradient(90deg,var(--blue),var(--cyan));border-radius:3px;transition:width .35s var(--ease);width:0%}
.cb .cbv{font-family:var(--mono);color:var(--text);font-variant-numeric:tabular-nums}

/* --- keyboard hint --- */
.kbd{font-family:var(--mono);font-size:11px;padding:2px 6px;border-radius:4px;background:var(--surface-2);border:1px solid var(--border-2);color:var(--sub);box-shadow:0 1px 0 var(--border-2)}

</style>
</head>
<body>

<div class="hdr">
  <div class="brand">
    <span class="glyph" data-tip="MiddleOut — token compression proxy for Claude">M/O</span>
    <span>MiddleOut Proxy <small>v0.2.0</small></span>
  </div>
  <div class="hdr-right">
    <span class="up-chip" id="cfg-up" data-tip-pos="left" data-tip="Upstream URL — all requests proxy through this host.">upstream</span>
    <div class="pill"><div class="dot" id="dot"></div><span id="stxt">connecting</span></div>
  </div>
</div>

<section>
  <div class="sh">Traffic</div>
  <div class="grid">
    <div class="cell" data-tip="All requests proxied since startup,&#10;including failed ones."><div class="cl">Total requests</div><div class="cv" id="m-total">-</div></div>
    <div class="cell" data-tip="Requests where at least one compression&#10;engine modified the payload."><div class="cl">Compressed</div><div class="cv" id="m-comp">-</div></div>
    <div class="cell" data-tip="Upstream 5xx responses or connection&#10;failures since startup."><div class="cl">Errors</div><div class="cv" id="m-err">-</div></div>
    <div class="cell" data-tip="Characters removed from request bodies&#10;before sending upstream."><div class="cl">Chars saved (in)</div><div class="cv" id="m-cin">-</div></div>
    <div class="cell" data-tip="Characters removed from response bodies&#10;before returning to the client."><div class="cl">Chars saved (out)</div><div class="cv" id="m-cout">-</div></div>
    <div class="cell" data-tip="Seconds since the proxy process started."><div class="cl">Uptime</div><div class="cv" id="m-up">-</div></div>
  </div>
</section>

<section>
  <div class="sh">Compression</div>
  <div class="card">
    <div class="brow">
      <div data-tip="Compressed requests ÷ total requests."><div class="cl">Requests compressed</div><div class="bpct"><span id="cpct" class="cv" style="font-size:22px">-</span></div></div>
      <div style="text-align:right" data-tip-pos="left" data-tip="Total chars saved ÷ 4 — rough English&#10;token estimate. Real tokenizers vary."><div class="cl" style="justify-content:flex-end">Est. tokens saved</div><div class="bpct small"><span id="tok" class="cv" style="font-size:16px">-</span></div></div>
    </div>
    <div class="track" data-tip="Compression rate across all requests."><div class="fill" id="cfill"></div></div>
    <div class="bsub">
      <div>Input <b id="s-cin">-</b> &nbsp;·&nbsp; Output <b id="s-cout">-</b></div>
      <div class="spark" id="spark" data-tip-pos="top" data-tip="Recent request rate.&#10;Each bar = requests during one 4s poll.&#10;Newest on the right."></div>
    </div>
  </div>
</section>

<section>
  <div class="sh">Cache</div>
  <div class="card">
  <div class="brow">
  <div data-tip="Local compression-result cache hits over lookups.&#10;Skips re-running middle-out/caveman/rtk on text&#10;we already compressed this session.&#10;Independent from Anthropic&apos;s native prompt cache."><div class="cl">Local cache hit rate</div><div class="bpct"><span id="ch-pct" class="cv" style="font-size:22px">-</span></div></div>
  <div style="text-align:right" data-tip-pos="left" data-tip="Blocks left untouched because they sit at-or-before&#10;an Anthropic cache_control marker. Mutating them&#10;would invalidate the upstream prompt cache."><div class="cl" style="justify-content:flex-end">Cache-protected blocks</div><div class="bpct small"><span id="ch-prot" class="cv" style="font-size:16px">-</span></div></div>
  </div>
  <div class="track" data-tip="Local LRU fill against configured max entries."><div class="fill" id="cache-fill" style="background:var(--cyan)"></div></div>
  <div class="bsub">
  <div>Hits <b id="ch-hits">-</b> &nbsp;&middot;&nbsp; Misses <b id="ch-misses">-</b> &nbsp;&middot;&nbsp; Size <b id="ch-size">-</b></div>
  <div><span class="badge subtle" id="ch-preserve" data-tip-pos="top" data-tip="When ON, proxy refuses to mutate any block&#10;at-or-before an Anthropic cache_control marker.&#10;Keeps upstream prompt cache valid.">anthropic-cache: -</span></div>
  </div>
  </div>
</section>

<section>
  <div class="sh">Brain &middot; Phase 2/3/4</div>
  <div class="kpi-row">
    <div class="cell" data-tip="Total spend tracked across all providers&#10;based on token usage and per-model pricing."><div class="cl">Total spend</div><div class="cv" id="b-cost">-</div></div>
    <div class="cell" data-tip="Total requests with cost attribution."><div class="cl">Costed requests</div><div class="cv" id="b-creq">-</div></div>
    <div class="cell" data-tip="Requests where no pricing entry matched the&#10;requested model (cost = 0 for these)."><div class="cl">Unmatched models</div><div class="cv" id="b-cunm">-</div></div>
  </div>
  <div class="brain-grid">
    <div class="card">
      <div class="brow">
        <div><div class="cl">Cost by model</div></div>
        <div style="text-align:right"><div class="cl" style="justify-content:flex-end">Top 6</div></div>
      </div>
      <div class="cost-bars" id="b-cost-bars">
        <div class="nodata" style="font-family:var(--sans);color:var(--muted);font-style:italic;padding:8px 0">No costed traffic yet.</div>
      </div>
    </div>
    <div class="card">
      <div class="brow">
        <div data-tip="Two-tier response cache state.&#10;L1 = SHA256 exact-match.&#10;L2 = embedding similarity (Qdrant)."><div class="cl">Response cache</div></div>
        <div style="text-align:right"><span class="chip" id="b-l1">L1 -</span><span class="chip" id="b-l2">L2 -</span></div>
      </div>
      <div class="bsub" style="margin-top:8px">
        <div>L2 lookups <b id="b-l2-lk">-</b> &middot; hits <b id="b-l2-hit">-</b> &middot; threshold <b id="b-l2-th">-</b></div>
      </div>
      <div class="brow" style="margin-top:14px">
        <div data-tip="Adapter scaffolds available for routing.&#10;Use X-Brain-Model-Hint header to override."><div class="cl">Providers</div></div>
        <div style="text-align:right" data-tip-pos="left" data-tip="Budget cap (chars/tokens). Once exceeded,&#10;requests still flow but counter is flagged."><div class="cl" style="justify-content:flex-end">Budget</div></div>
      </div>
      <div class="bsub" style="margin-top:6px">
        <div id="b-providers"><span class="chip off">-</span></div>
        <div><span class="chip" id="b-budget">-</span></div>
      </div>
    </div>
  </div>
</section>

<section>
  <div class="sh">Engines</div>
  <div class="engines">

    <div class="eng" data-accent="blue" id="eng-input">
      <div class="erow">
        <div class="etitle"><span class="edot"></span>Middle-Out</div>
        <div class="tog" id="t-input_compression" data-tip-pos="left" data-tip="Master switch. When off, request bodies&#10;pass through unchanged."><div class="thumb"></div></div>
      </div>
      <div class="edesc">Master input pipeline. Disables all request-side compression when off.</div>
    </div>

    <div class="eng" data-accent="cyan" id="eng-jl">
      <div class="erow">
        <div class="etitle"><span class="edot"></span>JL Dedupe</div>
        <div class="tog" id="t-jl_dedupe" data-tip-pos="left" data-tip="Removes near-duplicate text blocks within&#10;the same request using a Johnson-Lindenstrauss&#10;sketch. Lossless on unique content."><div class="thumb"></div></div>
      </div>
      <div class="edesc">Strips near-duplicate chunks inside a single request via JL sketches.</div>
    </div>

    <div class="eng" data-accent="amber" id="eng-cv">
      <div class="erow">
        <div class="etitle"><span class="edot"></span>Caveman</div>
        <div class="tog" id="t-caveman" data-tip-pos="left" data-tip="Drops articles, filler words, and&#10;pleasantries. Lossy — may degrade&#10;model output quality."><div class="thumb"></div></div>
      </div>
      <div class="edesc">Lossy. Strips articles &amp; filler. Choose how aggressive.</div>
      <div class="levels" id="lv-caveman">
        <button class="lvl" data-lvl="lite"       data-tip-pos="top" data-tip="Removes &quot;the&quot;, &quot;a&quot;, &quot;an&quot;. Minimal&#10;quality impact, modest savings.">lite</button>
        <button class="lvl" data-lvl="standard"   data-tip-pos="top" data-tip="Drops articles + common filler&#10;(&quot;please&quot;, &quot;just&quot;, &quot;basically&quot;).">standard</button>
        <button class="lvl" data-lvl="aggressive" data-tip-pos="top" data-tip="Standard + auxiliaries and pleasantries.&#10;Can affect tone-sensitive outputs.">aggressive</button>
        <button class="lvl" data-lvl="ultra"      data-tip-pos="top" data-tip="Telegraph mode. Strips most non-content&#10;words. Highest savings, highest risk.">ultra</button>
      </div>
    </div>

    <div class="eng" data-accent="violet" id="eng-rtk">
      <div class="erow">
        <div class="etitle"><span class="edot"></span>RTK</div>
        <div class="tog" id="t-rtk" data-tip-pos="left" data-tip="Dictionary-based phrase shortening&#10;(&quot;function&quot;→&quot;fn&quot;). Lossy if the model&#10;doesn't recognize the shortened form."><div class="thumb"></div></div>
      </div>
      <div class="edesc">Lossy. Dictionary substitutions (&quot;function&quot;&rarr;&quot;fn&quot;).</div>
      <div class="levels" id="lv-rtk">
        <button class="lvl" data-lvl="minimal"    data-tip-pos="top" data-tip="Only the safest, well-known abbreviations&#10;(&quot;function&quot;→&quot;fn&quot;).">minimal</button>
        <button class="lvl" data-lvl="standard"   data-tip-pos="top" data-tip="Broader dictionary covering common&#10;programming and English phrases.">standard</button>
        <button class="lvl" data-lvl="aggressive" data-tip-pos="top" data-tip="Full dictionary, including ambiguous&#10;substitutions. Test before relying on it.">aggressive</button>
      </div>
    </div>

  </div>
</section>


<section>
  <div class="sh">Latency &amp; Errors</div>
  <div class="dual">
    <div class="card">
      <div class="brow">
        <div data-tip="Rolling error rate over the time-series window&#10;(default 60 min). Errors are upstream 5xx or&#10;proxy connection failures."><div class="cl">Error rate</div><div class="bpct"><span id="o-err-pct" class="cv" style="font-size:22px">-</span></div></div>
        <div style="text-align:right" data-tip-pos="left" data-tip="Errors observed in the current rolling&#10;window."><div class="cl" style="justify-content:flex-end">Errors (window)</div><div class="bpct small"><span id="o-err-count" class="cv" style="font-size:16px">-</span></div></div>
      </div>
      <div class="gauge" data-tip="0% (green) -> 10% (red).&#10;Bar reflects current rolling error rate."><div class="gf" id="o-err-gauge"></div></div>
      <div class="bsub" style="margin-top:10px">
        <div>Window <b id="o-err-window">-</b> &middot; Requests <b id="o-err-req">-</b></div>
      </div>
    </div>
    <div class="card">
      <div class="lat-tiles">
        <div class="cell" data-tip="Median request latency across all requests&#10;since startup (fixed-bin histogram, ms)."><div class="cl">p50 latency</div><div class="cv" id="o-p50">-</div></div>
        <div class="cell" data-tip="95th percentile request latency across all&#10;requests since startup."><div class="cl">p95 latency</div><div class="cv" id="o-p95">-</div></div>
      </div>
    </div>
  </div>
</section>

<section>
  <div class="sh">Chars saved (last 60 min)</div>
  <div class="card">
    <div class="chart-wrap" data-tip-pos="top" data-tip="Live: input vs output chars saved per minute.&#10;Empty until requests arrive."><canvas id="o-chart-saved" width="800" height="170"></canvas></div>
    <div class="legend">
      <span><span class="sw" style="background:var(--blue)"></span>input chars saved/min</span>
      <span><span class="sw" style="background:var(--green)"></span>output chars saved/min</span>
      <span><span class="sw" style="background:var(--red)"></span>errors/min</span>
    </div>
  </div>
</section>

<section>
  <div class="sh">Engine attribution</div>
  <div class="card">
    <div class="eng-bars" id="o-eng-bars">
      <div class="nodata" style="font-family:var(--sans);color:var(--muted);font-style:italic;padding:8px 0">No engine activity yet.</div>
    </div>
  </div>
</section>

<section>
  <div class="sh">Recent requests</div>
  <div class="tbl-wrap">
    <table class="tbl" id="o-recent">
      <thead><tr><th>Path</th><th>Status</th><th>ms</th><th>Saved (in/out)</th><th>Request ID</th></tr></thead>
      <tbody id="o-recent-body"><tr><td class="nodata" colspan="5">No requests yet.</td></tr></tbody>
    </table>
  </div>
</section>

<section>
  <div class="sh">Config</div>
  <div class="card">
    <table>
      <tr><td>Upstream</td><td><span class="badge" id="cfg-up-2" data-tip-pos="left" data-tip="Where this proxy forwards traffic.">-</span></td></tr>
      <tr><td>Auth mode</td><td><span class="badge subtle" id="cfg-auth" data-tip-pos="left" data-tip="Subscription OAuth passthrough only —&#10;API keys are rejected.">-</span></td></tr>
      <tr><td>Output compression</td><td>
        <div class="tog" id="t-output_compression" data-tip-pos="left" data-tip="When on, the proxy may rewrite long text&#10;in responses. Can break tools that need&#10;exact output. Off by default."><div class="thumb"></div></div>
      </td></tr>
    </table>
  </div>
</section>

<footer>
  <div class="v"><b>middleout-claude-proxy</b> &middot; <span id="ver">v0.2.0</span></div>
  <div data-tip="Press R to refresh, I for input compression,&#10;O for output compression, J for JL dedupe,&#10;C for caveman.">shortcuts: <span class="kbd">R</span> <span class="kbd">I</span> <span class="kbd">O</span> <span class="kbd">J</span> <span class="kbd">C</span> &middot; updated <span class="ts" id="ts">-</span></div>
</footer>

<script>
const $=id=>document.getElementById(id);
const RM=window.matchMedia&&window.matchMedia('(prefers-reduced-motion: reduce)').matches;
const SPARK_N=20;
const sparkData=[];

function fmt(n){if(n==null||isNaN(n))return'-';if(n>=1e9)return(n/1e9).toFixed(1)+'B';if(n>=1e6)return(n/1e6).toFixed(1)+'M';if(n>=1e3)return(n/1e3).toFixed(1)+'k';return''+n}
function fup(s){if(s==null)return'-';if(s<60)return s.toFixed(0)+'s';if(s<3600)return Math.floor(s/60)+'m '+Math.floor(s%60)+'s';if(s<86400)return Math.floor(s/3600)+'h '+Math.floor((s%3600)/60)+'m';return Math.floor(s/86400)+'d '+Math.floor((s%86400)/3600)+'h'}

function setVal(id,val){
  const el=$(id);if(!el)return;
  if(el.textContent===val)return;
  if(RM){el.textContent=val;return}
  el.classList.add('fading');
  setTimeout(()=>{el.textContent=val;el.classList.remove('fading')},180);
}

function renderSpark(){
  const el=$('spark');if(!el)return;
  const max=Math.max(1,...sparkData);
  const html=[];
  for(let i=0;i<SPARK_N;i++){
    const v=sparkData[i]||0;
    const h=Math.max(2,Math.round((v/max)*18));
    const lit=v>0?' lit':'';
    html.push('<div class="bar'+lit+'" style="height:'+h+'px"></div>');
  }
  el.innerHTML=html.join('');
}

let lastTotal=null;
function pushSpark(total){
  if(lastTotal!=null){
    const delta=Math.max(0,total-lastTotal);
    sparkData.push(delta);
    while(sparkData.length>SPARK_N)sparkData.shift();
    renderSpark();
  }else{
    for(let i=0;i<SPARK_N;i++)sparkData.push(0);
    renderSpark();
  }
  lastTotal=total;
}

function fetching(on){
  const d=$('dot');if(!d)return;
  if(on)d.classList.add('fetching');else d.classList.remove('fetching');
}

async function refreshStats(){
  fetching(true);
  try{
    const d=await fetch('/stats').then(r=>r.json());
    setVal('m-total',fmt(d.requests_total));
    setVal('m-comp',fmt(d.compressed_requests));
    const eEl=$('m-err');
    const errTxt=fmt(d.upstream_errors);
    if(eEl.textContent!==errTxt){
      if(RM){eEl.textContent=errTxt}else{eEl.classList.add('fading');setTimeout(()=>{eEl.textContent=errTxt;eEl.classList.remove('fading')},180)}
    }
    eEl.classList.toggle('red',d.upstream_errors>0);
    if(!eEl.classList.contains('red'))eEl.classList.remove('red');
    if(d.upstream_errors>0)eEl.classList.add('red');
    setVal('m-up',fup(d.uptime_s));
    setVal('m-cin',fmt(d.chars_saved_in));
    setVal('m-cout',fmt(d.chars_saved_out));
    const pct=d.requests_total>0?Math.round(d.compressed_requests/d.requests_total*100):0;
    setVal('cpct',pct+'%');
    $('cfill').style.width=pct+'%';
    const sv=(d.chars_saved_in||0)+(d.chars_saved_out||0);
    setVal('s-cin',fmt(d.chars_saved_in));
    setVal('s-cout',fmt(d.chars_saved_out));
    setVal('tok','~'+fmt(Math.round(sv/4)));
    $('ts').textContent=new Date().toLocaleTimeString();
    pushSpark(d.requests_total||0);
 const rc=d.result_cache||{hits:0,misses:0,size:0,max_entries:0};
 const cTot=(rc.hits||0)+(rc.misses||0);
 const cPct=cTot>0?Math.round((rc.hits||0)/cTot*100):0;
 setVal('ch-pct',cPct+'%');
 setVal('ch-hits',fmt(rc.hits||0));
 setVal('ch-misses',fmt(rc.misses||0));
 setVal('ch-size',(rc.size||0)+'/'+(rc.max_entries||0));
 setVal('ch-prot',fmt(d.protected_blocks||0));
 const fillPct=rc.max_entries>0?Math.round((rc.size||0)/rc.max_entries*100):0;
 const cfEl=$('cache-fill');if(cfEl)cfEl.style.width=fillPct+'%';
 const presEl=$('ch-preserve');
 if(presEl){const on=d.preserve_anthropic_cache!==false;presEl.textContent='anthropic-cache: '+(on?'preserved':'NOT preserved');presEl.classList.toggle('warn',!on);}
  }catch(e){}
  setTimeout(()=>fetching(false),250);
}

async function refreshHealth(){
  try{
    const d=await fetch('/healthz').then(r=>r.json());
    $('dot').className='dot'+(d.ok?'':' err');
    $('stxt').textContent=d.ok?'live':'error';
    const up=d.upstream||'-';
    const upShort=up.length>34?up.slice(0,32)+'…':up;
    $('cfg-up').textContent=upShort;
    $('cfg-up').setAttribute('data-tip',up);
    $('cfg-up-2').textContent=upShort;
    $('cfg-up-2').setAttribute('data-tip',up);
    $('cfg-auth').textContent=d.auth_mode||'-';
  }catch(e){
    $('dot').className='dot err';
    $('stxt').textContent='offline';
  }
}

function setTog(key,on,bounce){
  const el=$('t-'+key);if(!el)return;
  const was=el.classList.contains('on');
  el.classList.toggle('on',!!on);
  if(bounce && !was && on && !RM){el.classList.add('bounce');setTimeout(()=>el.classList.remove('bounce'),360)}
}

function setEngState(engId,on){
  const el=$(engId);if(!el)return;
  el.classList.toggle('off',!on);
}

function setLevels(group,active){
  const root=$('lv-'+group);if(!root)return;
  for(const b of root.querySelectorAll('.lvl')){
    b.classList.toggle('active',b.dataset.lvl===active);
  }
}

let lastSettings={};
async function refreshSettings(){
  try{
    const d=await fetch('/settings').then(r=>r.json());
    lastSettings=d;
    setTog('input_compression',!!d.input_compression);
    setTog('output_compression',!!d.output_compression);
    setTog('jl_dedupe',!!d.jl_dedupe);
    const cv=d.caveman||{};const rtk=d.rtk||{};
    setTog('caveman',!!cv.enabled);
    setTog('rtk',!!rtk.enabled);
    setEngState('eng-cv',!!cv.enabled);
    setEngState('eng-rtk',!!rtk.enabled);
    setEngState('eng-input',!!d.input_compression);
    setEngState('eng-jl',!!d.jl_dedupe);
    setLevels('caveman',cv.level||'standard');
    setLevels('rtk',rtk.level||'minimal');
  }catch(e){}
}

async function postSettings(patch){
  try{
    await fetch('/settings',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(patch)});
  }catch(e){}
  await refreshSettings();
}

function bindToggle(key,nested){
  const el=$('t-'+key);if(!el)return;
  el.addEventListener('click',()=>{
    if(nested){
      const cur=lastSettings[nested]||{};
      const next=!el.classList.contains('on');
      setTog(key,next,true);
      postSettings({[nested]:{...cur,enabled:next}});
    }else{
      const next=!el.classList.contains('on');
      setTog(key,next,true);
      postSettings({[key]:next});
    }
  });
}

function bindLevels(group,nested){
  const root=$('lv-'+group);if(!root)return;
  root.addEventListener('click',ev=>{
    const b=ev.target.closest('.lvl');if(!b)return;
    const eng=lastSettings[nested]||{};
    if(!eng.enabled)return;
    setLevels(group,b.dataset.lvl);
    postSettings({[nested]:{...eng,level:b.dataset.lvl}});
  });
}

bindToggle('input_compression');
bindToggle('output_compression');
bindToggle('jl_dedupe');
bindToggle('caveman','caveman');
bindToggle('rtk','rtk');
bindLevels('caveman','caveman');
bindLevels('rtk','rtk');

renderSpark();
refreshStats();refreshHealth();refreshSettings();
setInterval(refreshStats,1000);
setInterval(refreshHealth,30000);
setInterval(refreshSettings,10000);

// --- observability ---
const O_PALETTE={input:'#3b82f6',output:'#3fb950',err:'#f85149',axis:'#525a6b',grid:'#222936',text:'#8a94a6'};

function oFmtMs(v){if(v==null||isNaN(v)||v<=0)return'-';if(v<1)return v.toFixed(2)+'ms';if(v<1000)return Math.round(v)+'ms';return (v/1000).toFixed(1)+'s'}
function oFmtPct(v){if(v==null||isNaN(v))return'-';return v.toFixed(v>=10?0:1)+'%'}
function oFmtTs(ts){try{const d=new Date(ts*1000);return d.toLocaleTimeString();}catch(e){return '-';}}

function oDrawSavedChart(buckets){
  const canvas=document.getElementById('o-chart-saved');if(!canvas)return;
  // High-DPI scale.
  const cssW=canvas.clientWidth||canvas.width;const cssH=canvas.clientHeight||canvas.height;
  const dpr=window.devicePixelRatio||1;
  if(canvas.width!==cssW*dpr||canvas.height!==cssH*dpr){canvas.width=cssW*dpr;canvas.height=cssH*dpr;}
  const ctx=canvas.getContext('2d');ctx.setTransform(dpr,0,0,dpr,0,0);
  ctx.clearRect(0,0,cssW,cssH);
  const W=cssW,H=cssH;const padL=36,padR=8,padT=6,padB=18;const innerW=W-padL-padR,innerH=H-padT-padB;
  // Axes / grid
  ctx.strokeStyle=O_PALETTE.grid;ctx.lineWidth=1;
  for(let i=0;i<=4;i++){const y=padT+innerH*i/4;ctx.beginPath();ctx.moveTo(padL,y);ctx.lineTo(W-padR,y);ctx.stroke();}
  ctx.fillStyle=O_PALETTE.text;ctx.font='11px ui-monospace,Menlo,monospace';ctx.textAlign='right';ctx.textBaseline='middle';
  if(!buckets||buckets.length===0){
    ctx.fillStyle=O_PALETTE.text;ctx.textAlign='center';ctx.textBaseline='middle';
    ctx.fillText('waiting for traffic',W/2,H/2);
    return;
  }
  let maxV=1;for(const b of buckets){const v=Math.max(b.chars_saved_in||0,b.chars_saved_out||0);if(v>maxV)maxV=v;}
  // y-axis labels
  for(let i=0;i<=4;i++){const y=padT+innerH*i/4;const val=Math.round(maxV*(1-i/4));ctx.fillStyle=O_PALETTE.text;ctx.textAlign='right';ctx.fillText(fmt(val),padL-4,y);}
  const n=buckets.length;const xFor=i=>padL+(n<=1?innerW/2:innerW*i/(n-1));
  const yFor=v=>padT+innerH*(1-Math.min(1,v/maxV));
  function drawLine(key,color){
    ctx.strokeStyle=color;ctx.lineWidth=1.8;ctx.beginPath();
    for(let i=0;i<n;i++){const x=xFor(i);const y=yFor(buckets[i][key]||0);if(i===0)ctx.moveTo(x,y);else ctx.lineTo(x,y);}
    ctx.stroke();
    ctx.fillStyle=color;for(let i=0;i<n;i++){const x=xFor(i);const y=yFor(buckets[i][key]||0);ctx.beginPath();ctx.arc(x,y,1.5,0,Math.PI*2);ctx.fill();}
  }
  drawLine('chars_saved_in',O_PALETTE.input);
  drawLine('chars_saved_out',O_PALETTE.output);
  // Error overlay (red dots scaled to errors)
  ctx.fillStyle=O_PALETTE.err;
  let maxErr=0;for(const b of buckets)if((b.errors||0)>maxErr)maxErr=b.errors||0;
  if(maxErr>0){
    for(let i=0;i<n;i++){const e=buckets[i].errors||0;if(!e)continue;const x=xFor(i);const r=Math.min(5,2+(e/maxErr)*3);ctx.beginPath();ctx.arc(x,padT+innerH+padB*0.4,r,0,Math.PI*2);ctx.fill();}
  }
  // Time axis: first and last bucket labels.
  ctx.textAlign='center';ctx.textBaseline='top';ctx.fillStyle=O_PALETTE.text;
  ctx.fillText(oFmtTs(buckets[0].minute_ts),xFor(0),H-padB+2);
  ctx.fillText(oFmtTs(buckets[n-1].minute_ts),xFor(n-1),H-padB+2);
}

function oRenderEngineBars(buckets){
  const wrap=document.getElementById('o-eng-bars');if(!wrap)return;
  const totals={};
  for(const b of (buckets||[])){for(const [k,v] of Object.entries(b.engines||{})){totals[k]=(totals[k]||0)+v;}}
  const entries=Object.entries(totals).filter(([,v])=>v>0).sort((a,b)=>b[1]-a[1]);
  if(entries.length===0){wrap.innerHTML='<div class="nodata" style="font-family:var(--sans);color:var(--muted);font-style:italic;padding:8px 0">No engine activity yet.</div>';return;}
  const max=entries[0][1];
  const html=[];
  for(const [name,saved] of entries){
    const w=Math.max(2,Math.round((saved/max)*100));
    const safeName=String(name).replace(/[<>&"]/g,c=>({"<":"&lt;",">":"&gt;","&":"&amp;","\"":"&quot;"})[c]);
    html.push('<div class="eb" data-engine="'+safeName+'"><div class="ebn">'+safeName+'</div><div class="ebbar"><div class="ebbarfill" style="width:'+w+'%"></div></div><div class="ebt">'+fmt(saved)+'</div></div>');
  }
  wrap.innerHTML=html.join('');
}

function oRenderRecent(items){
  const body=document.getElementById('o-recent-body');if(!body)return;
  if(!items||items.length===0){body.innerHTML='<tr><td class="nodata" colspan="5">No requests yet.</td></tr>';return;}
  const newest=items.slice().reverse();
  const html=[];
  for(let i=0;i<newest.length;i++){
    const r=newest[i];
    let cls='st-na';if(r.status_code){if(r.status_code<300)cls='st-2xx';else if(r.status_code<500)cls='st-4xx';else cls='st-5xx';}
    const path=String(r.path||'').replace(/[<>&"]/g,c=>({"<":"&lt;",">":"&gt;","&":"&amp;","\"":"&quot;"})[c]);
    const reqId=r.request_id?String(r.request_id).slice(0,18):'-';
    const saved=fmt(r.chars_saved_in||0)+' / '+fmt(r.chars_saved_out||0);
    html.push('<tr class="rrow" data-idx="'+i+'"><td>'+path+'</td><td class="'+cls+'">'+(r.status_code||'-')+'</td><td>'+oFmtMs(r.ms)+'</td><td>'+saved+'</td><td>'+reqId+'</td></tr>');
    // Detail row, hidden by default.
    const engEntries=Object.entries(r.engines||{}).filter(([,v])=>v>0);
    const engHtml=engEntries.length?engEntries.map(([k,v])=>'<span class="det-eng"><b>'+k+'</b> '+fmt(v)+'</span>').join(''):'<span style="font-style:italic">no engine activity</span>';
    const extra='ts '+oFmtTs(r.ts)+' &middot; bytes '+fmt(r.bytes_in||0)+'/'+fmt(r.bytes_out||0);
    html.push('<tr class="detail" data-for="'+i+'" style="display:none"><td colspan="5"><div class="det-inner">'+engHtml+'<span style="margin-left:auto;color:var(--muted)">'+extra+'</span></div></td></tr>');
  }
  body.innerHTML=html.join('');
  body.querySelectorAll('tr.rrow').forEach(row=>{
    row.addEventListener('click',()=>{
      const det=body.querySelector('tr.detail[data-for="'+row.dataset.idx+'"]');
      if(!det)return;det.style.display=det.style.display==='none'?'':'none';
    });
  });
}

async function refreshObservability(){
  try{
    const [tsRes,recRes]=await Promise.all([
      fetch('/stats/timeseries').then(r=>r.json()).catch(()=>({buckets:[]})),
      fetch('/stats/recent?n=50').then(r=>r.json()).catch(()=>({items:[]})),
    ]);
    const buckets=(tsRes&&tsRes.buckets)||[];
    oDrawSavedChart(buckets);
    oRenderEngineBars(buckets);
    // Error rate over window.
    let req=0,err=0,lastP50=0,lastP95=0;
    for(const b of buckets){req+=b.requests||0;err+=b.errors||0;if(b.p50_ms)lastP50=b.p50_ms;if(b.p95_ms)lastP95=b.p95_ms;}
    const pct=req>0?(err/req)*100:0;
    setVal('o-err-pct',oFmtPct(pct));
    setVal('o-err-count',fmt(err));
    setVal('o-err-req',fmt(req));
    setVal('o-err-window',(tsRes&&tsRes.window_minutes?tsRes.window_minutes:60)+'m');
    const gauge=document.getElementById('o-err-gauge');
    if(gauge){const clamped=Math.min(100,pct*10);gauge.style.width=clamped+'%';}
    // p50/p95 — prefer global from /stats for stability across the whole uptime.
    try{const d=await fetch('/stats').then(r=>r.json());setVal('o-p50',oFmtMs(d.p50_ms));setVal('o-p95',oFmtMs(d.p95_ms));}
    catch(e){setVal('o-p50',oFmtMs(lastP50));setVal('o-p95',oFmtMs(lastP95));}
    oRenderRecent((recRes&&recRes.items)||[]);
  }catch(e){}
}

refreshObservability();
setInterval(refreshObservability,3000);
window.addEventListener('resize',()=>{const c=document.getElementById('o-chart-saved');if(c){c.width=0;refreshObservability();}});

// --- brain (Phase 2/3/4) ---
function fmtUsd(v){if(v==null||isNaN(v))return'-';if(v<0.0001)return '$0';if(v<0.01)return '$'+v.toFixed(4);if(v<1)return '$'+v.toFixed(3);return '$'+v.toFixed(2)}
function escHtml(s){return String(s).replace(/[<>&"]/g,c=>({"<":"&lt;",">":"&gt;","&":"&amp;","\"":"&quot;"})[c])}

function renderCostBars(byModel){
  const wrap=document.getElementById('b-cost-bars');if(!wrap)return;
  const entries=Object.entries(byModel||{}).filter(([,v])=>(v.usd||0)>0).sort((a,b)=>(b[1].usd||0)-(a[1].usd||0)).slice(0,6);
  if(entries.length===0){wrap.innerHTML='<div class="nodata" style="font-family:var(--sans);color:var(--muted);font-style:italic;padding:8px 0">No costed traffic yet.</div>';return;}
  const max=entries[0][1].usd||1;
  const html=entries.map(([model,row])=>{
    const w=Math.max(2,Math.round((row.usd/max)*100));
    return '<div class="cb"><div class="cbn" title="'+escHtml(model)+'">'+escHtml(model)+'</div><div class="cbbar"><div class="cbbarfill" style="width:'+w+'%"></div></div><div class="cbv">'+fmtUsd(row.usd)+'</div></div>';
  });
  wrap.innerHTML=html.join('');
}

async function refreshBrain(){
  try{
    const [cost,cache,providers]=await Promise.all([
      fetch('/cost').then(r=>r.json()).catch(()=>({total_usd:0,total_requests:0,unmatched_requests:0,by_model:{},budget:{}})),
      fetch('/cache/stats').then(r=>r.json()).catch(()=>({l1:{enabled:false},l2:{enabled:false}})),
      fetch('/providers').then(r=>r.json()).catch(()=>({adapters:[]})),
    ]);
    setVal('b-cost',fmtUsd(cost.total_usd||0));
    setVal('b-creq',fmt(cost.total_requests||0));
    setVal('b-cunm',fmt(cost.unmatched_requests||0));
    renderCostBars(cost.by_model||{});

    const l1El=document.getElementById('b-l1');
    if(l1El){const on=!!(cache.l1&&cache.l1.enabled);l1El.textContent='L1 '+(on?'on':'off');l1El.className='chip '+(on?'on':'off');}
    const l2El=document.getElementById('b-l2');
    if(l2El){const on=!!(cache.l2&&cache.l2.enabled);const mis=!!cache.l2_misconfigured;l2El.textContent='L2 '+(mis?'misconfig':(on?'on':'off'));l2El.className='chip '+(mis?'warn':(on?'on':'off'));}
    setVal('b-l2-lk',fmt((cache.l2&&cache.l2.lookups)||0));
    setVal('b-l2-hit',fmt((cache.l2&&cache.l2.hits)||0));
    const thr=cache.l2&&cache.l2.threshold;setVal('b-l2-th',thr!=null?thr.toFixed(2):'-');

    const pWrap=document.getElementById('b-providers');
    if(pWrap){
      const list=(providers.adapters||[]);
      pWrap.innerHTML=list.length?list.map(n=>'<span class="chip">'+escHtml(n)+'</span>').join(''):'<span class="chip off">none</span>';
    }
    const bEl=document.getElementById('b-budget');
    if(bEl){
      const b=cost.budget||{};
      const cap=b.char_limit||b.token_limit;
      if(!cap){bEl.textContent='no cap';bEl.className='chip off';}
      else{
        const used=b.chars_used||b.tokens_used||0;
        const pct=cap>0?Math.min(100,Math.round((used/cap)*100)):0;
        const lbl=b.char_limit?(used+' / '+cap+' chars'):(used+' / '+cap+' tokens');
        bEl.textContent=lbl+' ('+pct+'%)';
        bEl.className='chip '+(b.exceeded?'warn':'on');
      }
    }
  }catch(e){}
}
refreshBrain();
setInterval(refreshBrain,5000);

// --- keyboard shortcuts ---
document.addEventListener('keydown',ev=>{
  if(ev.target&&['INPUT','TEXTAREA'].includes(ev.target.tagName))return;
  const k=ev.key.toLowerCase();
  if(k==='r'){ev.preventDefault();refreshStats();refreshHealth();refreshSettings();refreshObservability();refreshBrain();}
  else if(k==='i'){ev.preventDefault();const t=document.getElementById('t-input_compression');if(t)t.click();}
  else if(k==='o'){ev.preventDefault();const t=document.getElementById('t-output_compression');if(t)t.click();}
  else if(k==='j'){ev.preventDefault();const t=document.getElementById('t-jl_dedupe');if(t)t.click();}
  else if(k==='c'){ev.preventDefault();const t=document.getElementById('t-caveman');if(t)t.click();}
});

</script>
</body>
</html>"""
