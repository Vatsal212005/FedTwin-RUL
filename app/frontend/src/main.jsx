import React, { useEffect, useMemo, useState } from "react"
import { createRoot } from "react-dom/client"
import "./styles.css"

const BRIDGE_URL = "http://127.0.0.1:8765"
const TAB_SESSION = Math.random().toString(36).slice(2, 10)

const SENSOR_LABELS = {
  sensor_2: "Fan inlet temp", sensor_3: "LPC outlet temp", sensor_4: "HPC outlet temp",
  sensor_7: "HPC outlet pressure", sensor_8: "Physical fan speed", sensor_9: "Core speed",
  sensor_11: "Static pressure", sensor_12: "Fuel-flow ratio", sensor_13: "Corrected fan speed",
  sensor_14: "Corrected core speed", sensor_15: "Bypass ratio", sensor_17: "Bleed enthalpy",
  sensor_20: "HPT coolant bleed", sensor_21: "LPT coolant bleed",
}
const SENSOR_SHORT = ["sensor_2", "sensor_3", "sensor_4", "sensor_7", "sensor_9", "sensor_15"]
const RESULT_COLORS = { fedtwin_rul: "#59e6bd", fedavg_moe: "#94a3b8", fedavg_transformer: "#6f88a8", fedprox_transformer: "#9a7bb5", trans_lighter_repro: "#e18768" }

function pathFor(values, width, height, min, max) {
  if (!values.length) return ""
  return values.map((value, index) => {
    const x = values.length === 1 ? width / 2 : (index / (values.length - 1)) * width
    const y = height - ((value - min) / Math.max(max - min, .0001)) * height
    return `${index ? "L" : "M"}${x.toFixed(1)},${y.toFixed(1)}`
  }).join(" ")
}

function Sparkline({ values, color = "#59e6bd" }) {
  const clean = values.length ? values : [0]
  const min = Math.min(...clean), max = Math.max(...clean)
  return <svg className="sparkline" viewBox="0 0 120 34" preserveAspectRatio="none"><path d={pathFor(clean, 120, 30, min, max)} fill="none" stroke={color} strokeWidth="2" vectorEffect="non-scaling-stroke"/></svg>
}

function RulChart({ frames, cursor, critical, healthy }) {
  const shown = frames.slice(0, cursor + 1)
  const predicted = shown.map(row => row.predictedRul)
  const actual = shown.map(row => row.trueRul)
  const width = 760, height = 225, max = 130
  const y = value => height - value / max * height
  return <div className="rul-chart">
    <svg viewBox={`0 0 ${width} ${height}`} preserveAspectRatio="none" aria-label="Predicted and reference RUL over replay cycles">
      {[0, 25, 50, 75, 100, 125].map(v => <g key={v}><line x1="0" x2={width} y1={y(v)} y2={y(v)} className="grid-line"/><text x="5" y={y(v)-5}>{v}</text></g>)}
      <rect x="0" y={y(critical)} width={width} height={height-y(critical)} className="critical-zone"/>
      <rect x="0" y={y(healthy)} width={width} height={y(critical)-y(healthy)} className="warning-zone"/>
      <path d={pathFor(actual, width, height, 0, max)} fill="none" stroke="#60727c" strokeWidth="2" strokeDasharray="6 6" vectorEffect="non-scaling-stroke"/>
      <path d={pathFor(predicted, width, height, 0, max)} fill="none" stroke="#59e6bd" strokeWidth="3" vectorEffect="non-scaling-stroke"/>
      {predicted.length > 0 && <circle cx={width} cy={y(predicted[predicted.length - 1])} r="5" fill="#59e6bd"/>}
    </svg>
    <div className="chart-legend"><span><i className="line live"/>FedTwin prediction</span><span><i className="line actual"/>NASA reference RUL</span><span className="chart-range">0–125 cycles</span></div>
  </div>
}

function ResultChart({ results }) {
  const width = 740, height = 245, min = 23.5, max = 28.5
  const y = v => height - ((v - min) / (max - min)) * height
  return <div className="result-chart"><svg viewBox={`0 0 ${width} ${height}`} preserveAspectRatio="none">
    {[24, 25, 26, 27, 28].map(v => <g key={v}><line x1="42" x2={width-10} y1={y(v)} y2={y(v)} className="grid-line"/><text x="5" y={y(v)+4}>{v}</text></g>)}
    {results.map(result => {
      const vals = result.scores.map(s => s.rmse)
      const xs = [70, width/2, width-38]
      const d = vals.map((v,i)=>`${i?"L":"M"}${xs[i]},${y(v)}`).join(" ")
      return <g key={result.key}><path d={d} fill="none" stroke={RESULT_COLORS[result.key]} strokeWidth={result.key === "fedtwin_rul" ? 4 : 2} opacity={result.key === "fedtwin_rul" ? 1 : .7}/>{vals.map((v,i)=><circle key={i} cx={xs[i]} cy={y(v)} r={result.key === "fedtwin_rul" ? 5 : 3} fill={RESULT_COLORS[result.key]}/>)}</g>
    })}
    {[[70,"α 1.0"],[width/2,"α 0.5"],[width-38,"α 0.1"]].map(([x,t])=><text className="x-label" key={t} x={x} y={height-4} textAnchor="middle">{t}</text>)}
  </svg></div>
}

function HealthLight({ status }) {
  return <div className="health-light" aria-label={`${status} health state`}>
    {['HEALTHY','WARNING','CRITICAL'].map(item => <div key={item} className={`lamp lamp--${item.toLowerCase()} ${status===item?'active':''}`}><i/><span>{item}</span></div>)}
  </div>
}

function Loading() { return <main className="loading"><div className="pulse-logo">⌁</div><h1>Loading FedTwin-RUL</h1><p>Preparing real FD004 replay data…</p></main> }

function App() {
  const [data, setData] = useState(null)
  const [engineId, setEngineId] = useState(100)
  const [cursor, setCursor] = useState(0)
  const [running, setRunning] = useState(false)
  const [speed, setSpeed] = useState(4)
  const [critical, setCritical] = useState(20)
  const [healthy, setHealthy] = useState(50)
  const [activeSection, setActiveSection] = useState("live")

  useEffect(() => { fetch("/demo-data.json").then(r => { if (!r.ok) throw new Error("Replay data unavailable"); return r.json() }).then(setData).catch(error => console.error(error)) }, [])
  const engine = useMemo(() => data?.engines.find(item => item.id === engineId) ?? data?.engines[0], [data, engineId])
  const frame = engine?.timeline[cursor]
  const status = frame ? frame.predictedRul <= critical ? "CRITICAL" : frame.predictedRul <= healthy ? "WARNING" : "HEALTHY" : "HEALTHY"

  // ESP32 bridge (hardware/rul_bridge.py). Optional: the dashboard works without it.
  const [hw, setHw] = useState({ bridge: false, ack: null })
  const [hwTest, setHwTest] = useState(null)   // labelled TEST value, overrides the model while set
  useEffect(() => {
    if (!frame) return
    const state = hwTest === null
      ? { engineId: engine.id, cycle: frame.cycle, predictedRul: frame.predictedRul, critical, healthy, mode: "dashboard" }
      : { engineId: 0, cycle: 0, predictedRul: hwTest, critical, healthy, mode: "test" }
    // Only one tab drives the board: the tab you are using (focused) takes control.
    const post = () => fetch(`${BRIDGE_URL}/state`, { method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ ...state, session: TAB_SESSION, claim: document.hasFocus() }) })
      .then(r => r.json()).then(j => setHw({ bridge: true, ack: j.accepted ? j.last_ack ?? null : null, other: j.accepted === false }))
      .catch(() => setHw({ bridge: false, ack: null }))
    post()
    const keepAlive = setInterval(post, 1500)   // heartbeat while this frame is shown
    return () => clearInterval(keepAlive)
  }, [engine?.id, frame?.cycle, frame?.predictedRul, critical, healthy, hwTest])

  useEffect(() => { setCursor(0); setRunning(false) }, [engineId])
  useEffect(() => {
    if (!running || !engine) return
    const timer = setInterval(() => setCursor(current => {
      if (current >= engine.timeline.length - 1) { setRunning(false); return current }
      return current + 1
    }), 1000 / speed)
    return () => clearInterval(timer)
  }, [running, speed, engine])

  if (!data || !frame) return <Loading/>
  const recent = engine.timeline.slice(Math.max(0, cursor - 39), cursor + 1)
  const sensorIndex = name => data.dataset.sensorNames.indexOf(name)
  const hwRul = hwTest ?? frame.predictedRul
  const hwStatus = hwRul <= critical ? "CRITICAL" : hwRul <= healthy ? "WARNING" : "HEALTHY"
  const command = (hwTest !== null ? "TEST · " : "") + (hwStatus === "CRITICAL" ? "RED LED + FAST BUZZER" : hwStatus === "WARNING" ? "YELLOW LED + CHIRP" : "BLUE LED")
  const testValues = [["HEALTHY", Math.min(125, healthy + 25)], ["WARNING", (critical + healthy) / 2], ["CRITICAL", Math.max(0, critical - 5)]]
  const ackState = hw.ack?.match(/state=(\w+)/)?.[1]

  return <div className="app">
    <header className="topbar">
      <a className="brand" href="#live"><span className="brand-mark">⌁</span><span><strong>FedTwin<span>RUL</span></strong><small>Digital-twin fleet console</small></span></a>
      <nav>{[["live","Live replay"],["federation","Federation"],["results","Results"],["about","How it works"]].map(([id,label])=><a key={id} className={activeSection===id?"active":""} href={`#${id}`} onClick={()=>setActiveSection(id)}>{label}</a>)}</nav>
      <div className="model-chip"><i/> real checkpoint · round {data.model.bestRound}</div>
    </header>

    <main>
      <section className="hero" id="live">
        <div><div className="eyebrow">NASA C-MAPSS FD004 · Live historical replay</div><h1>See an engine’s digital twin <em>think.</em></h1><p>Replay real turbofan measurements cycle by cycle. Every RUL value below is produced by the trained FedTwin-RUL checkpoint using a 40-cycle, 14-sensor window.</p></div>
        <div className="hero-badges"><span><b>10</b> federated clients</span><span><b>60,159</b> parameters</span><span><b>24.371</b> mean RMSE</span></div>
      </section>

      <section className="control-bar">
        <label>Test engine<select value={engineId} onChange={e=>setEngineId(Number(e.target.value))}>{data.engines.map(item=><option key={item.id} value={item.id}>Engine {String(item.id).padStart(3,'0')} · {item.cycles} observed cycles</option>)}</select></label>
        <div className="playback"><button className="play" onClick={()=>setRunning(!running)}>{running?"Pause":"Start replay"}</button><button onClick={()=>{setCursor(0);setRunning(false)}}>Reset</button></div>
        <label>Replay speed<select value={speed} onChange={e=>setSpeed(Number(e.target.value))}><option value="1">1×</option><option value="2">2×</option><option value="4">4×</option><option value="8">8×</option></select></label>
        <div className="cycle-step" aria-label="Cycle control">{[[-10,"−10"],[-1,"−1"]].map(([d,l])=><button key={l} onClick={()=>{setRunning(false);setCursor(c=>Math.max(0,c+d))}}>{l}</button>)}<input aria-label="Go to cycle" type="number" min={engine.timeline[0].cycle} max={engine.timeline[engine.timeline.length-1].cycle} value={frame.cycle} onChange={e=>{const want=Number(e.target.value);const i=engine.timeline.findIndex(f=>f.cycle>=want);setRunning(false);setCursor(i<0?engine.timeline.length-1:i)}}/>{[[1,"+1"],[10,"+10"]].map(([d,l])=><button key={l} onClick={()=>{setRunning(false);setCursor(c=>Math.min(engine.timeline.length-1,c+d))}}>{l}</button>)}<span className="cycle-rul">RUL {frame.predictedRul.toFixed(1)} · {status}</span></div>
        <div className="timeline-control"><span>Cycle {frame.cycle} / {engine.cycles}</span><input aria-label="Replay cycle" type="range" min="0" max={engine.timeline.length-1} value={cursor} onChange={e=>{setCursor(Number(e.target.value));setRunning(false)}}/></div>
      </section>

      <section className="summary-grid">
        <article className={`rul-card state-${status.toLowerCase()}`}><div className="card-label">Predicted remaining life</div><div className="rul-number">{frame.predictedRul.toFixed(1)}<small>cycles</small></div><div className="rul-delta"><span>NASA reference</span><strong>{frame.trueRul.toFixed(0)} cycles</strong></div></article>
        <article className="status-card"><div><span className="card-label">Machine health</span><h2>{status}</h2><p>{status==="HEALTHY"?"Operating comfortably above the warning threshold.":status==="WARNING"?"Maintenance window is approaching.":"Immediate maintenance response recommended."}</p></div><HealthLight status={status}/></article>
        <article className="command-card"><span className="card-label">ESP32 indicator command</span><code>{command}</code><p>{hwTest !== null ? `Hardware TEST value ${hwTest.toFixed(1)} — not a model prediction.` : "Model prediction sent over USB serial by hardware/rul_bridge.py → ESP32 (blue D21, yellow D22, red D23, buzzer D18)."}</p><div className="hw-test">{testValues.map(([label, v]) => <button key={label} className={hwTest === v ? "active" : ""} onClick={() => setHwTest(v)}>Test {label}</button>)}<button className={hwTest === null ? "active" : ""} onClick={() => setHwTest(null)}>Follow model</button></div><div className="device-row"><span><i className={hw.bridge?"on":"off"}/>{hw.bridge?"Bridge running":"Bridge not running"}</span><span><i className={ackState?"on":"off"}/>{hw.other?"Another tab controls the board — click here to take over":ackState?`ESP32 confirmed ${ackState}`:"No ESP32 confirmation"}</span></div></article>
      </section>

      <section className="workspace-grid">
        <article className="panel chart-panel"><div className="panel-head"><div><span className="eyebrow">Prediction timeline</span><h2>RUL against engine cycles</h2></div><span className="live-badge"><i/> replay frame {cursor+1}</span></div><RulChart frames={engine.timeline} cursor={cursor} critical={critical} healthy={healthy}/></article>
        <article className="panel twin-panel"><div className="panel-head"><div><span className="eyebrow">Digital twin state</span><h2>Context beyond sensor history</h2></div><span className="regime">Regime {frame.regime} / 6</span></div><div className="twin-orbit"><div className="orbit-ring"><span>z</span></div><div className="twin-center"><small>health proxy q</small><strong>{frame.q.toFixed(3)}</strong><em>1 = closest to healthy reference</em></div></div><div className="twin-metrics"><div><span>Window drift d</span><strong>{frame.drift.toFixed(3)}</strong></div><div><span>Twin-state norm</span><strong>{frame.twinNorm.toFixed(3)}</strong></div></div><div className="ops"><span>Operating settings</span>{frame.ops.map((op,i)=><div key={i}><small>o{i+1}</small><b>{op.toFixed(2)}</b></div>)}</div></article>
      </section>

      <section className="panel sensors-panel"><div className="panel-head"><div><span className="eyebrow">Selected telemetry</span><h2>Six signals from the 14-sensor model input</h2></div><p>Standardized with training-only statistics · trailing 40-cycle window</p></div><div className="sensor-grid">{SENSOR_SHORT.map((name,index)=>{const values=recent.map(row=>row.sensors[sensorIndex(name)]);const latest=values[values.length-1]??0;return <article key={name}><div><span>S{data.dataset.sensorNames.indexOf(name)+1}</span><small>{name.replace('sensor_','sensor ')}</small></div><h3>{SENSOR_LABELS[name]}</h3><Sparkline values={values} color={["#59e6bd","#6bb6ff","#e6b459","#a98bf1","#ee7d68","#73d4db"][index]}/><footer><strong>{latest.toFixed(2)} σ</strong><span>{latest>=0?"above":"below"} training mean</span></footer></article>})}</div></section>

      <section className="section" id="federation"><div className="section-heading"><div><span className="eyebrow">Federated learning</span><h2>Ten sites train one shared model.</h2></div><p>Each client keeps its engine windows local. Only model parameters are aggregated with FedAvg, protecting raw operating histories while learning across heterogeneous regimes.</p></div><div className="federation-layout"><article className="panel federation-map"><div className="server-node"><span>GLOBAL</span><strong>FedTwin-RUL</strong><small>round {data.model.bestRound} selected</small></div><div className="client-ring">{data.clients.map(client=><div className="client-node" key={client.id}><span>C{client.id}</span><b>{client.engines} engines</b><small>{client.windows} windows · R{client.dominantRegime}</small><i/></div>)}</div></article><article className="panel federation-copy"><div className="flow-step"><b>01</b><div><strong>Local learning</strong><p>Transformer reads temporal sensor patterns; the twin encoder turns operating context, q and drift into a compact state.</p></div></div><div className="flow-step"><b>02</b><div><strong>Feature fusion</strong><p>The 64-D temporal representation and 16-D twin state are concatenated before RUL regression.</p></div></div><div className="flow-step"><b>03</b><div><strong>FedAvg aggregation</strong><p>All 10 clients contribute model updates. Raw C-MAPSS windows never leave their client partition.</p></div></div><div className="flow-step"><b>04</b><div><strong>Global prediction</strong><p>The shared model estimates RUL on previously unseen FD004 test engines.</p></div></div></article></div></section>

      <section className="section" id="results"><div className="section-heading"><div><span className="eyebrow">Final three-seed study</span><h2>Accuracy holds as clients diverge.</h2></div><p>Test RMSE across Dirichlet heterogeneity settings. Lower is better; α = 0.1 is the most heterogeneous split.</p></div><div className="results-layout"><article className="panel"><ResultChart results={data.results}/><div className="result-legend">{data.results.map(r=><span key={r.key}><i style={{background:RESULT_COLORS[r.key]}}/>{r.name}</span>)}</div></article><article className="panel rank-panel"><span className="eyebrow">Across-setting mean</span>{[...data.results].sort((a,b)=>a.meanRmse-b.meanRmse).map((r,i)=><div className={i===0?"rank first":"rank"} key={r.key}><span>0{i+1}</span><div><strong>{r.name}</strong><small>{r.parameters.toLocaleString()} parameters</small></div><b>{r.meanRmse.toFixed(3)}</b></div>)}</article></div></section>

      <section className="section" id="about"><div className="section-heading"><div><span className="eyebrow">System explanation</span><h2>What the project is actually doing.</h2></div><p>A practical predictive-maintenance stack: physical history becomes an evolving digital state, then a federated model estimates how long the asset can keep operating.</p></div><div className="architecture"><div><span>01</span><b>40-cycle window</b><small>14 selected FD004 sensor channels</small></div><i>→</i><div><span>02</span><b>Temporal Transformer</b><small>learns degradation patterns over time</small></div><i>+</i><div><span>03</span><b>Digital Twin</b><small>o₁, o₂, o₃, health proxy q, drift d</small></div><i>→</i><div><span>04</span><b>Fusion + RUL head</b><small>predicts normalized remaining life</small></div></div>
        <div className="explain-grid"><article><span>RUL</span><h3>Remaining Useful Life</h3><p>The estimated number of operating cycles before failure. It gives maintenance teams time to act before an unscheduled shutdown.</p></article><article><span>DT</span><h3>Digital Twin state</h3><p>A compact, data-driven representation of current operating conditions and deviation from a healthy training reference.</p></article><article><span>FL</span><h3>Federated Learning</h3><p>Ten simulated industrial sites train collaboratively without pooling their raw engine sensor histories.</p></article></div>
      </section>

      <section className="threshold-panel"><div><span className="eyebrow">Arduino-ready state mapping</span><h2>Configure the future LED / relay thresholds.</h2><p>These controls change the live dashboard state immediately. Later, the same command can be returned by an API for ESP8266 polling.</p></div><div className="threshold-controls"><label>Critical at or below<input type="number" min="1" max={healthy-1} value={critical} onChange={e=>setCritical(Math.min(Number(e.target.value),healthy-1))}/><span>cycles</span></label><label>Healthy above<input type="number" min={critical+1} max="125" value={healthy} onChange={e=>setHealthy(Math.max(Number(e.target.value),critical+1))}/><span>cycles</span></label><div className="mapping"><span className="green">RUL &gt; {healthy}: GREEN</span><span className="amber">{critical+1}–{healthy}: AMBER</span><span className="red">RUL ≤ {critical}: RED</span></div></div></section>
    </main>

    <footer className="site-footer"><div><strong>FedTwin-RUL</strong><span>Paper ID 1041 · UPWIECON 2026 · Submitted</span></div><p>Replay values are generated from the repository’s real FD004 data and trained checkpoint. Hardware indicators are a software preview until physical integration is completed.</p></footer>
  </div>
}

createRoot(document.getElementById("root")).render(<React.StrictMode><App/></React.StrictMode>)
