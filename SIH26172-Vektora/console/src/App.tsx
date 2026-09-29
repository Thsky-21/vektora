import { useState } from 'react'
import { BeforeAfter } from './pages/BeforeAfter'
import { Evidence } from './pages/Evidence'
import { Latency } from './pages/Latency'
import { Ledger } from './pages/Ledger'
import { Live } from './pages/Live'
import { downloadJson, useEvents, type Mode } from './useEvents'

const PAGES = ['Live', 'Latency', 'Ledger', 'Evidence', 'Before/After'] as const
type Page = (typeof PAGES)[number]

// VITE_MODE=replay builds the static Vercel version: starts in Replay and
// never talks to a backend.
const REPLAY_ONLY = import.meta.env.VITE_MODE === 'replay'
const PRODUCT = import.meta.env.VITE_PRODUCT_NAME ?? 'Vektora'

// Only claim "physical ESP32" when the recording actually contains device data.
function replaySource(events: { type: string; source?: string; fw?: string }[]) {
  if (events.some((e) => e.source === 'device' || (e.type === 'hello' && e.fw && e.fw !== 'test_sender')))
    return 'Recorded session from physical ESP32'
  if (events.some((e) => e.type === 'hello' && e.fw === 'test_sender'))
    return 'Simulated session (test_sender.py, no hardware)'
  return 'Recorded session (server events only, no device telemetry)'
}

export default function App() {
  const [mode, setMode] = useState<Mode>(REPLAY_ONLY ? 'replay' : 'live')
  const [page, setPage] = useState<Page>('Live')
  const { state, conn, replayInfo, loadReplay, speed, setSpeed, push } = useEvents(mode)
  const build = (state.tick?.firmware_hash as string | undefined) ?? state.events.find((e) => e.type === 'boot')?.firmware_hash ?? '—'
  const date = new Date().toISOString().slice(0, 10)

  return (
    <div className="app">
      <header className="hdr">
        <div className="hdr-title">SIH26172 · ISRO · {PRODUCT} <span className="hdr-sub num">· build {build} · session {date}</span></div>
        <nav className="tabs">
          {PAGES.map((p) => <button key={p} className={page === p ? 'on' : ''} onClick={() => setPage(p)}>{p}</button>)}
        </nav>
        <div className="hdr-right">
          {!REPLAY_ONLY && (
            <div className="seg">
              <button className={mode === 'live' ? 'on' : ''} onClick={() => setMode('live')}>Live</button>
              <button className={mode === 'replay' ? 'on' : ''} onClick={() => setMode('replay')}>Replay</button>
            </div>
          )}
          <span className={`conn ${conn}`}>{conn === 'open' ? 'server connected' : conn === 'replay' ? 'replay' : conn}</span>
        </div>
      </header>

      {mode === 'replay' && (
        <div className="banner">
          <span>
            {replayInfo
              ? <>{replaySource(state.events)} · <span className="num">{replayInfo.name}</span> · {replayInfo.played}/{replayInfo.n} events</>
              : 'Replay: load a recording (demo_session.json or a sessions/*.jsonl file)'}
          </span>
          <span className="row gap">
            <select value={speed} onChange={(e) => setSpeed(Number(e.target.value))} aria-label="replay speed">
              {[1, 2, 4, 10].map((x) => <option key={x} value={x}>{x}×</option>)}
            </select>
            <label className="file">Open file…
              <input type="file" accept=".json,.jsonl" onChange={(e) => e.target.files?.[0] && loadReplay(e.target.files[0], speed)} />
            </label>
          </span>
        </div>
      )}
      {mode === 'live' && (
        <div className="toolbar">
          <span className="muted num">{state.events.length} events this page load</span>
          <button onClick={() => downloadJson(state.events, `recording-${Date.now()}.json`)} disabled={!state.events.length}>
            Download recording (→ replay)
          </button>
        </div>
      )}

      <main>
        {page === 'Live' && <Live s={state} />}
        {page === 'Latency' && <Latency s={state} />}
        {page === 'Ledger' && <Ledger s={state} push={push} live={mode === 'live'} />}
        {page === 'Evidence' && <Evidence s={state} build={build} />}
        {page === 'Before/After' && <BeforeAfter s={state} />}
      </main>
    </div>
  )
}
