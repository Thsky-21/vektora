import { useEffect, useState } from 'react'
import { fmt, type Ev, type State } from '../store'

// Ledger (CLAUDE.md §10.3): the accuracy evidence, scored from the event
// stream itself. The operator drops a marker ("I just said Vektora at 1 m",
// "I just said vector"); a device `detect` within WINDOW_MS after it is a hit.
// Markers are events too, so "Download recording" keeps them next to the
// detects, and tools/analyze.py scores the same file the same way.

const WINDOW_MS = 3000
const DISTANCES = ['0.3 m', '1 m', '2 m']
const CONFUSABLES = ['vector', 'victor', 'Vectra', 'spectra', 'sector', 'doctor', 'actor', 'factor', 'detector',
  'projector', 'very', 'video', 'voice', 'camera', 'tomorrow', 'vek', 'tora', 'model', 'training']

const detectTimes = (s: State) => s.detects.map((d) => d.rx_ms).filter((t): t is number => typeof t === 'number')
const hit = (t0: number, dets: number[]) => dets.some((d) => d >= t0 && d <= t0 + WINDOW_MS)

export function Ledger({ s, push, live }: { s: State; push: (e: Ev) => void; live: boolean }) {
  const [dist, setDist] = useState(DISTANCES[1])
  const [speaker, setSpeaker] = useState('S1')
  const [desc, setDesc] = useState('multilingual talk radio (Hindi/Kannada/English) + fan, speaker at 1 m')
  const [, tick] = useState(0)
  useEffect(() => { const id = window.setInterval(() => tick((x) => x + 1), 1000); return () => clearInterval(id) }, [])

  const dets = detectTimes(s)
  const trials = s.events.filter((e) => e.type === 'trial')
  const soakEv = s.events.filter((e) => e.type === 'soak')
  const soakStart = [...soakEv].reverse().find((e) => e.action === 'start')
  const soakStop = soakStart && soakEv.find((e) => e.action === 'stop' && e.rx_ms >= soakStart.rx_ms)
  const soakEnd = soakStop?.rx_ms ?? (soakStart ? Date.now() : undefined)
  const soakH = soakStart && soakEnd ? (soakEnd - soakStart.rx_ms) / 3.6e6 : undefined
  const soakFA = soakStart ? dets.filter((d) => d >= soakStart.rx_ms && d <= (soakEnd ?? Infinity)).length : undefined
  const thr = s.tick?.threshold as number | undefined

  const pos = trials.filter((t) => t.kind === 'positive')
  const byDist = DISTANCES.map((d) => {
    const ts = pos.filter((t) => t.label === d)
    const h = ts.filter((t) => hit(t.rx_ms, dets)).length
    return { d, n: ts.length, h, speakers: new Set(ts.map((t) => t.speaker)).size }
  })
  const conf = new Map<string, { n: number; fa: number }>()
  for (const t of trials.filter((x) => x.kind === 'confusable')) {
    const c = conf.get(t.label) ?? { n: 0, fa: 0 }
    c.n += 1
    c.fa += hit(t.rx_ms, dets) ? 1 : 0
    conf.set(t.label, c)
  }
  const hms = (h?: number) => {
    if (h === undefined) return '—'
    const sec = Math.floor(h * 3600)
    return `${String(Math.floor(sec / 3600)).padStart(2, '0')}:${String(Math.floor(sec / 60) % 60).padStart(2, '0')}:${String(sec % 60).padStart(2, '0')}`
  }

  return (
    <div className="grid">
      <section className="card span2">
        <div className="row between">
          <h2>M4 · False-accept soak</h2>
          {live && (
            <div className="row gap">
              <button disabled={!!soakStart && !soakStop} onClick={() => push({ type: 'soak', action: 'start', desc, threshold: thr })}>Start soak</button>
              <button disabled={!soakStart || !!soakStop} onClick={() => push({ type: 'soak', action: 'stop' })}>Stop</button>
            </div>
          )}
        </div>
        <div className="kv">
          Negative audio:{' '}
          {live && !soakStart ? <input className="wide" value={desc} onChange={(e) => setDesc(e.target.value)} /> : <span>{soakStart?.desc ?? '—'}</span>}
        </div>
        <table className="tbl num">
          <thead><tr><th>elapsed</th><th>detections (all false)</th><th>FA / hour</th><th>threshold</th><th>status</th></tr></thead>
          <tbody><tr>
            <td>{hms(soakH)}</td><td>{soakFA ?? '—'}</td>
            <td>{soakH && soakH > 0 && soakFA !== undefined ? fmt(soakFA / soakH, 2) : '—'}</td>
            <td>{fmt(soakStart?.threshold ?? thr, 2)}</td>
            <td className="text">{!soakStart ? 'not started' : soakStop ? 'done' : 'running'}{soakH !== undefined && soakH < 2 ? ' (M4 needs ≥ 2 h)' : ''}</td>
          </tr></tbody>
        </table>
        <div className="kv muted">Every detect during the soak counts as a false accept, so play no keyword. For long runs, keep tools/serial_logger.py logging as a backup and score its file with tools/analyze.py --negative.</div>
      </section>

      <section className="card span2">
        <div className="row between">
          <h2>M5 · True-positive rate by distance</h2>
          {live && (
            <div className="row gap">
              <select value={dist} onChange={(e) => setDist(e.target.value)} aria-label="distance">
                {DISTANCES.map((d) => <option key={d}>{d}</option>)}
              </select>
              <input className="narrow" value={speaker} onChange={(e) => setSpeaker(e.target.value)} aria-label="speaker id" />
              <button className="accent" onClick={() => push({ type: 'trial', kind: 'positive', label: dist, speaker })}>Said “Vektora”</button>
            </div>
          )}
        </div>
        <table className="tbl num">
          <thead><tr><th>distance</th><th>utterances</th><th>speakers</th><th>detected</th><th>TPR</th></tr></thead>
          <tbody>
            {byDist.map((r) => (
              <tr key={r.d}><td className="text">{r.d}</td><td>{r.n}</td><td>{r.speakers || '—'}</td><td>{r.n ? r.h : '—'}</td>
                <td>{r.n ? `${fmt((100 * r.h) / r.n, 0)} %` : '—'}</td></tr>
            ))}
          </tbody>
        </table>
        <div className="kv muted">Click right after saying the keyword. A detect within {WINDOW_MS / 1000} s counts as a hit. Protocol: ≥ 20 utterances × ≥ 3 speakers per distance.</div>
      </section>

      <section className="card span2">
        <h2>Confusable words <span className="muted">(click after saying one; red = the device fired)</span></h2>
        <div className="chips">
          {CONFUSABLES.map((w) => {
            const c = conf.get(w)
            return (
              <button key={w} disabled={!live} className={`chip ${c ? (c.fa ? 'fail' : 'pass') : ''}`}
                onClick={() => push({ type: 'trial', kind: 'confusable', label: w })}>
                {w}{c ? <span className="num"> {c.fa}/{c.n}</span> : null}
              </button>
            )
          })}
        </div>
        <div className="kv muted num">
          Total: {[...conf.values()].reduce((a, c) => a + c.n, 0)} said, {[...conf.values()].reduce((a, c) => a + c.fa, 0)} false accepts
        </div>
      </section>
    </div>
  )
}
