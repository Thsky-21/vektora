import { fmt } from '../store'

// Horizontal meter with the ISRO limit drawn as a line. The bar is scaled to
// 1.5x the limit so both PASS and FAIL values stay visible.
export function Meter({ label, value, limit, unit, digits = 1, sub }: {
  label: string; value?: number; limit: number; unit: string; digits?: number; sub?: string
}) {
  const scale = limit * 1.5
  const has = typeof value === 'number' && !Number.isNaN(value)
  const pass = has && value! < limit
  const w = has ? Math.min(100, (value! / scale) * 100) : 0
  return (
    <div className="meter">
      <div className="meter-head">
        <span className="label">{label}</span>
        <span className={`badge ${has ? (pass ? 'pass' : 'fail') : 'none'}`}>{has ? (pass ? 'PASS' : 'FAIL') : '—'}</span>
      </div>
      <div className="meter-value num">
        {has ? fmt(value, digits) : '—'}<span className="unit"> {unit}</span>
        <span className="limit"> / limit &lt; {limit} {unit}</span>
      </div>
      <div className="meter-track">
        <div className={`meter-fill ${has ? (pass ? 'pass' : 'fail') : ''}`} style={{ width: `${w}%` }} />
        <div className="meter-limit" style={{ left: `${(limit / scale) * 100}%` }} />
      </div>
      {sub && <div className="meter-sub num">{sub}</div>}
    </div>
  )
}

export function Funnel({ f }: { f?: Record<string, number> }) {
  const rows: [string, string][] = [['frames', 'frames heard'], ['s0', 'passed energy gate'], ['s1', 'passed voice gate'],
    ['infer', 'DS-CNN inferences'], ['det', 'detections']]
  const top = f?.frames || 0
  return (
    <div className="funnel">
      {rows.map(([k, name]) => {
        const v = f?.[k]
        const pct = top && typeof v === 'number' ? (v / top) * 100 : undefined
        return (
          <div className="funnel-row" key={k}>
            <span className="funnel-name">{name}</span>
            <div className="funnel-track">
              <div className={`funnel-fill ${k === 'det' ? 'accent' : ''}`} style={{ width: `${pct ?? 0}%` }} />
            </div>
            <span className="num funnel-val">{typeof v === 'number' ? v.toLocaleString() : '—'}</span>
            <span className="num funnel-pct">{pct === undefined ? '—' : `${pct < 10 ? pct.toFixed(2) : pct.toFixed(1)}%`}</span>
          </div>
        )
      })}
    </div>
  )
}

export function ScoreTrace({ ticks, threshold }: { ticks: { t: number; score: number }[]; threshold?: number }) {
  const W = 600, H = 120, pad = 4
  if (!ticks.length) return <div className="empty">no device telemetry yet — score trace shows here</div>
  const t1 = ticks[ticks.length - 1].t
  // last 2 min, but fit to the data until 2 min have accumulated (min 10 s)
  const span = Math.max(10e6, Math.min(120e6, t1 - ticks[0].t))
  const x = (t: number) => pad + ((t - (t1 - span)) / span) * (W - 2 * pad)
  const y = (s: number) => H - pad - Math.max(0, Math.min(1, s)) * (H - 2 * pad)
  const pts = ticks.filter((p) => p.t >= t1 - span).map((p) => `${x(p.t).toFixed(1)},${y(p.score).toFixed(1)}`).join(' ')
  return (
    <svg className="trace" viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none" role="img" aria-label="keyword score, last 2 minutes">
      <line x1={0} x2={W} y1={y(0)} y2={y(0)} className="axis" />
      {threshold !== undefined && <line x1={0} x2={W} y1={y(threshold)} y2={y(threshold)} className="thr" />}
      <polyline points={pts} className="score" />
    </svg>
  )
}

const STATES = ['LISTEN', 'SPEECH', 'DETECTED', 'LINKING', 'STREAMING', 'CLOSING']
export function StateStrip({ state }: { state?: string }) {
  return (
    <div className="strip">
      {STATES.map((s) => (
        <span key={s} className={`strip-cell ${state === s ? (s === 'DETECTED' ? 'on accent' : 'on') : ''}`}>{s}</span>
      ))}
      {state && !STATES.includes(state) && <span className="strip-cell on fail">{state}</span>}
    </div>
  )
}
