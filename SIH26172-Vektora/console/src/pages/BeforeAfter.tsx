import { useState } from 'react'
import { fmt, median, p90, type Ev, type State } from '../store'

// Before/After (CLAUDE.md §10.5): the gate's effect, measured. Device ticks are
// grouped by (gate, config_hash): an M1 run (GATE_ENABLED 0) and an M2/M3 run
// (gate on) land in different groups. Ticks come from this page's stream plus
// any recordings loaded below (e.g. sessions/m1.jsonl from serial_logger.py).

type Group = { key: string; gate: number | string; cfg: string; fw: string; src: string; ticks: Ev[] }

function parse(text: string): Ev[] {
  try {
    const j = JSON.parse(text)
    return Array.isArray(j) ? j : j.events ?? []
  } catch {
    return text.split('\n').filter((l) => l.trim().startsWith('{')).flatMap((l) => { try { return [JSON.parse(l)] } catch { return [] } })
  }
}

function groups(ticks: { e: Ev; src: string }[]): Group[] {
  const m = new Map<string, Group>()
  for (const { e, src } of ticks) {
    const key = `${e.gate}|${e.config_hash}|${src}`
    const g: Group = m.get(key) ?? { key, gate: e.gate ?? '—', cfg: e.config_hash ?? '—', fw: e.firmware_hash ?? '—', src, ticks: [] }
    g.ticks.push(e)
    m.set(key, g)
  }
  return [...m.values()].sort((a, b) => String(a.gate).localeCompare(String(b.gate)))
}

const nums = (xs: unknown[]) => xs.filter((v): v is number => typeof v === 'number' && !Number.isNaN(v))

export function BeforeAfter({ s }: { s: State }) {
  const [loaded, setLoaded] = useState<{ name: string; ticks: Ev[] }[]>([])
  const own = s.events.filter((e) => e.type === 'tick').map((e) => ({ e, src: 'this session' }))
  const ext = loaded.flatMap((f) => f.ticks.map((e) => ({ e, src: f.name })))
  const G = groups([...own, ...ext])

  const rows = G.map((g) => {
    const idle = g.ticks.filter((t) => t.state === 'LISTEN')
    const f0 = g.ticks.find((t) => t.funnel)?.funnel, f1 = [...g.ticks].reverse().find((t) => t.funnel)?.funnel
    const frames = f0 && f1 ? f1.frames - f0.frames : 0
    return {
      g, n: idle.length,
      cpu: nums(idle.map((t) => t.cpu_avg)), c0: nums(idle.map((t) => t.cpu0)), c1: nums(idle.map((t) => t.cpu1)),
      duty: nums(g.ticks.map((t) => t.dscnn_duty)),
      s0: frames > 0 ? (100 * (f1.s0 - f0.s0)) / frames : undefined,
      infer: f0 && f1 ? f1.infer - f0.infer : undefined,
    }
  })
  const max = Math.max(10, ...rows.map((r) => median(r.cpu) ?? 0)) * 1.2

  const load = async (fs: FileList | null) => {
    if (!fs) return
    const add = await Promise.all([...fs].map(async (f) => ({ name: f.name, ticks: parse(await f.text()).filter((e) => e.type === 'tick') })))
    setLoaded((x) => [...x, ...add])
  }

  return (
    <div className="grid">
      <section className="card span2">
        <div className="row between">
          <h2>Idle CPU: gate off vs on <span className="muted">(LISTEN ticks, median)</span></h2>
          <label className="file">Add recording…
            <input type="file" multiple accept=".json,.jsonl" onChange={(e) => load(e.target.files)} />
          </label>
        </div>
        {rows.length ? rows.map((r) => {
          const m = median(r.cpu)
          return (
            <div className="wf-row" key={r.g.key}>
              <span className="num wf-id">gate {String(r.g.gate)} <span className="muted">{r.g.src}</span></span>
              <div className="wf-track">
                {m !== undefined && <div className={`wf-seg ${m < 10 ? 'pass' : 'fail'}`} style={{ width: `${(m / max) * 100}%` }} title={`${m.toFixed(2)} %`} />}
                <div className="meter-limit" style={{ left: `${(10 / max) * 100}%` }} />
              </div>
              <span className="num wf-total">{fmt(m, 2, ' %')}</span>
            </div>
          )
        }) : <div className="empty">no device ticks yet: run M1 (GATE_ENABLED 0) and M2 (gate on), then load both logs here</div>}
      </section>

      <section className="card span2">
        <h2>Detail</h2>
        <table className="tbl num">
          <thead><tr><th>gate</th><th>source</th><th>fw / cfg</th><th>LISTEN ticks</th><th>CPU avg p50 / p90</th><th>core0</th><th>core1</th><th>DS-CNN duty</th><th>frames past S0</th><th>inferences</th></tr></thead>
          <tbody>
            {rows.map((r) => (
              <tr key={r.g.key}>
                <td>{String(r.g.gate)}</td><td className="text">{r.g.src}</td><td className="text">{r.g.fw} / {r.g.cfg}</td><td>{r.n}</td>
                <td>{fmt(median(r.cpu), 2)} / {fmt(p90(r.cpu), 2)} %</td><td>{fmt(median(r.c0), 2)} %</td><td>{fmt(median(r.c1), 2)} %</td>
                <td>{r.duty.length ? fmt(r.duty.reduce((a, b) => a + b, 0) / r.duty.length, 2) : '—'} %</td>
                <td>{fmt(r.s0, 2)} %</td><td>{r.infer ?? '—'}</td>
              </tr>
            ))}
          </tbody>
        </table>
        <div className="kv muted">Same firmware, one flag: GATE_ENABLED 0 runs features + the model on every frame (M1); 1 runs them only while the gate is open (M2 quiet, M3 ambient speech).</div>
      </section>
    </div>
  )
}
