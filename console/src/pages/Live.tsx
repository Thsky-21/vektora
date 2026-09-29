import { Funnel, Meter, ScoreTrace, StateStrip } from '../components/Parts'
import { fmt, type State } from '../store'

const KB = 1024

export function Live({ s }: { s: State }) {
  const t = s.tick
  const radio = t?.radio ?? (t?.state ? (['LINKING', 'STREAMING'].includes(t.state) ? 'ON' : 'OFF') : undefined)
  const cur = s.order.length ? s.sessions[s.order[s.order.length - 1]] : undefined
  const lastDetect = s.detects[s.detects.length - 1]
  const recent = [...s.order].reverse().slice(0, 6).map((id) => s.sessions[id])

  return (
    <div className="grid live">
      <section className="card span2">
        <div className="row between">
          <h2>Device state</h2>
          <span className={`radio ${radio === 'ON' ? 'on' : radio === 'OFF' ? 'off' : ''}`}>RF: {radio ?? '—'}</span>
        </div>
        <StateStrip state={t?.state} />
      </section>

      <section className="card">
        <h2>CPU (radio-dark idle)</h2>
        <Meter label="Average, both cores" value={t?.cpu_avg} limit={10} unit="%"
          sub={`core0 ${fmt(t?.cpu0)}%   core1 ${fmt(t?.cpu1)}%`} />
      </section>

      <section className="card">
        <h2>Internal RAM</h2>
        <Meter label="Used now" value={t?.ram_used !== undefined ? t.ram_used / KB : undefined} limit={256} unit="KB"
          sub={`peak ${fmt(t?.ram_peak !== undefined ? t.ram_peak / KB : undefined)} KB   arena ${fmt(t?.arena_used !== undefined ? t.arena_used / KB : undefined)} KB`} />
      </section>

      <section className="card span2">
        <h2>Gate cascade <span className="muted">(since boot)</span></h2>
        <Funnel f={t?.funnel} />
        <div className="kv num">DS-CNN duty {fmt(t?.dscnn_duty, 2)}%   ring overflow {t?.ring_ovf ?? '—'}   mic rms {t?.mic_rms ?? '—'}   noise floor {t?.noise_floor ?? '—'}</div>
      </section>

      <section className="card span2">
        <div className="row between">
          <h2>Keyword score <span className="muted">(up to last 2 min)</span></h2>
          <span className="num muted">now {fmt(t?.score, 2)}   thr {fmt(t?.threshold ?? t?.thr, 2)}</span>
        </div>
        <ScoreTrace ticks={s.ticks} threshold={t?.threshold ?? t?.thr} />
        {lastDetect && (
          <div className="kv num accent-text">
            last wake: score {fmt(lastDetect.score, 2)}   infer {fmt(lastDetect.infer_us / 1000, 1, ' ms')}   L1 {fmt((lastDetect.t_detect_us - lastDetect.t_win_end_us) / 1000, 1, ' ms')}
          </div>
        )}
      </section>

      <section className="card span2 transcript-card">
        <div className="row between">
          <h2>Live transcript</h2>
          <span className="muted num">{cur ? `${cur.id} · ${cur.transport}` : 'waiting for a wake'}</span>
        </div>
        <div className="transcript">
          {cur ? (cur.done || cur.final
            ? <span className="final">{cur.final || <span className="muted">(no speech recognised)</span>}</span>
            : <span className="partial">{cur.partial || '…'}</span>) : <span className="muted">—</span>}
        </div>
        {cur?.finalRaw !== undefined && cur.finalRaw !== cur.final && (
          <div className="muted">Vosk heard: “{cur.finalRaw}” · wake word located by {cur.kwMode ?? '—'}</div>
        )}
        {cur?.done && (
          <div className="kv num">
            END→final {fmt(cur.endToFinalMs ?? cur.lat.final_after_last_byte_ms, 1, ' ms')}   server lag med {fmt(cur.lagMedian, 1, ' ms')}
            p90 {fmt(cur.lagP90, 1, ' ms')}   lost {cur.lost ?? '—'}/{cur.packets ?? '—'} pkts
            {cur.clock && `   clock ±${fmt(cur.clock.error_bound_us / 1000, 2, ' ms')}`}
          </div>
        )}
      </section>

      <section className="card span2">
        <h2>Recent wakes</h2>
        {recent.length ? (
          <table className="tbl num">
            <thead><tr><th>session</th><th>via</th><th>transcript</th><th>END→final</th><th>lost</th></tr></thead>
            <tbody>
              {recent.map((x) => (
                <tr key={x.id}>
                  <td>{x.id}</td><td>{x.transport}</td>
                  <td className="text">{x.final ?? <span className="partial">{x.partial}</span>}</td>
                  <td>{fmt(x.endToFinalMs ?? x.lat.final_after_last_byte_ms, 1, ' ms')}</td><td>{x.lost ?? '—'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        ) : <div className="empty">no wakes yet</div>}
      </section>
    </div>
  )
}
