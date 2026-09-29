import { fmt, median, p90, type Sess, type State } from '../store'

const STAGES = [
  { k: 'L1_ms', name: 'L1 detect', cls: 's1' },
  { k: 'L2_ms', name: 'L2 wake→link', cls: 's2' },
  { k: 'L3_ms', name: 'L3 uplink', cls: 's3' },
  { k: 'L4_ms', name: 'L4 ASR first text', cls: 's4' },
] as const

const complete = (x: Sess) => x.done && STAGES.slice(0, 3).every((st) => typeof x.lat[st.k] === 'number')

export function Latency({ s }: { s: State }) {
  const all = s.order.map((id) => s.sessions[id]).filter((x) => x.done)
  const udp = all.filter((x) => x.transport === 'udp')
  const good = udp.filter(complete).slice(-20)
  const skipped = udp.length - udp.filter(complete).length
  const http = all.filter((x) => x.transport === 'http').slice(-20)
  const max = Math.max(1, ...good.map((x) => STAGES.reduce((a, st) => a + (x.lat[st.k] ?? 0), 0)))
  const col = (f: (x: Sess) => number | undefined, xs: Sess[]) => xs.map(f).filter((v): v is number => typeof v === 'number')

  return (
    <div className="grid">
      <section className="card span2">
        <div className="row between">
          <h2>Wake latency waterfall <span className="muted">(UDP link, last {good.length})</span></h2>
          <div className="legend">{STAGES.map((st) => <span key={st.k}><i className={`sw ${st.cls}`} />{st.name}</span>)}</div>
        </div>
        {good.length ? good.map((x) => (
          <div className="wf-row" key={x.id}>
            <span className="num wf-id">{x.id.slice(-12)}{x.fw === 'test_sender' && <span className="muted"> sim</span>}</span>
            <div className="wf-track">
              {STAGES.map((st) => {
                const v = x.lat[st.k]
                return typeof v === 'number' && v > 0
                  ? <div key={st.k} className={`wf-seg ${st.cls}`} style={{ width: `${(v / max) * 100}%` }} title={`${st.name}: ${v.toFixed(1)} ms`} />
                  : null
              })}
            </div>
            <span className="num wf-total">{fmt(x.lat.total_ms, 1, ' ms')}</span>
          </div>
        )) : <div className="empty">no complete wakes yet</div>}
        {skipped > 0 && <div className="kv muted">{skipped} incomplete wake(s) skipped (missing clock sync or HELLO fields)</div>}
      </section>

      <section className="card span2">
        <h2>Summary</h2>
        <table className="tbl num">
          <thead><tr><th>stage</th><th>median</th><th>p90</th><th>n</th></tr></thead>
          <tbody>
            {[...STAGES.map((st) => ({ name: st.name, v: col((x) => x.lat[st.k], good) })),
              { name: 'Total keyword-end → server (L1+L2+L3)', v: col((x) => x.lat.total_ms, good) },
              { name: 'Keyword-end → first text', v: col((x) => x.lat.keyword_end_to_first_text_ms, good) },
              { name: 'Server decode lag (audio in → text out)', v: col((x) => x.lagMedian, all) },
              { name: 'END → final transcript (UDP)', v: col((x) => x.endToFinalMs, udp) },
              { name: 'Last byte → final transcript (HTTP)', v: col((x) => x.lat.final_after_last_byte_ms, http) },
            ].map((r) => (
              <tr key={r.name}><td className="text">{r.name}</td><td>{fmt(median(r.v), 1, ' ms')}</td><td>{fmt(p90(r.v), 1, ' ms')}</td><td>{r.v.length}</td></tr>
            ))}
          </tbody>
        </table>
      </section>

      <section className="card span2 method">
        <h2>Measurement method</h2>
        <ul>
          <li><b>L1</b> = t_detect − t_win_end, device clock (esp_timer, µs). t_win_end is the capture time of the last sample in the triggering window.</li>
          <li><b>L2</b> = t_sock_ready − t_detect, device clock: radio start + association + socket.</li>
          <li><b>L3</b> = server receipt of the first packet − t_sock_ready. Clocks are synced from 3+ PING/PONG pairs; the min-RTT sample gives offset = t_srv − (t_dev + RTT/2), with error bound ±RTT/2 shown per wake.</li>
          <li><b>L4</b> = first Vosk partial − first audio receipt, server clock (QueryPerformanceCounter, 100 ns).</li>
          <li>Incomplete wakes are counted, not hidden. Median / p90 over the last 20.</li>
        </ul>
      </section>
    </div>
  )
}
