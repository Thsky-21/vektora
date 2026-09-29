import { useRef } from 'react'
import { downloadJson } from '../useEvents'
import { fmt, isHardware, ledgerSummary, median, type State } from '../store'

type Row = { id: string; metric: string; req: string; measured: string; method: string; status: 'PASS' | 'FAIL' | '—' }

// Every "measured" cell comes from events in this session/recording; anything
// not in the stream is "—" (CLAUDE.md Rule 0.1). No manual entry on purpose.
function rows(s: State): Row[] {
  const udp = s.order.map((id) => s.sessions[id]).filter((x) => x.done && x.transport === 'udp' && isHardware(x))
  const tot = udp.map((x) => x.lat.total_ms).filter((v): v is number => typeof v === 'number')
  const kwText = udp.map((x) => x.lat.keyword_end_to_first_text_ms).filter((v): v is number => typeof v === 'number')
  const cpu = median(s.cpuIdle)
  const peakKB = s.maxRamPeak !== undefined ? s.maxRamPeak / 1024 : undefined
  const idleKB = s.tick?.state === 'LISTEN' && typeof s.tick.ram_used === 'number' ? s.tick.ram_used / 1024 : undefined
  const st = (ok: boolean | undefined): Row['status'] => (ok === undefined ? '—' : ok ? 'PASS' : 'FAIL')
  const L = ledgerSummary(s)
  const tpr = Object.entries(L.byDist).filter(([, r]) => r.n > 0)
    .map(([d, r]) => `${d} ${Math.round((100 * r.h) / r.n)}% (${r.h}/${r.n})`).join(' · ')
  const faPerH = L.fa && L.fa.hours > 0 ? L.fa.n / L.fa.hours : undefined
  return [
    { id: 'C1', metric: 'Internal RAM', req: '< 256 KB',
      measured: peakKB === undefined ? '—' : `idle ${fmt(idleKB)} KB · peak ${fmt(peakKB)} KB`,
      method: 'static (.data+.bss) + heap_caps internal used; peak over session', status: st(peakKB === undefined ? undefined : peakKB < 256) },
    { id: 'C2', metric: 'Idle CPU', req: '< 10 %',
      measured: cpu === undefined ? '—' : `${fmt(cpu)} % (median, n=${s.cpuIdle.length} ticks)`,
      method: 'FreeRTOS run-time stats, 100 − IDLE share, both cores, 240 MHz fixed', status: st(cpu === undefined ? undefined : cpu < 10) },
    { id: 'C3', metric: 'Open-source only', req: 'OSI licences', measured: 'see docs/licences.md',
      method: 'dependency audit', status: '—' },
    { id: 'C4a', metric: 'True-positive rate', req: 'high', measured: tpr || '—',
      method: 'M5 via Ledger: detect ≤3 s after each spoken-keyword marker, per distance', status: '—' },
    { id: 'C4b', metric: 'False accepts / hour', req: '≈ 0',
      measured: faPerH === undefined ? '—' : `${fmt(faPerH, 2)} FA/h (${L.fa!.n} in ${fmt(L.fa!.hours, 2)} h)`,
      method: 'M4 via Ledger soak: every detect on negative-only audio', status: '—' },
    { id: 'C5', metric: 'Keyword-end → server', req: 'low',
      measured: tot.length ? `${fmt(median(tot))} ms median (n=${tot.length})` : '—',
      method: 'L1+L2+L3, PING/PONG clock sync ±RTT/2', status: '—' },
    { id: 'C5b', metric: 'Keyword-end → first text', req: 'low',
      measured: kwText.length ? `${fmt(median(kwText))} ms median (n=${kwText.length})` : '—',
      method: 'first Vosk partial, synced clocks', status: '—' },
  ]
}

const esc = (t: string) => t.replace(/&/g, '&amp;').replace(/</g, '&lt;')

export function Evidence({ s, build }: { s: State; build: string }) {
  const svgRef = useRef<SVGSVGElement>(null)
  const R = rows(s)
  const W = 1600, H = 900, top = 190, rh = 88
  const cols = [60, 170, 520, 760, 1170, 1450]

  const exportPng = () => {
    const svg = svgRef.current
    if (!svg) return
    const img = new Image()
    const url = URL.createObjectURL(new Blob([new XMLSerializer().serializeToString(svg)], { type: 'image/svg+xml' }))
    img.onload = () => {
      const c = document.createElement('canvas')
      c.width = W; c.height = H
      c.getContext('2d')!.drawImage(img, 0, 0)
      URL.revokeObjectURL(url)
      const a = document.createElement('a')
      a.href = c.toDataURL('image/png'); a.download = 'compliance_card.png'; a.click()
    }
    img.src = url
  }
  const exportCsv = () => {
    const lines = ['id,metric,requirement,measured,method,status',
      ...R.map((r) => [r.id, r.metric, r.req, r.measured, r.method, r.status].map((v) => `"${v.replace(/"/g, '""')}"`).join(','))]
    const a = document.createElement('a')
    a.href = URL.createObjectURL(new Blob([lines.join('\n')], { type: 'text/csv' }))
    a.download = 'compliance_card.csv'; a.click()
  }

  return (
    <div className="grid">
      <section className="card span2">
        <div className="row between">
          <h2>Compliance card</h2>
          <div className="row gap">
            <button onClick={exportPng}>Export PNG</button>
            <button onClick={exportCsv}>CSV</button>
            <button onClick={() => downloadJson({ build, rows: R }, 'compliance_card.json')}>JSON</button>
          </div>
        </div>
        <svg ref={svgRef} className="cc" viewBox={`0 0 ${W} ${H}`} xmlns="http://www.w3.org/2000/svg"
          fontFamily="Inter, Segoe UI, Arial, sans-serif">
          <rect width={W} height={H} fill="#ffffff" />
          <rect width={W} height={110} fill="#0B2545" />
          <text x={60} y={70} fill="#fff" fontSize={40} fontWeight={700}>SIH26172 · ISRO · Edge Voice Activator — Compliance</text>
          <text x={60} y={150} fill="#44516b" fontSize={22}>{esc(`build ${build} · generated ${new Date().toLocaleString()} · values measured on hardware; "—" = not yet measured`)}</text>
          {['ID', 'Metric', 'ISRO requirement', 'Measured', 'Method', 'Status'].map((h, i) => (
            <text key={h} x={cols[i]} y={top} fontSize={20} fontWeight={700} fill="#0B2545">{h}</text>
          ))}
          <line x1={60} x2={W - 60} y1={top + 14} y2={top + 14} stroke="#0B2545" strokeWidth={2} />
          {R.map((r, i) => {
            const y = top + 60 + i * rh
            const color = r.status === 'PASS' ? '#1a7f37' : r.status === 'FAIL' ? '#c62828' : '#8a94a6'
            return (
              <g key={r.id}>
                {i % 2 === 1 && <rect x={40} y={y - 44} width={W - 80} height={rh} fill="#f4f6fa" />}
                <text x={cols[0]} y={y} fontSize={22} fontWeight={600} fill="#0B2545">{r.id}</text>
                <text x={cols[1]} y={y} fontSize={24} fontWeight={600} fill="#111">{r.metric}</text>
                <text x={cols[2]} y={y} fontSize={22} fill="#111">{r.req}</text>
                <text x={cols[3]} y={y} fontSize={22} fill="#111" fontFamily="JetBrains Mono, Consolas, monospace">{r.measured}</text>
                <foreignObject x={cols[4]} y={y - 26} width={260} height={rh - 10}>
                  <div style={{ fontSize: 15, color: '#44516b', fontFamily: 'Inter, Arial, sans-serif', lineHeight: 1.25 }}>{r.method}</div>
                </foreignObject>
                <rect x={cols[5]} y={y - 28} width={96} height={38} rx={4} fill={color} />
                <text x={cols[5] + 48} y={y - 2} fontSize={20} fontWeight={700} fill="#fff" textAnchor="middle">{r.status}</text>
              </g>
            )
          })}
        </svg>
      </section>
    </div>
  )
}
