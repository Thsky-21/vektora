// Event store: every console view is derived from the stream of /ws/events
// messages (live) or a recorded array of the same messages (replay).

export type Ev = { type: string; t_mono_us?: number; wall?: string; session?: string; source?: string; [k: string]: any }

export type Latency = {
  L1_ms?: number; L2_ms?: number; L3_ms?: number; L3_error_ms?: number; L4_ms?: number
  total_ms?: number; keyword_end_to_first_text_ms?: number
  upload_ms?: number; final_after_last_byte_ms?: number; first_text_after_request_ms?: number
}

export type Sess = {
  id: string
  transport: string
  fw?: string          // from HELLO; 'test_sender' = simulated, never evidence
  startWall?: string
  partial: string
  final?: string
  finalRaw?: string    // exactly what Vosk said; `final` shows the wake word as "Vektora"
  kwMode?: string      // how the wake word was located: timed | fuzzy | prepend | ...
  done: boolean
  lat: Latency
  lost?: number
  packets?: number
  endToFinalMs?: number
  lagMedian?: number
  lagP90?: number
  rtf?: number
  clock?: { rtt_us: number; offset_us: number; error_bound_us: number }
}

export type State = {
  events: Ev[]          // everything, for "download recording"
  tick?: Ev             // latest device telemetry tick
  ticks: { t: number; score: number }[]
  detects: Ev[]
  links: Ev[]
  sessions: Record<string, Sess>
  order: string[]       // session ids, oldest first
  asr?: Record<string, any>
  maxRamPeak?: number
  cpuIdle: number[]     // cpu_avg samples while state == LISTEN (for the compliance card)
}

export const initialState: State = { events: [], ticks: [], detects: [], links: [], sessions: {}, order: [], cpuIdle: [] }

const TICK_KEEP = 240 // 2 Hz -> 2 min of score trace

export function reduce(s: State, e: Ev): State {
  const n: State = { ...s, events: [...s.events, e] }
  if (e.type === 'hello_console') return { ...n, asr: e.asr }

  if (e.source === 'device' || e.type === 'tick') {
    if (e.type === 'tick') {
      n.tick = e
      n.ticks = [...s.ticks, { t: e.t_mono_us ?? Date.now() * 1000, score: Number(e.score ?? 0) }].slice(-TICK_KEEP)
      if (typeof e.ram_peak === 'number') n.maxRamPeak = Math.max(s.maxRamPeak ?? 0, e.ram_peak)
      if (e.state === 'LISTEN' && typeof e.cpu_avg === 'number') n.cpuIdle = [...s.cpuIdle, e.cpu_avg].slice(-7200)
    } else if (e.type === 'detect') n.detects = [...s.detects, e].slice(-50)
    else if (e.type === 'link') n.links = [...s.links, e].slice(-50)
    return n
  }

  const id = e.session
  if (!id) return n
  const prev: Sess = s.sessions[id] ?? { id, transport: '—', partial: '', done: false, lat: {} }
  const x: Sess = { ...prev }
  switch (e.type) {
    case 'session_start': x.transport = e.transport; x.startWall = e.wall; break
    case 'hello': x.fw = e.fw; break
    case 'asr_partial': x.partial = e.text; break
    case 'asr_final':
      x.final = e.text; x.finalRaw = e.text_raw; x.kwMode = e.kw?.mode; x.lagMedian = e.decode_lag_median_ms; x.lagP90 = e.decode_lag_p90_ms; x.rtf = e.rtf
      break
    case 'clock_sync': x.clock = { rtt_us: e.rtt_us, offset_us: e.offset_us, error_bound_us: e.error_bound_us }; break
    case 'latency': x.lat = { ...x.lat, ...stripLat(e) }; break
    case 'session_end':
      x.done = true; x.final = e.text; x.finalRaw = e.text_raw ?? x.finalRaw; x.kwMode = e.kw?.mode ?? x.kwMode; x.lat = { ...x.lat, ...(e.latency ?? {}) }
      x.lost = e.lost; x.packets = e.packets; x.endToFinalMs = e.end_to_final_ms
      break
  }
  n.sessions = { ...s.sessions, [id]: x }
  n.order = s.sessions[id] ? s.order : [...s.order, id]
  return n
}

function stripLat(e: Ev): Latency {
  const { type, t_mono_us, wall, session, partial, ...rest } = e
  void type; void t_mono_us; void wall; void session; void partial
  return rest as Latency
}

// -- small stats helpers -------------------------------------------------------
export function median(xs: number[]): number | undefined {
  if (!xs.length) return undefined
  const a = [...xs].sort((p, q) => p - q)
  return a.length % 2 ? a[(a.length - 1) / 2] : (a[a.length / 2 - 1] + a[a.length / 2]) / 2
}
export function p90(xs: number[]): number | undefined {
  if (!xs.length) return undefined
  const a = [...xs].sort((p, q) => p - q)
  return a[Math.floor(0.9 * (a.length - 1))]
}
export const fmt = (v: number | undefined | null, d = 1, unit = '') =>
  v === undefined || v === null || Number.isNaN(v) ? '—' : `${v.toFixed(d)}${unit}`

// A wake counts as hardware evidence only if a real firmware sent its HELLO.
export const isHardware = (x: Sess) => !!x.fw && x.fw !== 'test_sender'

// Ledger scoring (pages/Ledger.tsx, pages/Evidence.tsx, tools/analyze.py agree):
// a device detect within LEDGER_WINDOW_MS after a trial marker is a hit.
export const LEDGER_WINDOW_MS = 3000
export function ledgerSummary(s: State) {
  const dets = s.detects.map((d) => d.rx_ms).filter((t): t is number => typeof t === 'number')
  const hit = (t0: number) => dets.some((d) => d >= t0 && d <= t0 + LEDGER_WINDOW_MS)
  const trials = s.events.filter((e) => e.type === 'trial')
  const byDist: Record<string, { n: number; h: number }> = {}
  for (const t of trials.filter((x) => x.kind === 'positive')) {
    const r = (byDist[t.label] ??= { n: 0, h: 0 })
    r.n += 1
    r.h += hit(t.rx_ms) ? 1 : 0
  }
  const soak = s.events.filter((e) => e.type === 'soak')
  const start = [...soak].reverse().find((e) => e.action === 'start')
  const stop = start && soak.find((e) => e.action === 'stop' && e.rx_ms >= start.rx_ms)
  let fa: { n: number; hours: number } | undefined
  if (start && stop) {
    const n = dets.filter((d) => d >= start.rx_ms && d <= stop.rx_ms).length
    fa = { n, hours: (stop.rx_ms - start.rx_ms) / 3.6e6 }
  }
  return { byDist, fa }
}
