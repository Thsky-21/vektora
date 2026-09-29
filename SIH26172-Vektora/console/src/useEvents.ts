import { useCallback, useEffect, useReducer, useRef, useState } from 'react'
import { initialState, reduce, type Ev, type State } from './store'

export type Mode = 'live' | 'replay'
export type Conn = 'connecting' | 'open' | 'closed' | 'replay'

type Action = { kind: 'ev'; e: Ev } | { kind: 'reset' }
const reducer = (s: State, a: Action): State => (a.kind === 'reset' ? initialState : reduce(s, a.e))

// ?server=192.168.1.10:8000 points the console at another machine; default is
// same origin (Vite dev proxy, or FastAPI serving console/dist).
function wsUrl() {
  const q = new URLSearchParams(location.search).get('server')
  const host = q ?? location.host
  return `${location.protocol === 'https:' ? 'wss' : 'ws'}://${host}/ws/events`
}

export function useEvents(mode: Mode) {
  const [state, dispatch] = useReducer(reducer, initialState)
  const [conn, setConn] = useState<Conn>(mode === 'live' ? 'connecting' : 'replay')
  const [replayInfo, setReplayInfo] = useState<{ name: string; n: number; played: number } | null>(null)
  const [speed, setSpeed] = useState(1)
  const timers = useRef<number[]>([])

  // -- live ---------------------------------------------------------------
  useEffect(() => {
    if (mode !== 'live') return
    dispatch({ kind: 'reset' })
    let ws: WebSocket | null = null
    let retry = 0
    let alive = true
    const open = () => {
      setConn('connecting')
      ws = new WebSocket(wsUrl())
      ws.onopen = () => setConn('open')
      ws.onmessage = (m) => {
        // rx_ms: browser receipt time, the common clock for Ledger trial
        // markers and the detects that answer them
        try { dispatch({ kind: 'ev', e: { ...JSON.parse(m.data), rx_ms: Date.now() } }) } catch { /* ignore */ }
      }
      ws.onclose = () => {
        setConn('closed')
        if (alive) retry = window.setTimeout(open, 1000)
      }
    }
    open()
    return () => { alive = false; clearTimeout(retry); ws?.close() }
  }, [mode])

  // -- replay -------------------------------------------------------------
  const clearTimers = () => { timers.current.forEach(clearTimeout); timers.current = [] }

  const play = useCallback((events: Ev[], name: string, spd: number) => {
    clearTimers()
    dispatch({ kind: 'reset' })
    setConn('replay')
    const evs = events.filter((e) => e && typeof e === 'object')
    const t0 = evs.find((e) => typeof e.t_mono_us === 'number')?.t_mono_us ?? 0
    setReplayInfo({ name, n: evs.length, played: 0 })
    evs.forEach((e, i) => {
      // keep the original spacing (that's what makes latency visible), but cap
      // dead air between sessions at 2 s so a recording doesn't drag
      const dt = typeof e.t_mono_us === 'number' ? (e.t_mono_us - t0) / 1000 / spd : i
      timers.current.push(window.setTimeout(() => {
        dispatch({ kind: 'ev', e })
        setReplayInfo((r) => (r ? { ...r, played: i + 1 } : r))
      }, dt))
    })
  }, [])

  const loadReplay = useCallback(async (src: File | string, spd = speed) => {
    const text = typeof src === 'string' ? await (await fetch(src)).text() : await src.text()
    let evs: Ev[]
    try {
      const j = JSON.parse(text)
      evs = Array.isArray(j) ? j : j.events
    } catch { // JSONL (a sessions/*.jsonl file)
      evs = text.split('\n').filter((l) => l.trim()).map((l) => JSON.parse(l))
    }
    play(compressGaps(evs), typeof src === 'string' ? src : src.name, spd)
  }, [play, speed])

  useEffect(() => {
    if (mode !== 'replay') return
    loadReplay('/demo_session.json').catch(() => setReplayInfo(null))
    return clearTimers
  }, [mode]) // eslint-disable-line react-hooks/exhaustive-deps

  // Local events (Ledger trial / soak markers): go into the same stream, so
  // "Download recording" keeps them next to the detects they are scored on.
  const push = useCallback((e: Ev) => dispatch({ kind: 'ev', e: { ...e, rx_ms: Date.now(), source: 'console' } }), [])

  return { state, conn, replayInfo, loadReplay, speed, setSpeed, push }
}

// Collapse idle gaps > 2 s between events to 2 s (only between, never inside,
// a session's timing that matters).
function compressGaps(evs: Ev[]): Ev[] {
  let shift = 0
  let prev: number | undefined
  return evs.map((e) => {
    if (typeof e.t_mono_us !== 'number') return e
    if (prev !== undefined && e.t_mono_us - prev > 2e6) shift += e.t_mono_us - prev - 2e6
    prev = e.t_mono_us
    return { ...e, t_mono_us: e.t_mono_us - shift }
  })
}

export function downloadJson(obj: unknown, name: string) {
  const a = document.createElement('a')
  a.href = URL.createObjectURL(new Blob([JSON.stringify(obj)], { type: 'application/json' }))
  a.download = name
  a.click()
  URL.revokeObjectURL(a.href)
}
