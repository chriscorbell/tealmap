import { Activity, ChevronDown, CircleHelp, House, Layers, Navigation2 } from 'lucide-react'
import { useCallback, useEffect, useRef, useState } from 'react'
import { Coords } from './Coords'
import { Help } from './Help'
import { type CameraState, type LoadState, type MapInfo, MapEngine, type Stats, WebGL2Missing } from './map/engine'
import { PinCard } from './PinCard'
import { parseHash, type Pin } from './route'
import { type MapStatus, statusLine } from './status'

/** Moving less than this between press and release, in CSS pixels, counts as a click. */
const CLICK_SLOP = { mouse: 5, other: 10 }
const CLICK_MS = 500
const NOTICE_MS = 6000
const RETRY_FIRST_MS = 1000
const RETRY_LAST_MS = 30_000
/** How far the pin's card stays from the screen's edges, in CSS pixels. */
const CARD_MARGIN = 8
/** How far down the screen a card below its pin starts, to clear the top bars. */
const CARD_CLEAR_TOP = 64

export default function App() {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const engineRef = useRef<MapEngine | null>(null)
  const mapsRef = useRef<MapInfo[]>([])
  const xRef = useRef<HTMLInputElement>(null)
  const yRef = useRef<HTMLElement>(null)
  const zRef = useRef<HTMLInputElement>(null)
  const compassRef = useRef<HTMLSpanElement>(null)
  const pickerRef = useRef<HTMLDivElement>(null)
  const pickerButtonRef = useRef<HTMLButtonElement>(null)
  const markerRef = useRef<HTMLDivElement>(null)
  const pinRef = useRef<Pin | null>(null)
  const pressRef = useRef<{ id: number; x: number; y: number; at: number; kind: string } | null>(null)
  const pointersRef = useRef(new Set<number>())
  const [maps, setMaps] = useState<MapInfo[]>([])
  const [current, setCurrent] = useState<MapInfo | null>(null)
  const [stats, setStats] = useState<Stats | null>(null)
  const [pin, setPinState] = useState<Pin | null>(null)
  const [bench] = useState(() => new URLSearchParams(location.search).has('bench'))
  const [panel, setPanel] = useState<'stats' | 'help' | null>(bench ? 'stats' : null)
  const [menu, setMenu] = useState(false)
  /** Why the viewer can't run at all, or why the open map failed. */
  const [error, setError] = useState<string | null>(null)
  const [notice, setNotice] = useState<string | null>(null)
  const [noServer, setNoServer] = useState(false)
  const [load, setLoad] = useState<LoadState>({ problem: null, opened: false, hasTiles: false })
  const [statuses, setStatuses] = useState<Record<string, MapStatus>>({})

  // For the listeners set up once, which would otherwise only see the first value.
  const menuRef = useRef(menu)
  const panelRef = useRef(panel)
  useEffect(() => {
    menuRef.current = menu
    panelRef.current = panel
  }, [menu, panel])

  const placeMarker = useCallback(() => {
    const el = markerRef.current
    const p = pinRef.current
    const engine = engineRef.current
    if (!el || !p || !engine) return
    // On top of the block, not inside it.
    const at = engine.project(p.x + 0.5, p.y + 1, p.z + 0.5)
    const { clientWidth: width, clientHeight: height } = canvasRef.current!
    // While another map opens, the camera is still looking at the last one.
    const onScreen = engine.loaded && at && at.x >= 0 && at.x <= width && at.y >= 0 && at.y <= height
    el.style.visibility = onScreen ? 'visible' : 'hidden'
    if (!onScreen) return
    el.style.transform = `translate(${at.x}px, ${at.y}px)`
    el.classList.toggle('below', at.y < 90)
    // Under the top bars, a card's buttons couldn't be reached.
    el.style.setProperty('--drop', `${Math.max(0, CARD_CLEAR_TOP - at.y)}px`)
    const half = (el.querySelector<HTMLElement>('.pin-card')?.offsetWidth ?? 0) / 2
    const nudge = Math.min(Math.max(at.x, half + CARD_MARGIN), width - half - CARD_MARGIN) - at.x
    el.style.setProperty('--nudge', `${nudge}px`)
  }, [])

  const writeCoords = useCallback((c: Pick<CameraState, 'x' | 'y' | 'z'>) => {
    if (yRef.current) yRef.current.textContent = String(Math.round(c.y))
    // Don't type over someone typing, in either field.
    const typing = document.activeElement === xRef.current || document.activeElement === zRef.current
    if (typing || !xRef.current || !zRef.current) return
    xRef.current.value = String(Math.round(c.x))
    zRef.current.value = String(Math.round(c.z))
  }, [])

  const setPin = (p: Pin | null) => {
    pinRef.current = p
    engineRef.current?.setPin(p)
    setPinState(p)
  }

  useEffect(() => {
    let engine: MapEngine
    try {
      engine = new MapEngine(canvasRef.current!, {
        onStats: setStats,
        // Straight to the DOM: this runs every frame the camera moves.
        onCamera: (c: CameraState) => {
          writeCoords(c)
          placeMarker()
          if (compassRef.current) compassRef.current.style.transform = `rotate(${c.az}deg)`
        },
        onLoadState: setLoad,
        onMapStatus: (id, status) => setStatuses((all) => ({ ...all, [id]: status })),
        bench,
      })
    } catch (e) {
      console.error(e)
      const why =
        e instanceof WebGL2Missing
          ? 'This map needs WebGL2, which this browser or device doesn’t provide.'
          : 'This map’s renderer couldn’t start in this browser.'
      queueMicrotask(() => setError(why))
      return
    }
    engineRef.current = engine
    ;(window as unknown as { tealmap: MapEngine }).tealmap = engine

    const open = (m: MapInfo, view: Parameters<MapEngine['setMap']>[1]) => {
      setCurrent(m)
      setError(null)
      engine.setMap(m, view).catch((e) => {
        console.error(e)
        setError('This map couldn’t be opened.')
      })
    }

    // Startup, links pasted into the address bar and the Back button all come through here.
    const follow = () => {
      engine.cancelHashWrite()
      const { id, view, pin: linked } = parseHash(location.hash)
      const list = mapsRef.current
      const target = list.find((m) => m.id === id)
      setNotice(id && !target ? `There's no map called ${id}.` : null)
      if (!target && engine.mapId) return
      const pick = target ?? list[0]
      if (!pick) return
      if (pick.id !== engine.mapId) open(pick, target ? view : undefined)
      else if (view) engine.flyTo(view)
      setPin(target ? (linked ?? null) : null)
    }

    let closed = false
    let retryTimer = 0
    const loadMaps = async (retry: number) => {
      try {
        const res = await fetch('/api/maps')
        if (!res.ok) throw new Error(`${res.status} ${res.statusText}`)
        const list = (await res.json()) as MapInfo[]
        if (closed) return
        setNoServer(false)
        mapsRef.current = list
        setMaps(list)
        setStatuses(Object.fromEntries(list.flatMap((m) => (m.status ? [[m.id, m.status]] : []))))
        follow()
      } catch (e) {
        if (closed) return
        console.warn(`maps: ${e}`)
        setNoServer(true)
        retryTimer = window.setTimeout(() => loadMaps(Math.min(retry * 2, RETRY_LAST_MS)), retry)
      }
    }
    void loadMaps(RETRY_FIRST_MS)
    // popstate comes first, in the same task as the URL changing, so no pending
    // write can replace the URL before it's read. hashchange covers the rest.
    window.addEventListener('popstate', follow)
    window.addEventListener('hashchange', follow)
    // Typing and browser shortcuts keep their keys. Everything else on the page
    // can share them with the map, which is most of what the page is.
    const keyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        const menuOpen = menuRef.current
        if (menuOpen) {
          setMenu(false)
          pickerButtonRef.current?.focus()
        } else if (panelRef.current) setPanel(null)
        else if (pinRef.current) setPin(null)
        return
      }
      const typing = e.target instanceof HTMLElement && e.target.closest('input, textarea, select, [contenteditable]')
      if (typing || e.ctrlKey || e.metaKey || e.altKey) return
      if (e.key === '?') {
        setPanel((p) => (p === 'help' ? null : 'help'))
        return
      }
      if (e.repeat ? engine.keyHeld(e.code) : engine.keyDown(e.code)) e.preventDefault()
    }
    const keyUp = (e: KeyboardEvent) => engine.keyUp(e.code)
    const leave = () => engine.releaseKeys()
    const clickAway = (e: PointerEvent) => {
      if (menuRef.current && !pickerRef.current?.contains(e.target as Node)) setMenu(false)
    }
    window.addEventListener('keydown', keyDown)
    window.addEventListener('keyup', keyUp)
    window.addEventListener('blur', leave)
    window.addEventListener('pointerdown', clickAway)
    return () => {
      window.removeEventListener('popstate', follow)
      window.removeEventListener('hashchange', follow)
      window.removeEventListener('keydown', keyDown)
      window.removeEventListener('keyup', keyUp)
      window.removeEventListener('blur', leave)
      window.removeEventListener('pointerdown', clickAway)
      closed = true
      clearTimeout(retryTimer)
      engine.dispose()
    }
  }, [bench, placeMarker, writeCoords])

  // A notice is about something that just happened, and goes away by itself.
  useEffect(() => {
    if (!notice) return
    const timer = setTimeout(() => setNotice(null), NOTICE_MS)
    return () => clearTimeout(timer)
  }, [notice])

  const switchMap = (m: MapInfo) => {
    setMenu(false)
    const engine = engineRef.current
    if (!engine || m.id === current?.id) return
    setNotice(null)
    // A step the Back button undoes, like any other move to somewhere else.
    history.replaceState(null, '', engine.hash() || location.hash)
    history.pushState(null, '', `#${m.id}`)
    setPin(null)
    setCurrent(m)
    setError(null)
    engine.setMap(m).catch((e) => {
      console.error(e)
      setError('This map couldn’t be opened.')
    })
  }

  const goToCoords = (to: { x?: number; z?: number }) => {
    const engine = engineRef.current
    // Until the map opens, there's no camera to go from.
    if (!engine?.mapId || !engine.loaded) return
    const here = engine.view()
    const x = Math.round(to.x ?? here.x)
    const z = Math.round(to.z ?? here.z)
    if (x !== here.x || z !== here.z) engine.jumpTo({ ...here, x, z })
  }

  const linkHere = () => `${location.origin}${location.pathname}${engineRef.current?.hash() ?? ''}`

  const press = (e: React.PointerEvent) => {
    // The first finger of a gesture starts the count over, in case a release got lost.
    if (e.isPrimary) pointersRef.current.clear()
    pointersRef.current.add(e.pointerId)
    // A second finger makes this a gesture, not a tap, until they're all lifted.
    pressRef.current =
      pointersRef.current.size === 1 && e.button === 0
        ? { id: e.pointerId, x: e.clientX, y: e.clientY, at: e.timeStamp, kind: e.pointerType }
        : null
  }

  // Dragging away and back is still a drag, wherever it ends.
  const move = (e: React.PointerEvent) => {
    const p = pressRef.current
    if (!p || p.id !== e.pointerId) return
    const slop = p.kind === 'mouse' ? CLICK_SLOP.mouse : CLICK_SLOP.other
    if (Math.hypot(e.clientX - p.x, e.clientY - p.y) > slop) pressRef.current = null
  }

  const release = (e: React.PointerEvent) => {
    pointersRef.current.delete(e.pointerId)
    const p = pressRef.current
    pressRef.current = null
    if (!p || p.id !== e.pointerId || e.timeStamp - p.at > CLICK_MS) return
    // The first click away from a pin you can see only clears it, so a stray click
    // can't lose your place.
    const showing = pinRef.current && markerRef.current?.style.visibility !== 'hidden'
    setPin(showing ? null : (engineRef.current?.pick(e.clientX, e.clientY) ?? null))
  }

  const loading = (stats?.pending ?? 0) > 0
  const line = error
    ? { text: error, alert: true }
    : noServer
      ? { text: 'Can’t reach the server. Retrying…', alert: true }
      : notice
        ? { text: notice, alert: false }
        : statusLine(current?.name, load.problem, current ? statuses[current.id] : undefined, load.opened, load.hasTiles)

  return (
    <div className="app">
      <canvas
        ref={canvasRef}
        className="map"
        tabIndex={0}
        role="application"
        aria-label="Map. Arrow keys or W A S D pan, Q and E rotate, R and F tilt, plus and minus zoom."
        onContextMenu={(e) => e.preventDefault()}
        onPointerDown={press}
        onPointerMove={move}
        // Zooming moves the camera, so what's in the fields is out of date.
        onWheel={() => {
          const active = document.activeElement
          if (active === xRef.current || active === zRef.current) (active as HTMLElement).blur()
        }}
        onPointerUp={release}
        onPointerCancel={(e) => {
          pointersRef.current.delete(e.pointerId)
          pressRef.current = null
        }}
      />

      <div className={`progress ${loading ? 'on' : ''}`} />

      {pin && <PinCard key={`${pin.x}/${pin.y}/${pin.z}`} pin={pin} marker={markerRef} link={linkHere} close={() => setPin(null)} place={placeMarker} />}

      <header className="bar top-left">
        <div className="brand">
          <Layers size={16} strokeWidth={2.2} />
          <span>tealmap</span>
        </div>
        {maps.length > 0 && (
          <div className="picker" ref={pickerRef}>
            <button
              ref={pickerButtonRef}
              className="picker-button"
              onClick={() => setMenu((v) => !v)}
              aria-expanded={menu}
              aria-controls="maps"
              aria-label={`Map: ${current?.name ?? 'loading'}`}
            >
              {current?.name ?? '…'}
              <ChevronDown size={14} className={menu ? 'flip' : ''} />
            </button>
            <ul id="maps" className={`menu ${menu ? 'open' : ''}`}>
              {maps.map((m) => (
                <li key={m.id}>
                  <button
                    className={m.id === current?.id ? 'active' : ''}
                    aria-current={m.id === current?.id ? 'true' : undefined}
                    onClick={() => switchMap(m)}
                  >
                    {m.name}
                  </button>
                </li>
              ))}
            </ul>
          </div>
        )}
      </header>

      <div className="bar top-right">
        <button
          className={`icon ${panel === 'help' ? 'active' : ''}`}
          onClick={() => setPanel((p) => (p === 'help' ? null : 'help'))}
          aria-label="Controls"
          aria-expanded={panel === 'help'}
          title="Controls"
        >
          <CircleHelp size={16} />
        </button>
        <button
          className={`icon ${panel === 'stats' ? 'active' : ''}`}
          onClick={() => setPanel((p) => (p === 'stats' ? null : 'stats'))}
          aria-label="Stats"
          aria-expanded={panel === 'stats'}
          title="Stats"
        >
          <Activity size={16} />
        </button>
      </div>

      <div className="bar bottom-right">
        <button className="icon" onClick={() => engineRef.current?.pointNorth()} aria-label="Point north" title="Point north">
          <span ref={compassRef} className="compass">
            <Navigation2 size={16} />
          </span>
        </button>
        <button
          className="icon"
          onClick={() => {
            const engine = engineRef.current
            if (engine?.loaded) engine.jumpTo(engine.homeView())
          }}
          aria-label="Go to the map's start"
          title="Go to the map's start"
        >
          <House size={16} />
        </button>
      </div>

      <section className={`panel ${panel === 'help' ? 'open' : ''}`} aria-hidden={panel !== 'help'}>
        <Help />
      </section>

      <section className={`panel ${panel === 'stats' ? 'open' : ''}`} aria-hidden={panel !== 'stats'}>
        {stats && (
          <dl className="stats">
            <dt>fps</dt>
            <dd>{stats.fps.toFixed(0)}</dd>
            <dt>frame</dt>
            <dd>{stats.frameMs.toFixed(2)} ms</dd>
            {stats.gpuMs !== null && (
              <>
                <dt>gpu p95</dt>
                <dd>{stats.gpuMs.toFixed(2)} ms</dd>
              </>
            )}
            <dt>draw calls</dt>
            <dd>{stats.calls}</dd>
            <dt>triangles</dt>
            <dd>{(stats.triangles / 1e6).toFixed(2)} M</dd>
            <dt>tiles by level</dt>
            <dd>{stats.levels.join(' · ')}</dd>
            <dt>downloaded</dt>
            <dd>{stats.downloadedMB.toFixed(1)} MB</dd>
            <dt>gpu memory</dt>
            <dd>{stats.gpuMB.toFixed(0)} MB</dd>
          </dl>
        )}
      </section>

      <Coords
        x={xRef}
        y={yRef}
        z={zRef}
        go={goToCoords}
        restore={() => {
          const t = engineRef.current?.controls.target
          if (t) writeCoords(t)
        }}
        link={linkHere}
      />

      {line && (
        <div className={`status ${line.alert ? 'alert' : ''}`} role="status">
          {line.text}
        </div>
      )}
    </div>
  )
}
