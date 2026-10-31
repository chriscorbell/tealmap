import { Activity, ChevronDown, Layers } from 'lucide-react'
import { useEffect, useRef, useState } from 'react'
import { type CameraState, type MapInfo, MapEngine, type Stats } from './map/engine'

function parseHash() {
  const [id, ...nums] = location.hash.slice(1).split('/')
  const v = nums.map(Number)
  if (!id || v.length !== 5 || v.some((n) => !Number.isFinite(n))) return { id: id || null, view: undefined }
  return { id, view: { x: v[0], z: v[1], dist: v[2], az: v[3], polar: v[4] } }
}

export default function App() {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const engineRef = useRef<MapEngine | null>(null)
  const [maps, setMaps] = useState<MapInfo[]>([])
  const [current, setCurrent] = useState<MapInfo | null>(null)
  const [stats, setStats] = useState<Stats | null>(null)
  const coordsRef = useRef<HTMLElement>(null)
  const [bench] = useState(() => new URLSearchParams(location.search).has('bench'))
  const [statsOpen, setStatsOpen] = useState(bench)
  const [menu, setMenu] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    const engine = new MapEngine(canvasRef.current!, {
      onStats: setStats,
      // Straight to the DOM: this runs every frame the camera moves.
      onCamera: (c: CameraState) => {
        const values = coordsRef.current?.querySelectorAll('b')
        if (values) [c.x, c.y, c.z].forEach((v, i) => (values[i].textContent = String(Math.round(v))))
      },
      bench,
    })
    engineRef.current = engine
    ;(window as unknown as { tealmap: MapEngine }).tealmap = engine
    fetch('/api/maps')
      .then((r) => r.json() as Promise<MapInfo[]>)
      .then((list) => {
        setMaps(list)
        const { id, view } = parseHash()
        const pick = list.find((m) => m.id === id) ?? list[0]
        if (!pick) {
          setError('No maps configured.')
          return
        }
        setCurrent(pick)
        engine.setMap(pick, pick.id === id ? view : undefined).catch((e) => setError(String(e)))
      })
      .catch((e) => setError(String(e)))
    const onHash = () => {
      const { id, view } = parseHash()
      if (view && id === engine.mapId) engine.goTo(view)
    }
    window.addEventListener('hashchange', onHash)
    return () => {
      window.removeEventListener('hashchange', onHash)
      engine.dispose()
    }
  }, [bench])

  const switchMap = (m: MapInfo) => {
    setMenu(false)
    if (m.id === current?.id) return
    setCurrent(m)
    engineRef.current?.setMap(m).catch((e) => setError(String(e)))
  }

  const loading = (stats?.pending ?? 0) > 0

  return (
    <div className="app">
      <canvas ref={canvasRef} className="map" onContextMenu={(e) => e.preventDefault()} />

      <div className={`progress ${loading ? 'on' : ''}`} />

      <header className="bar top-left">
        <div className="brand">
          <Layers size={16} strokeWidth={2.2} />
          <span>tealmap</span>
        </div>
        {maps.length > 0 && (
          <div className="picker">
            <button className="picker-button" onClick={() => setMenu((v) => !v)} aria-expanded={menu}>
              {current?.name ?? '…'}
              <ChevronDown size={14} className={menu ? 'flip' : ''} />
            </button>
            <ul className={`menu ${menu ? 'open' : ''}`} role="menu">
              {maps.map((m) => (
                <li key={m.id}>
                  <button className={m.id === current?.id ? 'active' : ''} onClick={() => switchMap(m)}>
                    {m.name}
                  </button>
                </li>
              ))}
            </ul>
          </div>
        )}
      </header>

      <div className="bar top-right">
        <button className={`icon ${statsOpen ? 'active' : ''}`} onClick={() => setStatsOpen((v) => !v)} aria-label="Stats">
          <Activity size={16} />
        </button>
      </div>

      <section className={`panel ${statsOpen ? 'open' : ''}`} aria-hidden={!statsOpen}>
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

      <footer ref={coordsRef} className="bar bottom-left coords">
        <span>
          x <b>0</b>
        </span>
        <span>
          y <b>0</b>
        </span>
        <span>
          z <b>0</b>
        </span>
      </footer>

      {error && <div className="error">{error}</div>}
    </div>
  )
}
