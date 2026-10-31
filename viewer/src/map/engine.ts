import { Box3, Frustum, MathUtils, Matrix4, PerspectiveCamera } from 'three'
import { MapControls } from 'three/examples/jsm/controls/MapControls.js'
import { type Mesh, Pass, Renderer } from './renderer'
import { NO_TOP } from './tile'
import type { Request, Response } from './worker'

export interface MapInfo {
  id: string
  name: string
}

interface Manifest {
  id: string
  name: string
  minY: number
  maxY: number
  tileSize: number
  textures: number
  /** Per region: [rx, rz, low 32 bits, high 32 bits] of the 8×8 hires tile mask. */
  hires: [number, number, number, number][]
  /** Nodes over mapped tiles that have no mesh, as [level, x, z]; level k uses voxels of 2^k blocks. */
  lodMissing?: [number, number, number][]
  /** Older manifests list the tiles that do have one. */
  lod?: [number, number, number][]
  lodLevels: number
}

export interface Stats {
  fps: number
  frameMs: number
  /** GPU time per frame (p95), when the timer extension is available in bench mode. */
  gpuMs: number | null
  calls: number
  triangles: number
  /** Tiles drawn per level, finest first. */
  levels: number[]
  pending: number
  downloadedMB: number
  gpuMB: number
}

export interface CameraState {
  x: number
  y: number
  z: number
}

const BG = [0x0b / 255, 0x14 / 255, 0x16 / 255]
const TILE = 64
const MAX_INFLIGHT = 32
/** Tiles swap with a dissolve: the incoming one draws the pixels the outgoing one drops. */
const FADE_MS = 200
const UPLOAD_BYTES_PER_FRAME = 12e6
const LEAF_NEAR = 150
/** Past this, plants' crossed faces are a pixel or less, and they stop drawing. */
const PLANT_NEAR = 300
/**
 * Largest allowed on-screen size of a voxel per level, in CSS pixels, before
 * refining. The coarsest levels only ever show the whole map, where a voxel of
 * one color is all there is to see, so they get to be finer.
 */
const MAX_VOXEL_PX = [4, 4, 4, 4, 3]
/** Tiles not wanted for this long are dropped. */
const KEEP_MS = 10_000
/** Idle frames before the loop sleeps, and how often it wakes to plan. */
const IDLE_FRAMES = 30
const SLEEP_MS = 1000

interface Tile {
  key: number
  level: number
  x: number
  z: number
  /** 'failed' is a load error with nothing to show; it's retried at `retryAt`. */
  state: 'loading' | 'ready' | 'empty' | 'failed'
  retryAt: number
  req: number
  mesh: Mesh | null
  stale: boolean
  /** Top surface per column (64×64) in the tile's own units, or NO_TOP where empty. */
  tops: Int16Array | null
  lastWanted: number
  drawn: boolean
  /** Start of the current dissolve in or out, 0 when settled. */
  fadeStart: number
  /** Camera distance, for drawing nearest first. */
  distance: number
}

/** A number per tile: coordinates of any map fit, and it hashes faster than a string. */
const key = (level: number, x: number, z: number) => ((x + 0x200000) * 0x400000 + (z + 0x200000)) * 16 + level
const CHILD_OFFSETS = [[0, 0], [1, 0], [0, 1], [1, 1]]
const EMPTY_KIDS: [number, number, number][] = []

class Workers {
  private workers: Worker[]
  private next = 0
  private seq = 0
  private pending = new Map<number, { resolve: (r: Response) => void; worker: Worker }>()

  constructor(n: number) {
    this.workers = Array.from({ length: n }, () => {
      const w = new Worker(new URL('./worker.ts', import.meta.url), { type: 'module' })
      w.onmessage = (e: MessageEvent<Response>) => {
        const p = this.pending.get(e.data.id)
        if (!p) return
        this.pending.delete(e.data.id)
        p.resolve(e.data)
      }
      return w
    })
  }

  request(url: string, fresh: boolean): { id: number; done: Promise<Response> } {
    const id = ++this.seq
    const worker = this.workers[this.next++ % this.workers.length]
    const done = new Promise<Response>((resolve) => this.pending.set(id, { resolve, worker }))
    worker.postMessage({ kind: 'tile', id, url, fresh } satisfies Request)
    return { id, done }
  }

  abort(id: number) {
    const p = this.pending.get(id)
    if (!p) return
    this.pending.delete(id)
    p.worker.postMessage({ kind: 'abort', id } satisfies Request)
  }

  dispose() {
    this.workers.forEach((w) => w.terminate())
  }
}


export class MapEngine {
  readonly camera: PerspectiveCamera
  readonly controls: MapControls
  private renderer: Renderer
  private gl: WebGL2RenderingContext
  private workers = new Workers(2)
  private events: EventSource | null = null

  private map: MapInfo | null = null
  private manifest: Manifest | null = null
  /** Hires tile masks by region, the same masks one level up per bit-grouping. */
  private masks = new Map<number, [number, number]>()
  /** Nodes above region size (level 4 and up), by key. */
  private bigNodes = new Set<number>()
  /** Top-level nodes as [x, z]. */
  private roots: [number, number][] = []
  /**
   * Nodes with no mesh. A sparse area can downsample to nothing at a coarse level
   * while its hires tiles still have geometry; those nodes still refine.
   */
  private missing = new Set<number>()
  private generation = 0
  private ready = false

  private tiles = new Map<number, Tile>()
  /** Tiles mid-dissolve. */
  private fading = new Set<Tile>()
  /** Everything drawing or dissolving, nearest first. */
  private visible: Tile[] = []

  /** Tiles waiting for their GPU upload. Each returns the bytes it uploads. */
  private uploads: (() => number)[] = []
  private pendingUpdate = false
  private autoView = true
  private dirty = true
  private lastPlan = 0
  private raf = 0
  private bench: boolean
  private downloaded = 0
  private frames = 0
  private frameMsAcc = 0
  private lastStats = performance.now()
  private drawCalls = 0
  private triangles = 0
  private onStats: (s: Stats) => void
  private onCamera: (c: CameraState) => void
  private hashTimer = 0
  private gpuTimer: GpuTimer | null = null

  private canvas: HTMLCanvasElement
  private asleep = false
  private sleepTimer = 0
  private idleFrames = 0
  /** Tiles the last plan wanted and didn't have. */
  private planWants = 0
  private wants: { level: number; x: number; z: number; d: number }[] = []
  private wantAt = 0
  private inflight = 0
  // Reused every plan and frame.
  private frustum = new Frustum()
  private box = new Box3()
  private viewProj = new Float32Array(16)
  private matrix = new Matrix4()

  constructor(
    canvas: HTMLCanvasElement,
    opts: { onStats: (s: Stats) => void; onCamera: (c: CameraState) => void; bench?: boolean },
  ) {
    this.canvas = canvas
    this.onStats = opts.onStats
    this.onCamera = opts.onCamera
    this.bench = opts.bench ?? false
    const gl = canvas.getContext('webgl2', {
      // On a high-density screen, edges are already fine, and multisampling
      // takes about half the GPU's frame.
      antialias: window.devicePixelRatio < 1.5,
      alpha: false,
      stencil: false,
      powerPreference: 'high-performance',
    })
    if (!gl) throw new Error('WebGL2 is not available')
    this.gl = gl
    this.renderer = new Renderer(gl)
    gl.clearColor(BG[0], BG[1], BG[2], 1)

    this.camera = new PerspectiveCamera(55, 1, 2, 120000)
    this.controls = new MapControls(this.camera, canvas)
    this.controls.enableDamping = true
    this.controls.dampingFactor = 0.12
    this.controls.screenSpacePanning = false
    this.controls.maxPolarAngle = 1.35
    this.controls.minDistance = 6
    this.controls.maxDistance = 30000
    this.controls.zoomSpeed = 1.3
    this.controls.zoomToCursor = true
    this.controls.addEventListener('start', () => {
      this.autoView = false
    })
    this.controls.addEventListener('change', () => {
      this.dirty = true
      this.lastPlan = 0
      this.scheduleHash()
      this.wake()
    })

    if (this.bench) this.gpuTimer = GpuTimer.create(gl)
    this.resize()
    window.addEventListener('resize', this.resize)
    for (const event of ['pointerdown', 'wheel', 'touchstart']) canvas.addEventListener(event, this.wake, { passive: true })
    this.raf = requestAnimationFrame(this.frame)
  }

  dispose() {
    cancelAnimationFrame(this.raf)
    clearTimeout(this.sleepTimer)
    window.removeEventListener('resize', this.resize)
    for (const event of ['pointerdown', 'wheel', 'touchstart']) this.canvas.removeEventListener(event, this.wake)
    this.events?.close()
    this.clearTiles()
    this.workers.dispose()
    this.controls.dispose()
    this.gl.getExtension('WEBGL_lose_context')?.loseContext()
  }

  private resize = () => {
    const { clientWidth: w, clientHeight: h } = this.canvas
    const ratio = Math.min(window.devicePixelRatio, 2)
    this.canvas.width = Math.floor(w * ratio)
    this.canvas.height = Math.floor(h * ratio)
    this.gl.viewport(0, 0, this.canvas.width, this.canvas.height)
    this.camera.aspect = w / Math.max(h, 1)
    this.camera.updateProjectionMatrix()
    this.dirty = true
    this.wake()
  }

  /** Loads a map. `view` restores a camera position (e.g. from the URL). */
  async setMap(map: MapInfo, view?: { x: number; z: number; dist: number; az: number; polar: number }) {
    const gen = ++this.generation
    this.clearTiles()
    this.map = map
    // Nothing plans against the old map's manifest or textures while loading.
    this.manifest = null
    this.ready = false
    this.pendingUpdate = false
    this.autoView = !view
    // Subscribe first so an update published while loading isn't missed.
    this.listen()
    let manifest: Manifest
    for (;;) {
      const res = await fetch(`/maps/${map.id}/map.json`, { cache: 'no-cache' })
      if (gen !== this.generation) return
      if (res.ok) {
        manifest = await res.json()
        break
      }
      // A fresh server publishes each map once its first render starts.
      await new Promise((r) => setTimeout(r, 3000))
    }
    const texRes = await fetch(`/maps/${map.id}/textures.bin`)
    const texData = new Uint8Array(await texRes.arrayBuffer())
    if (gen !== this.generation) return
    this.setManifest(manifest)
    this.renderer.setBlockTextures(texData, manifest.textures)
    this.ready = true

    if (view) this.placeCamera(view.x, view.z, view.dist, view.az, view.polar)
    else this.placeDefault()
    if (this.pendingUpdate) this.refreshManifest(gen)
    this.dirty = true
    this.wake()
  }

  private setManifest(m: Manifest) {
    this.manifest = m
    // A level 1–3 node is a 2^(3-level) grid cell of a region's mask, so masks
    // answer those without a set. Only nodes bigger than a region need one.
    this.masks = new Map((m.hires ?? []).map(([rx, rz, lo, hi]) => [key(0, rx, rz), [lo, hi]]))
    this.bigNodes = new Set()
    this.roots = []
    let row: [number, number][] = (m.hires ?? []).map(([rx, rz]) => [rx, rz])
    for (let l = 4; l <= m.lodLevels; l++) {
      const seen = new Set<number>()
      const up: [number, number][] = []
      for (const [x, z] of row) {
        const px = x >> 1
        const pz = z >> 1
        const k = key(l, px, pz)
        if (!seen.has(k)) {
          seen.add(k)
          this.bigNodes.add(k)
          up.push([px, pz])
        }
      }
      row = up
    }
    if (m.lodLevels >= 4) this.roots = row
    else {
      // Fewer levels than a region: every node at the top level is a root.
      const seen = new Set<number>()
      for (const [rx, rz, lo, hi] of m.hires ?? []) {
        for (let bit = 0; bit < 64; bit++) {
          if (!(((bit < 32 ? lo : hi) >>> (bit & 31)) & 1)) continue
          const x = (rx * 8 + (bit & 7)) >> m.lodLevels
          const z = (rz * 8 + (bit >> 3)) >> m.lodLevels
          const k = key(m.lodLevels, x, z)
          if (!seen.has(k)) {
            seen.add(k)
            this.roots.push([x, z])
          }
        }
      }
    }
    this.missing = new Set((m.lodMissing ?? []).map(([l, x, z]) => key(l, x, z)))
    if (m.lod) {
      // An older manifest lists meshes instead. Everything else is missing.
      const listed = new Set(m.lod.map(([l, x, z]) => key(l, x, z)))
      this.forEachNode((l, x, z) => {
        const k = key(l, x, z)
        if (!listed.has(k)) this.missing.add(k)
      })
    }
    // Drop resident tiles whose mesh the manifest no longer lists.
    for (const t of [...this.tiles.values()]) {
      const listed = t.level > 0 ? this.exists(t.level, t.x, t.z) && !this.missing.has(t.key) : this.exists(0, t.x, t.z)
      if (!listed) this.dropTile(t)
    }
    this.lastPlan = 0
  }

  /** Visits every LOD node. Only for rare, whole-map work. */
  private forEachNode(f: (level: number, x: number, z: number) => void) {
    const m = this.manifest!
    const walk = (level: number, x: number, z: number) => {
      f(level, x, z)
      if (level > 1) for (const [l, cx, cz] of this.children(level, x, z)) walk(l, cx, cz)
    }
    for (const [x, z] of this.roots) walk(m.lodLevels, x, z)
  }

  /** Starts over the mapped tile closest to the middle of everything mapped. */
  private placeDefault() {
    const m = this.manifest
    // Hires tiles always exist where anything is mapped; LOD meshes may not.
    const hires = m?.hires ?? []
    if (!hires.length) return this.placeCamera(0, 0, 520, 35, 50)
    let cx = 0
    let cz = 0
    let n = 0
    const each = (f: (x: number, z: number) => void) => {
      for (const [rx, rz, lo, hi] of hires) {
        for (let bit = 0; bit < 64; bit++) {
          if (((bit < 32 ? lo : hi) >>> (bit & 31)) & 1) f(rx * 8 + (bit & 7), rz * 8 + (bit >> 3))
        }
      }
    }
    each((x, z) => {
      cx += x
      cz += z
      n++
    })
    cx /= n
    cz /= n
    let best = [0, 0]
    let bestD = Infinity
    each((x, z) => {
      const d = (x - cx) ** 2 + (z - cz) ** 2
      if (d < bestD) {
        bestD = d
        best = [x, z]
      }
    })
    this.placeCamera((best[0] + 0.5) * TILE, (best[1] + 0.5) * TILE, 900, 35, 50)
  }

  private async refreshManifest(gen: number) {
    this.pendingUpdate = false
    const m: Manifest = await (await fetch(`/maps/${this.map!.id}/map.json`, { cache: 'no-cache' })).json()
    if (gen !== this.generation || !this.manifest) return
    const wasEmpty = this.masks.size === 0
    this.setManifest(m)
    // A map opened before its first render finished: frame it once tiles exist.
    if (wasEmpty && this.masks.size && this.autoView) this.placeDefault()
    this.dirty = true
    this.wake()
  }

  get mapId() {
    return this.map?.id ?? null
  }

  goTo(v: { x: number; z: number; dist: number; az: number; polar: number }) {
    this.placeCamera(v.x, v.z, v.dist, v.az, v.polar)
    this.lastPlan = 0
    this.dirty = true
    this.wake()
  }

  /** Debug summary of tile states per level. */
  debug() {
    const out: Record<string, Record<string, number>> = {}
    for (const t of this.tiles.values()) {
      const l = (out[`L${t.level}`] ??= {})
      l[t.state] = (l[t.state] ?? 0) + 1
      if (t.drawn) l.drawn = (l.drawn ?? 0) + 1
    }
    return { ...out, target: this.controls.target.toArray().map(Math.round) }
  }

  private placeCamera(x: number, z: number, dist: number, azDeg: number, polarDeg: number) {
    const az = (azDeg * Math.PI) / 180
    const polar = (polarDeg * Math.PI) / 180
    const y = this.heightAt(x, z) ?? 64
    this.controls.target.set(x, y, z)
    this.camera.position.set(
      x + dist * Math.sin(polar) * Math.sin(az),
      y + dist * Math.cos(polar),
      z + dist * Math.sin(polar) * Math.cos(az),
    )
    this.controls.update()
  }

  /** Camera view as URL-friendly numbers. */
  view() {
    const t = this.controls.target
    const off = this.camera.position.clone().sub(t)
    const dist = off.length()
    const polar = Math.acos(MathUtils.clamp(off.y / dist, -1, 1))
    const az = Math.atan2(off.x, off.z)
    return {
      x: Math.round(t.x),
      z: Math.round(t.z),
      dist: Math.round(dist),
      az: Math.round((az * 180) / Math.PI),
      polar: Math.round((polar * 180) / Math.PI),
    }
  }

  private scheduleHash() {
    clearTimeout(this.hashTimer)
    this.hashTimer = window.setTimeout(() => {
      if (!this.map) return
      const v = this.view()
      history.replaceState(null, '', `#${this.map.id}/${v.x}/${v.z}/${v.dist}/${v.az}/${v.polar}`)
    }, 400)
  }

  private listen() {
    this.events?.close()
    const es = new EventSource('/api/events')
    es.addEventListener('update', (e) => {
      const msg = JSON.parse((e as MessageEvent).data) as {
        map: string
        update: { hires: [number, number][]; lod: [number, number, number][] }
      }
      if (msg.map !== this.map?.id) return
      this.applyUpdate(msg.update)
    })
    // On connect and every reconnect, catch up on anything published while
    // the stream was down; the server doesn't replay events.
    es.onopen = () => {
      if (!this.manifest) {
        this.pendingUpdate = true
        return
      }
      // Revalidate resident tiles too; unchanged ones come back as 304s.
      for (const t of this.tiles.values()) t.stale = true
      this.refreshManifest(this.generation)
    }
    this.events = es
  }

  private async applyUpdate(up: { hires: [number, number][]; lod: [number, number, number][] }) {
    if (!this.manifest) {
      // Still loading: refresh the manifest once it's in.
      this.pendingUpdate = true
      return
    }
    // Tiles can appear or vanish, which changes the manifest.
    if (up.hires.length || up.lod.length) await this.refreshManifest(this.generation)
    for (const [x, z] of up.hires) {
      const t = this.tiles.get(key(0, x, z))
      if (t) t.stale = true
    }
    for (const [l, x, z] of up.lod) {
      const t = this.tiles.get(key(l, x, z))
      if (t) t.stale = true
    }
    this.lastPlan = 0
    this.dirty = true
    this.wake()
  }

  // ---------------------------------------------------------------- planning

  private exists(level: number, x: number, z: number): boolean {
    if (level > 3) return this.bigNodes.has(key(level, x, z))
    // A node at level l is a block of 2^l × 2^l bits in its region's 8×8 mask.
    const shift = 3 - level
    const mask = this.masks.get(key(0, x >> shift, z >> shift))
    if (!mask) return false
    const cx = x - ((x >> shift) << shift)
    const cz = z - ((z >> shift) << shift)
    const span = 1 << level
    for (let bz = cz * span; bz < (cz + 1) * span; bz++) {
      for (let bx = cx * span; bx < (cx + 1) * span; bx++) {
        const bit = bz * 8 + bx
        if ((mask[bit >> 5] >>> (bit & 31)) & 1) return true
      }
    }
    return false
  }

  private children(level: number, x: number, z: number): [number, number, number][] {
    const out: [number, number, number][] = []
    for (const [dx, dz] of CHILD_OFFSETS) {
      if (this.exists(level - 1, x * 2 + dx, z * 2 + dz)) out.push([level - 1, x * 2 + dx, z * 2 + dz])
    }
    return out
  }

  /**
   * Picks what to draw: a quadtree walk from the coarsest tiles that refines
   * wherever a voxel would cover more than MAX_VOXEL_PX pixels. A tile hands over
   * to its children only once all of them are loaded, so there are no holes.
   */
  private plan(now: number) {
    const m = this.manifest
    if (!m || !this.ready) return
    const cam = this.camera.position
    // Pixels per block at distance 1.
    const k = this.canvas.clientHeight / (2 * Math.tan(MathUtils.degToRad(this.camera.fov / 2)))
    this.camera.updateMatrixWorld()
    this.frustum.setFromProjectionMatrix(this.matrix.multiplyMatrices(this.camera.projectionMatrix, this.camera.matrixWorldInverse))
    const frustum = this.frustum
    const box = this.box
    const draw = new Set<Tile>()
    const want: { level: number; x: number; z: number; d: number }[] = []
    const missing = this.missing

    const meshed = (level: number, x: number, z: number) => level === 0 || !missing.has(key(level, x, z))
    const needs = (t: Tile | undefined) => !t || t.stale || (t.state === 'failed' && now >= t.retryAt)
    // Settled: loaded at least once (an old mesh counts while it refreshes),
    // confirmed empty, or a node with no mesh to load.
    const settled = (l: number, x: number, z: number) => {
      if (!meshed(l, x, z)) return true
      const t = this.tiles.get(key(l, x, z))
      return !!t && (!!t.mesh || t.state === 'empty')
    }
    const visit = (level: number, x: number, z: number, sibling: boolean) => {
      const size = TILE << level
      const tile = this.tiles.get(key(level, x, z))
      // A loaded tile knows how tall it really is, which beats the world's height
      // both for culling and for how close the camera really is.
      box.min.set(x * size, tile?.mesh ? tile.mesh.minY : m.minY, z * size)
      box.max.set((x + 1) * size, tile?.mesh ? tile.mesh.maxY : m.maxY, (z + 1) * size)
      const inView = frustum.intersectsBox(box)
      // Off-screen siblings of drawn tiles stay loaded: dropping one would make
      // its parent unsettled and swap the whole group back to coarse. Other
      // off-screen tiles, roots included, expire.
      if (tile && (inView || sibling)) tile.lastWanted = now
      if (!inView) return
      const d = Math.max(box.distanceToPoint(cam), 1)
      if (meshed(level, x, z) && needs(tile)) want.push({ level, x, z, d })
      const kids = level > 0 && ((1 << level) * k) / d > (MAX_VOXEL_PX[level] ?? 2) ? this.children(level, x, z) : EMPTY_KIDS
      if (kids.length) {
        if (kids.every(([l, kx, kz]) => settled(l, kx, kz))) {
          for (const [l, kx, kz] of kids) visit(l, kx, kz, true)
          return
        }
        for (const [l, kx, kz] of kids) {
          const t = this.tiles.get(key(l, kx, kz))
          if (t) t.lastWanted = now
          if (meshed(l, kx, kz) && needs(t)) want.push({ level: l, x: kx, z: kz, d })
        }
      }
      // Anything with a mesh draws, including an old one while it refreshes.
      if (tile?.mesh) {
        tile.distance = d
        draw.add(tile)
      }
    }
    for (const [x, z] of this.roots) visit(m.lodLevels, x, z, false)

    // Swap what's drawn, dissolving tiles in and out.
    for (const t of this.tiles.values()) {
      const on = draw.has(t)
      if (on !== t.drawn) {
        t.drawn = on
        t.fadeStart = now
        this.fading.add(t)
      }
    }
    this.visible = [...draw]
    for (const t of this.fading) if (!t.drawn && t.mesh) this.visible.push(t)
    this.visible.sort((a, b) => a.distance - b.distance)

    // Drop tiles that haven't been wanted for a while.
    for (const t of [...this.tiles.values()]) {
      if (!t.drawn && now - t.lastWanted > KEEP_MS && !this.fading.has(t)) this.dropTile(t)
    }

    // Requests: coarse levels first so there's always something to show, then nearest.
    let inflight = 0
    for (const t of this.tiles.values()) if (t.state === 'loading') inflight++
    this.planWants = want.length
    want.sort((a, b) => b.level - a.level || a.d - b.d)
    this.wants = want
    this.wantAt = 0
    this.inflight = inflight
    this.requestMore(now)
  }

  /** Starts the next wanted tiles, up to the limit. Runs whenever one finishes. */
  private requestMore(now: number) {
    while (this.inflight < MAX_INFLIGHT && this.wantAt < this.wants.length) {
      const w = this.wants[this.wantAt++]
      const t = this.tiles.get(key(w.level, w.x, w.z))
      if (t?.state === 'loading') continue
      this.loadTile(w.level, w.x, w.z, now)
      this.inflight++
    }
  }

  // ---------------------------------------------------------------- tiles

  private loadTile(level: number, x: number, z: number, now: number) {
    const gen = this.generation
    const k2 = key(level, x, z)
    const url =
      level === 0
        ? `/maps/${this.map!.id}/hires/${Math.floor(x / 8)}_${Math.floor(z / 8)}/${x}_${z}.tmh`
        : `/maps/${this.map!.id}/lod/${level}/${x}_${z}.tmh`
    const existing = this.tiles.get(k2)
    const { id, done } = this.workers.request(url, !!existing?.stale)
    const tile: Tile = existing ?? {
      key: k2,
      level,
      x,
      z,
      state: 'loading',
      retryAt: 0,
      req: id,
      mesh: null,
      stale: false,
      tops: null,
      lastWanted: now,
      drawn: false,
      fadeStart: 0,
      distance: 0,
    }
    tile.req = id
    tile.stale = false
    if (!existing) this.tiles.set(k2, tile)
    tile.state = 'loading'
    const current = () => gen === this.generation && this.tiles.get(k2) === tile && tile.req === id
    done.then((res) => {
      if (!current()) return
      // Refill the request slot right away, rather than waiting for a frame.
      this.inflight--
      this.requestMore(performance.now())
      this.lastPlan = 0
      this.wake()
      if (res.kind === 'error') {
        console.warn(`tile ${level}:${x},${z}: ${res.message}`)
        // Keep an old mesh; with nothing to show, retry later instead of
        // treating the tile as empty and leaving a hole.
        if (tile.mesh) tile.state = 'ready'
        else {
          tile.state = 'failed'
          tile.retryAt = performance.now() + 5000
        }
        return
      }
      if (res.kind !== 'tile') return
      this.downloaded += res.bytes
      if (!res.tile) {
        this.fading.delete(tile)
        if (tile.mesh) this.renderer.release(tile.mesh)
        tile.mesh = null
        tile.tops = null
        tile.state = 'empty'
        this.dirty = true
        return
      }
      const decoded = res.tile
      // Each frame uploads only a bounded amount; the tile counts as loading until then.
      this.uploads.push(() => {
        if (!current()) return 0
        let mesh = this.renderer.upload(decoded, level)
        // The pools are full: make room from tiles nothing is drawing.
        while (!mesh && this.evictOne(tile)) mesh = this.renderer.upload(decoded, level)
        if (!mesh) {
          tile.state = tile.mesh ? 'ready' : 'failed'
          tile.retryAt = performance.now() + 1000
          return 0
        }
        if (tile.mesh) this.renderer.release(tile.mesh)
        tile.mesh = mesh
        tile.tops = decoded.tops
        tile.state = 'ready'
        this.lastPlan = 0
        this.dirty = true
        return decoded.quads.byteLength + decoded.tables.byteLength
      })
    })
  }

  /** Frees the longest-unwanted tile that isn't being drawn. */
  private evictOne(except: Tile): boolean {
    let oldest: Tile | null = null
    for (const t of this.tiles.values()) {
      if (t === except || t.drawn || !t.mesh || this.fading.has(t)) continue
      if (!oldest || t.lastWanted < oldest.lastWanted) oldest = t
    }
    if (!oldest) return false
    this.dropTile(oldest)
    return true
  }

  private dropTile(tile: Tile) {
    if (tile.state === 'loading') this.workers.abort(tile.req)
    this.fading.delete(tile)
    if (tile.mesh) this.renderer.release(tile.mesh)
    tile.mesh = null
    const i = this.visible.indexOf(tile)
    if (i >= 0) this.visible.splice(i, 1)
    this.tiles.delete(tile.key)
    this.dirty = true
  }

  /** Terrain height under a point, from the finest loaded tile there. */
  heightAt(x: number, z: number): number | null {
    for (let level = 0; level <= (this.manifest?.lodLevels ?? 0); level++) {
      const size = TILE << level
      const tx = Math.floor(x / size)
      const tz = Math.floor(z / size)
      const t = this.tiles.get(key(level, tx, tz))
      if (!t?.tops) continue
      const cx = Math.floor(((x - tx * size) / size) * 64)
      const cz = Math.floor(((z - tz * size) / size) * 64)
      const h = t.tops[cz * 64 + cx]
      if (h !== NO_TOP) return h * (1 << level)
    }
    return null
  }

  // ---------------------------------------------------------------- frame

  private sleepTick = () => {
    this.plan(performance.now())
    if (this.planWants || this.dirty) this.wake()
    else this.sleepTimer = window.setTimeout(this.sleepTick, SLEEP_MS)
  }

  private wake = () => {
    clearTimeout(this.sleepTimer)
    this.idleFrames = 0
    if (!this.asleep) return
    this.asleep = false
    this.raf = requestAnimationFrame(this.frame)
  }

  private frame = (now: number) => {
    this.raf = requestAnimationFrame(this.frame)
    const moved = this.controls.update()

    // Keep the orbit target on the terrain surface.
    const t = this.controls.target
    const h = this.heightAt(t.x, t.z)
    if (h !== null && Math.abs(h - t.y) > 0.5) {
      const dy = (h - t.y) * 0.12
      t.y += dy
      this.camera.position.y += dy
      this.dirty = true
    }

    // Bounded GPU uploads per frame, so a burst of arrivals doesn't stall a frame.
    for (let budget = UPLOAD_BYTES_PER_FRAME; budget > 0 && this.uploads.length; ) {
      budget -= this.uploads.shift()!()
    }

    // A finished dissolve changes which tiles are visible, so this comes before planning.
    for (const tile of this.fading) {
      if (now - tile.fadeStart >= FADE_MS) {
        this.fading.delete(tile)
        tile.fadeStart = 0
        if (!tile.drawn) this.lastPlan = 0
      }
      this.dirty = true
    }

    if (now - this.lastPlan > 150) {
      this.lastPlan = now
      this.plan(now)
    }

    const drawing = moved || this.dirty || this.bench
    if (drawing) {
      const t0 = performance.now()
      const q = this.gpuTimer?.begin()
      this.drawFrame(now)
      if (q) this.gpuTimer!.end(q)
      this.frameMsAcc += performance.now() - t0
      this.frames++
      this.dirty = false
      this.onCamera({ x: t.x, y: t.y, z: t.z })
    }

    if (now - this.lastStats > 1000) {
      const levels = Array.from({ length: (this.manifest?.lodLevels ?? 0) + 1 }, () => 0)
      let pending = 0
      for (const tile of this.tiles.values()) {
        if (tile.drawn) levels[tile.level]++
        if (tile.state === 'loading') pending++
      }
      this.onStats({
        fps: (this.frames * 1000) / (now - this.lastStats),
        frameMs: this.frames ? this.frameMsAcc / this.frames : 0,
        gpuMs: this.gpuTimer?.p95() ?? null,
        calls: this.drawCalls,
        triangles: this.triangles,
        levels,
        pending,
        downloadedMB: this.downloaded / 1e6,
        gpuMB: this.renderer.gpuBytes / 1e6,
      })
      this.frames = 0
      this.frameMsAcc = 0
      this.lastStats = now
    }

    // With nothing to do, stop asking for frames, so an idle map costs no power.
    // A slow timer keeps planning, for retries and expiring tiles.
    if (!drawing && !this.uploads.length && !this.fading.size && !this.planWants) {
      if (++this.idleFrames > IDLE_FRAMES) {
        this.asleep = true
        cancelAnimationFrame(this.raf)
        this.sleepTimer = window.setTimeout(this.sleepTick, SLEEP_MS)
      }
    } else this.idleFrames = 0
  }

  /** Builds this frame's draw list, nearest tiles first, and draws it. */
  private drawFrame(now: number) {
    const r = this.renderer
    const cam = this.camera.position
    const camera = [cam.x, cam.y, cam.z]
    const dissolve = (t: Tile) => {
      if (!t.fadeStart) return 1
      const k = Math.min(1, (now - t.fadeStart) / FADE_MS)
      return t.drawn ? k : -k
    }
    r.begin()
    r.startPass(Pass.Opaque)
    for (const t of this.visible) if (!t.fadeStart) r.add(Pass.Opaque, t.mesh!, camera, false, 1)
    r.startPass(Pass.OpaqueFade)
    for (const t of this.visible) if (t.fadeStart) r.add(Pass.OpaqueFade, t.mesh!, camera, false, dissolve(t))
    r.startPass(Pass.Cutout)
    for (const t of this.visible) {
      // Leaf faces that touch other leaves only show up close; past LEAF_NEAR the
      // outer faces already cover them.
      const size = TILE << t.level
      const dx = Math.max(t.x * size - cam.x, 0, cam.x - (t.x + 1) * size)
      const dz = Math.max(t.z * size - cam.z, 0, cam.z - (t.z + 1) * size)
      const near = Math.hypot(dx, dz)
      r.add(Pass.Cutout, t.mesh!, camera, near < LEAF_NEAR, dissolve(t), near < PLANT_NEAR)
    }
    r.startPass(Pass.Translucent)
    for (const t of this.visible) r.add(Pass.Translucent, t.mesh!, camera, false, dissolve(t))

    // The view without its translation; positions reach the shader camera-relative.
    this.camera.updateMatrixWorld()
    const m = this.matrix.copy(this.camera.matrixWorldInverse)
    m.elements[12] = m.elements[13] = m.elements[14] = 0
    m.premultiply(this.camera.projectionMatrix)
    this.viewProj.set(m.elements)
    const d = cam.distanceTo(this.controls.target)
    const { calls, triangles } = r.draw(this.viewProj, camera, BG, [d * 4 + 3000, d * 9 + 12000])
    this.drawCalls = calls
    this.triangles = triangles
  }

  private clearTiles() {
    for (const t of [...this.tiles.values()]) this.dropTile(t)
    this.tiles.clear()
    this.visible = []
    this.uploads = []
  }
}

/** GPU frame timing with EXT_disjoint_timer_query_webgl2; results arrive a few frames late. */
class GpuTimer {
  private pending: WebGLQuery[] = []
  private samples: number[] = []

  private gl: WebGL2RenderingContext
  private ext: { TIME_ELAPSED_EXT: number; GPU_DISJOINT_EXT: number }

  private constructor(gl: WebGL2RenderingContext, ext: { TIME_ELAPSED_EXT: number; GPU_DISJOINT_EXT: number }) {
    this.gl = gl
    this.ext = ext
  }

  static create(gl: WebGL2RenderingContext): GpuTimer | null {
    const ext = gl.getExtension('EXT_disjoint_timer_query_webgl2')
    return ext ? new GpuTimer(gl, ext) : null
  }

  begin(): WebGLQuery | null {
    this.collect()
    const q = this.gl.createQuery()
    if (q) this.gl.beginQuery(this.ext.TIME_ELAPSED_EXT, q)
    return q
  }

  end(q: WebGLQuery) {
    this.gl.endQuery(this.ext.TIME_ELAPSED_EXT)
    this.pending.push(q)
  }

  private collect() {
    const gl = this.gl
    // Reading the disjoint flag clears it, so read it once: when set, every
    // pending measurement is suspect and gets dropped.
    if (gl.getParameter(this.ext.GPU_DISJOINT_EXT)) {
      this.pending.forEach((q) => gl.deleteQuery(q))
      this.pending = []
      return
    }
    while (this.pending.length && gl.getQueryParameter(this.pending[0], gl.QUERY_RESULT_AVAILABLE)) {
      const q = this.pending.shift()!
      this.samples.push(gl.getQueryParameter(q, gl.QUERY_RESULT) / 1e6)
      gl.deleteQuery(q)
    }
  }

  /** p95 of the samples since the last call, in ms. */
  p95(): number | null {
    if (!this.samples.length) return null
    const s = this.samples.sort((a, b) => a - b)
    const v = s[Math.min(s.length - 1, Math.floor(s.length * 0.95))]
    this.samples = []
    return v
  }
}
