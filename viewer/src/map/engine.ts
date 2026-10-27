import * as THREE from 'three'
import { MapControls } from 'three/examples/jsm/controls/MapControls.js'
import { type DecodedTile, GROUPS, LEAF_INTERIOR, QUAD_ROW, TABLE_ROW } from './tile'
import type { Request, Response } from './worker'
import { hiresFragment, hiresVertex } from './shaders'

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
  /** LOD tiles as [level, x, z]; level k uses voxels of 2^k blocks. */
  lod: [number, number, number][]
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

// Colors pass through untouched: block textures and vertex shading are in
// display space, as in Minecraft.
THREE.ColorManagement.enabled = false
const BG = new THREE.Color('#0b1416')
const TILE = 64
const MAX_INFLIGHT = 8
/** Tiles swap with a dissolve: the incoming one draws the pixels the outgoing one drops. */
const FADE_MS = 200
const UPLOAD_BYTES_PER_FRAME = 12e6
const LEAF_NEAR = 150
/** Tiles not wanted for this long are dropped. */
const KEEP_MS = 10_000

interface Tile {
  level: number
  x: number
  z: number
  /** 'failed' is a load error with nothing to show; it's retried at `retryAt`. */
  state: 'loading' | 'ready' | 'empty' | 'failed'
  retryAt: number
  req: number
  group: THREE.Group | null
  gpuBytes: number
  stale: boolean
  /** Top surface y per column (64×64) in world blocks, NaN where empty. */
  tops: Float32Array | null
  lastWanted: number
  drawn: boolean
  /** Start of the current dissolve in or out, 0 when settled. */
  fadeStart: number
}

const key = (level: number, x: number, z: number) => `${level}:${x},${z}`

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

  request(url: string): { id: number; done: Promise<Response> } {
    const id = ++this.seq
    const worker = this.workers[this.next++ % this.workers.length]
    const done = new Promise<Response>((resolve) => this.pending.set(id, { resolve, worker }))
    worker.postMessage({ kind: 'tile', id, url } satisfies Request)
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
  readonly renderer: THREE.WebGLRenderer
  readonly camera: THREE.PerspectiveCamera
  readonly controls: MapControls
  private scene = new THREE.Scene()
  private workers = new Workers(2)
  private events: EventSource | null = null

  private map: MapInfo | null = null
  private manifest: Manifest | null = null
  /** LOD tiles that have a mesh, by key. */
  private lodSet = new Set<string>()
  /**
   * Every LOD node over some hires tile, by key, mesh or not. A sparse area
   * can downsample to nothing at a coarse level while its hires tiles still
   * have geometry; those nodes have no mesh but still refine.
   */
  private nodes = new Set<string>()
  private masks = new Map<string, [number, number]>()
  private generation = 0
  private blockTex: THREE.DataArrayTexture | null = null
  private materials: THREE.ShaderMaterial[] = []
  private sharedIndex: THREE.BufferAttribute | null = null

  private tiles = new Map<string, Tile>()
  /** Tiles mid-dissolve. */
  private fading = new Set<Tile>()

  /** Largest allowed on-screen size of a voxel, in CSS pixels, before refining. */
  detail = 1
  /** Tiles waiting for their GPU upload. Each returns the bytes it uploads. */
  private uploads: (() => number)[] = []
  private justAdded: THREE.Group[] = []
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
  private onStats: (s: Stats) => void
  private onCamera: (c: CameraState) => void
  private hashTimer = 0
  private gpuTimer: GpuTimer | null = null

  private canvas: HTMLCanvasElement

  constructor(
    canvas: HTMLCanvasElement,
    opts: { onStats: (s: Stats) => void; onCamera: (c: CameraState) => void; bench?: boolean },
  ) {
    this.canvas = canvas
    this.onStats = opts.onStats
    this.onCamera = opts.onCamera
    this.bench = opts.bench ?? false
    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true, powerPreference: 'high-performance' })
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
    this.renderer.outputColorSpace = THREE.LinearSRGBColorSpace
    this.renderer.setClearColor(BG)
    this.renderer.info.autoReset = true

    this.camera = new THREE.PerspectiveCamera(55, 1, 2, 120000)
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
    })

    if (this.bench) this.gpuTimer = GpuTimer.create(this.renderer.getContext() as WebGL2RenderingContext)
    this.resize()
    window.addEventListener('resize', this.resize)
    this.raf = requestAnimationFrame(this.frame)
  }

  dispose() {
    cancelAnimationFrame(this.raf)
    window.removeEventListener('resize', this.resize)
    this.events?.close()
    this.clearTiles()
    this.workers.dispose()
    this.controls.dispose()
    this.renderer.dispose()
  }

  private resize = () => {
    const { clientWidth: w, clientHeight: h } = this.canvas
    this.renderer.setSize(w, h, false)
    this.camera.aspect = w / Math.max(h, 1)
    this.camera.updateProjectionMatrix()
    this.dirty = true
  }

  /** Loads a map. `view` restores a camera position (e.g. from the URL). */
  async setMap(map: MapInfo, view?: { x: number; z: number; dist: number; az: number; polar: number }) {
    const gen = ++this.generation
    this.clearTiles()
    this.map = map
    // Nothing plans against the old map's manifest or textures while loading.
    this.manifest = null
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
    this.setupMaterials(texData, manifest.textures)

    if (view) this.placeCamera(view.x, view.z, view.dist, view.az, view.polar)
    else this.placeDefault()
    if (this.pendingUpdate) this.refreshManifest(gen)
    this.dirty = true
  }

  private setManifest(m: Manifest) {
    this.manifest = m
    this.lodSet = new Set((m.lod ?? []).map(([l, x, z]) => key(l, x, z)))
    this.masks = new Map((m.hires ?? []).map(([rx, rz, lo, hi]) => [`${rx},${rz}`, [lo, hi]]))
    this.nodes = new Set(this.lodSet)
    for (const [rx, rz, lo, hi] of m.hires ?? []) {
      for (let bit = 0; bit < 64; bit++) {
        if (!(((bit < 32 ? lo : hi) >>> (bit & 31)) & 1)) continue
        const tx = rx * 8 + (bit & 7)
        const tz = rz * 8 + (bit >> 3)
        for (let l = 1; l <= m.lodLevels; l++) this.nodes.add(key(l, tx >> l, tz >> l))
      }
    }
    // Drop resident tiles whose mesh the manifest no longer lists.
    for (const [k2, t] of this.tiles) {
      const listed = t.level > 0 ? this.lodSet.has(k2) : this.exists(0, t.x, t.z)
      if (!listed) this.dropTile(k2, t)
    }
    this.lastPlan = 0
  }

  /** Starts over the mapped tile closest to the middle of everything mapped. */
  private placeDefault() {
    const m = this.manifest
    // Hires tiles always exist where anything is mapped; LOD meshes may not.
    const tiles: [number, number][] = []
    for (const [rx, rz, lo, hi] of m?.hires ?? []) {
      for (let bit = 0; bit < 64; bit++) {
        if (((bit < 32 ? lo : hi) >>> (bit & 31)) & 1) tiles.push([rx * 8 + (bit & 7), rz * 8 + (bit >> 3)])
      }
    }
    if (!tiles.length) return this.placeCamera(0, 0, 520, 35, 50)
    const cx = tiles.reduce((a, [x]) => a + x, 0) / tiles.length
    const cz = tiles.reduce((a, [, z]) => a + z, 0) / tiles.length
    let best = tiles[0]
    for (const t of tiles) {
      if ((t[0] - cx) ** 2 + (t[1] - cz) ** 2 < (best[0] - cx) ** 2 + (best[1] - cz) ** 2) best = t
    }
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
  }

  get mapId() {
    return this.map?.id ?? null
  }

  goTo(v: { x: number; z: number; dist: number; az: number; polar: number }) {
    this.placeCamera(v.x, v.z, v.dist, v.az, v.polar)
    this.lastPlan = 0
    this.dirty = true
  }

  setDetail(px: number) {
    this.detail = px
    this.lastPlan = 0
    this.dirty = true
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
    const polar = Math.acos(THREE.MathUtils.clamp(off.y / dist, -1, 1))
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

  private setupMaterials(data: Uint8Array, layers: number) {
    this.blockTex?.dispose()
    this.materials.forEach((m) => m.dispose())
    const tex = new THREE.DataArrayTexture(data, 16, 16, layers)
    tex.format = THREE.RGBAFormat
    tex.type = THREE.UnsignedByteType
    tex.magFilter = THREE.NearestFilter
    tex.minFilter = THREE.NearestMipmapLinearFilter
    tex.wrapS = tex.wrapT = THREE.RepeatWrapping
    tex.generateMipmaps = true
    tex.anisotropy = Math.min(8, this.renderer.capabilities.getMaxAnisotropy())
    tex.colorSpace = THREE.NoColorSpace
    tex.needsUpdate = true
    this.blockTex = tex

    const shared = {
      uTex: { value: tex },
      uFogColor: { value: BG },
      uFogRange: { value: new THREE.Vector2(4000, 9000) },
    }
    const make = (defines: Record<string, string>, extra: THREE.ShaderMaterialParameters = {}) =>
      new THREE.ShaderMaterial({
        glslVersion: THREE.GLSL3,
        vertexShader: hiresVertex,
        fragmentShader: hiresFragment,
        uniforms: { ...shared, uAppear: { value: 1 }, uFade: { value: 0 } },
        defines,
        ...extra,
      })
    this.materials = [
      make({}),
      make({ CUTOUT: '' }),
      make({ TRANSLUCENT: '' }, { transparent: true, depthWrite: false }),
    ]
  }

  private fogRange(): THREE.Vector2 {
    const d = this.camera.position.distanceTo(this.controls.target)
    return new THREE.Vector2(d * 4 + 3000, d * 9 + 12000)
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
  }

  // ---------------------------------------------------------------- planning

  private exists(level: number, x: number, z: number): boolean {
    if (level > 0) return this.nodes.has(key(level, x, z))
    const rx = Math.floor(x / 8)
    const rz = Math.floor(z / 8)
    const mask = this.masks.get(`${rx},${rz}`)
    const bit = (z - rz * 8) * 8 + (x - rx * 8)
    return !!mask && ((mask[bit >> 5] >>> (bit & 31)) & 1) === 1
  }

  private children(level: number, x: number, z: number): [number, number, number][] {
    const out: [number, number, number][] = []
    for (const [dx, dz] of [[0, 0], [1, 0], [0, 1], [1, 1]]) {
      if (this.exists(level - 1, x * 2 + dx, z * 2 + dz)) out.push([level - 1, x * 2 + dx, z * 2 + dz])
    }
    return out
  }

  /**
   * Picks what to draw: a quadtree walk from the coarsest tiles that refines
   * wherever a voxel would cover more than `detail` pixels. A tile hands over
   * to its children only once all of them are loaded, so there are no holes.
   */
  private plan(now: number) {
    const m = this.manifest
    if (!m || !this.blockTex) return
    const cam = this.camera.position
    // Pixels per block at distance 1.
    const k = this.canvas.clientHeight / (2 * Math.tan(THREE.MathUtils.degToRad(this.camera.fov / 2)))
    const frustum = new THREE.Frustum().setFromProjectionMatrix(
      new THREE.Matrix4().multiplyMatrices(this.camera.projectionMatrix, this.camera.matrixWorldInverse),
    )
    const box = new THREE.Box3()
    const draw = new Set<Tile>()
    const want: { level: number; x: number; z: number; d: number }[] = []

    const meshed = (level: number, x: number, z: number) => level === 0 || this.lodSet.has(key(level, x, z))
    const needs = (t: Tile | undefined) => !t || t.stale || (t.state === 'failed' && now >= t.retryAt)
    // Settled: loaded at least once (an old mesh counts while it refreshes),
    // confirmed empty, or a node with no mesh to load.
    const settled = (l: number, x: number, z: number) => {
      if (!meshed(l, x, z)) return true
      const t = this.tiles.get(key(l, x, z))
      return !!t && (!!t.group || t.state === 'empty')
    }
    const visit = (level: number, x: number, z: number) => {
      const size = TILE << level
      box.min.set(x * size, m.minY, z * size)
      box.max.set((x + 1) * size, m.maxY, (z + 1) * size)
      const tile = this.tiles.get(key(level, x, z))
      // Off-screen siblings of drawn tiles stay loaded: dropping one would make
      // its parent unsettled and swap the whole group back to coarse.
      if (tile) tile.lastWanted = now
      if (!frustum.intersectsBox(box)) return
      const d = Math.max(box.distanceToPoint(cam), 1)
      if (meshed(level, x, z) && needs(tile)) want.push({ level, x, z, d })
      const kids = level > 0 && ((1 << level) * k) / d > this.detail ? this.children(level, x, z) : []
      if (kids.length) {
        if (kids.every(([l, kx, kz]) => settled(l, kx, kz))) {
          for (const [l, kx, kz] of kids) visit(l, kx, kz)
          return
        }
        for (const [l, kx, kz] of kids) {
          const t = this.tiles.get(key(l, kx, kz))
          if (t) t.lastWanted = now
          if (meshed(l, kx, kz) && needs(t)) want.push({ level: l, x: kx, z: kz, d })
        }
      }
      // Anything with a mesh draws, including an old one while it refreshes.
      if (tile?.group) draw.add(tile)
    }
    for (const n of this.nodes) {
      const [l, rest] = n.split(':')
      if (Number(l) !== m.lodLevels) continue
      const [x, z] = rest.split(',').map(Number)
      visit(m.lodLevels, x, z)
    }

    // Swap what's drawn, dissolving tiles in and out.
    for (const t of this.tiles.values()) {
      const on = draw.has(t)
      if (on !== t.drawn) {
        t.drawn = on
        t.fadeStart = now
        this.fading.add(t)
        if (on && t.group) t.group.visible = true
      }
    }

    // Drop tiles that haven't been wanted for a while.
    for (const [k2, t] of this.tiles) {
      if (!t.drawn && now - t.lastWanted > KEEP_MS && !this.fading.has(t)) this.dropTile(k2, t)
    }

    // Requests: coarse levels first so there's always something to show, then nearest.
    let inflight = 0
    for (const t of this.tiles.values()) if (t.state === 'loading') inflight++
    want.sort((a, b) => b.level - a.level || a.d - b.d)
    for (const w of want) {
      if (inflight >= MAX_INFLIGHT) break
      const t = this.tiles.get(key(w.level, w.x, w.z))
      if (t?.state === 'loading') continue
      this.loadTile(w.level, w.x, w.z, now)
      inflight++
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
    const { id, done } = this.workers.request(url)
    const existing = this.tiles.get(k2)
    const tile: Tile = existing ?? {
      level,
      x,
      z,
      state: 'loading',
      retryAt: 0,
      req: id,
      group: null,
      gpuBytes: 0,
      stale: false,
      tops: null,
      lastWanted: now,
      drawn: false,
      fadeStart: 0,
    }
    tile.req = id
    tile.stale = false
    if (!existing) this.tiles.set(k2, tile)
    tile.state = 'loading'
    done.then((res) => {
      if (gen !== this.generation || this.tiles.get(k2) !== tile || tile.req !== id) return
      // Refill the request slot on the next frame instead of the next plan tick.
      this.lastPlan = 0
      if (res.kind === 'error') {
        console.warn(`tile ${k2}: ${res.message}`)
        // Keep an old mesh; with nothing to show, retry later instead of
        // treating the tile as empty and leaving a hole.
        if (tile.group) tile.state = 'ready'
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
        if (tile.group) this.disposeGroup(tile.group)
        tile.group = null
        tile.gpuBytes = 0
        tile.tops = null
        tile.state = 'empty'
        this.dirty = true
        return
      }
      const decoded = res.tile
      const { group, bytes } = this.buildTile(decoded, level)
      // Three uploads geometry the first time it's drawn. Queue the tile so each
      // frame uploads only a bounded amount; it counts as loading until then.
      this.uploads.push(() => {
        if (gen !== this.generation || this.tiles.get(k2) !== tile || tile.req !== id) {
          this.disposeGroup(group)
          return 0
        }
        const old = tile.group
        tile.group = group
        tile.gpuBytes = bytes
        tile.tops = decoded.tops.map((v) => (Number.isNaN(v) ? v : v * (1 << level)))
        tile.state = 'ready'
        group.visible = tile.drawn
        this.setFade(tile, tile.drawn ? 1 : 0, 0)
        this.scene.add(group)
        // Three uploads geometry only for meshes it draws, so draw this one
        // once regardless of the frustum to keep its upload in this frame's budget.
        group.visible = true
        for (const c of group.children) c.frustumCulled = false
        this.justAdded.push(group)
        if (old) this.disposeGroup(old)
        this.lastPlan = 0
        this.dirty = true
        return bytes
      })
    })
  }

  private dropTile(k2: string, tile: Tile) {
    if (tile.state === 'loading') this.workers.abort(tile.req)
    this.fading.delete(tile)
    if (tile.group) this.disposeGroup(tile.group)
    this.tiles.delete(k2)
    this.dirty = true
  }

  private disposeGroup(g: THREE.Group) {
    this.scene.remove(g)
    for (const c of g.children) {
      const geo = (c as THREE.Mesh).geometry
      // The index is shared between tiles; detach it so dispose keeps it.
      geo.setIndex(null)
      geo.dispose()
    }
    for (const m of g.userData.materials as THREE.Material[]) m.dispose()
    for (const t of g.userData.textures as THREE.Texture[]) t.dispose()
  }

  /** Dissolve state: draw pixels whose noise is below `appear` and not below `fade`. */
  private setFade(tile: Tile, appear: number, fade: number) {
    for (const m of (tile.group?.userData.materials ?? []) as THREE.ShaderMaterial[]) {
      m.uniforms.uAppear.value = appear
      m.uniforms.uFade.value = fade
    }
  }

  private indexFor(quads: number): THREE.BufferAttribute {
    if (!this.sharedIndex || this.sharedIndex.count < quads * 6) {
      let cap = 1 << 16
      while (cap < quads) cap *= 2
      const idx = new Uint32Array(cap * 6)
      for (let q = 0, v = 0, i = 0; q < cap; q++, v += 4) {
        idx[i++] = v
        idx[i++] = v + 1
        idx[i++] = v + 2
        idx[i++] = v
        idx[i++] = v + 2
        idx[i++] = v + 3
      }
      this.sharedIndex = new THREE.BufferAttribute(idx, 1)
    }
    return this.sharedIndex
  }

  private buildTile(t: DecodedTile, level: number): { group: THREE.Group; bytes: number } {
    const dataTex = (
      data: Uint32Array | Uint16Array | Uint8Array,
      w: number,
      h: number,
      format: THREE.PixelFormat,
      type: THREE.TextureDataType,
      internal: THREE.PixelFormatGPU | null,
    ) => {
      const tex = new THREE.DataTexture(data, w, h, format, type)
      tex.internalFormat = internal
      tex.magFilter = tex.minFilter = THREE.NearestFilter
      tex.generateMipmaps = false
      tex.flipY = false
      tex.colorSpace = THREE.NoColorSpace
      tex.needsUpdate = true
      return tex
    }
    const quads = dataTex(t.quads, QUAD_ROW, t.quadRows, THREE.RGIntegerFormat, THREE.UnsignedIntType, 'RG32UI')
    const templates = dataTex(t.templates, TABLE_ROW, t.templateRows, THREE.RGBAIntegerFormat, THREE.UnsignedShortType, 'RGBA16UI')
    const colors = dataTex(t.colors, TABLE_ROW, t.colorRows, THREE.RGBAFormat, THREE.UnsignedByteType, null)
    // One material per pass for this tile; the program and shared uniforms are reused.
    const materials = this.materials.map((base) => {
      const m = base.clone()
      m.uniforms = {
        ...base.uniforms,
        uQuads: { value: quads },
        uTpl: { value: templates },
        uPal: { value: colors },
        uAppear: { value: 1 },
        uFade: { value: 0 },
      }
      return m
    })
    const [minQ, maxQ] = [t.minQ, t.maxQ]
    const sphere = new THREE.Sphere(
      new THREE.Vector3(33 * 64, (minQ + maxQ) / 2, 33 * 64),
      Math.hypot(33 * 64, 33 * 64, (maxQ - minQ) / 2),
    )
    // A level-k tile is in voxels of 2^k blocks: scale its anchor space to blocks.
    const s = 1 << level
    const group = new THREE.Group()
    let start = 0
    for (let g = 0; g < GROUPS; g++) {
      const n = t.counts[g]
      if (n > 0) {
        const geo = new THREE.BufferGeometry()
        geo.setIndex(this.indexFor(start + n))
        geo.setDrawRange(start * 6, n * 6)
        geo.boundingSphere = sphere
        // Leaf interiors draw with the cutout pass; translucent is the last group.
        const pass = g === LEAF_INTERIOR ? 1 : Math.min(g, 2)
        const mesh = new THREE.Mesh(geo, materials[pass])
        mesh.position.set(t.origin[0] * s, t.origin[1] * s, t.origin[2] * s)
        mesh.scale.setScalar(s / 64)
        mesh.renderOrder = pass
        mesh.matrixAutoUpdate = false
        mesh.updateMatrix()
        if (g === LEAF_INTERIOR) group.userData.leafInterior = mesh
        group.add(mesh)
      }
      start += n
    }
    group.userData.materials = materials
    group.userData.textures = [quads, templates, colors]
    group.matrixAutoUpdate = false
    const bytes = t.quads.byteLength + t.templates.byteLength + t.colors.byteLength
    return { group, bytes }
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
      if (!Number.isNaN(h)) return h
    }
    return null
  }

  // ---------------------------------------------------------------- frame

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

    if (now - this.lastPlan > 150) {
      this.lastPlan = now
      this.plan(now)
    }

    // Dissolves: incoming tiles draw the pixels outgoing ones drop.
    for (const tile of this.fading) {
      const k = Math.min(1, (now - tile.fadeStart) / FADE_MS)
      if (tile.drawn) this.setFade(tile, k, 0)
      else this.setFade(tile, 1, k)
      if (k >= 1) {
        this.fading.delete(tile)
        if (!tile.drawn && tile.group) tile.group.visible = false
      }
      this.dirty = true
    }

    if (moved || this.dirty || this.bench) {
      this.gateLeaves()
      const fog = this.fogRange()
      for (const m of this.materials) (m.uniforms.uFogRange.value as THREE.Vector2).copy(fog)
      const t0 = performance.now()
      const q = this.gpuTimer?.begin()
      this.renderer.render(this.scene, this.camera)
      for (const g of this.justAdded) {
        for (const c of g.children) c.frustumCulled = true
        const tile = [...this.tiles.values()].find((x) => x.group === g)
        g.visible = !!tile && (tile.drawn || this.fading.has(tile))
      }
      this.justAdded = []
      if (q) this.gpuTimer!.end(q)
      this.frameMsAcc += performance.now() - t0
      this.frames++
      this.dirty = false
      this.onCamera({ x: t.x, y: t.y, z: t.z })
    }

    if (now - this.lastStats > 1000) {
      const info = this.renderer.info.render
      let gpu = this.blockTex && this.manifest ? this.manifest.textures * 1024 * 1.34 : 0
      const levels = Array.from({ length: (this.manifest?.lodLevels ?? 0) + 1 }, () => 0)
      let pending = 0
      for (const tile of this.tiles.values()) {
        gpu += tile.gpuBytes
        if (tile.drawn) levels[tile.level]++
        if (tile.state === 'loading') pending++
      }
      this.onStats({
        fps: (this.frames * 1000) / (now - this.lastStats),
        frameMs: this.frames ? this.frameMsAcc / this.frames : 0,
        gpuMs: this.gpuTimer?.p95() ?? null,
        calls: info.calls,
        triangles: info.triangles,
        levels,
        pending,
        downloadedMB: this.downloaded / 1e6,
        gpuMB: gpu / 1e6,
      })
      this.frames = 0
      this.frameMsAcc = 0
      this.lastStats = now
    }
  }

  /**
   * Leaf faces that touch other leaves only show up close; past LEAF_NEAR the
   * outer faces already cover them, so they're skipped.
   */
  private gateLeaves() {
    const cam = this.camera.position
    for (const tile of this.tiles.values()) {
      const mesh = tile.group?.userData.leafInterior as THREE.Mesh | undefined
      if (!mesh) continue
      const size = TILE << tile.level
      const dx = Math.max(tile.x * size - cam.x, 0, cam.x - (tile.x + 1) * size)
      const dz = Math.max(tile.z * size - cam.z, 0, cam.z - (tile.z + 1) * size)
      mesh.visible = Math.hypot(dx, dz) < LEAF_NEAR
    }
  }

  private clearTiles() {
    for (const [k2, t] of this.tiles) this.dropTile(k2, t)
    this.tiles.clear()
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
