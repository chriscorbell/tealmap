import * as THREE from 'three'
import { MapControls } from 'three/examples/jsm/controls/MapControls.js'
import { type DecodedTile, GROUPS, LEAF_INTERIOR, QUAD_ROW, TABLE_ROW } from './tile'
import type { Request, Response } from './worker'
import { hiresFragment, hiresVertex, lowresFragment, lowresVertex } from './shaders'

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
  lowres: [number, number][]
  lowres2: [number, number][]
  /** Per region: [rx, rz, low 32 bits, high 32 bits] of the 8×8 hires tile mask. */
  hires: [number, number, number, number][]
}

export interface Stats {
  fps: number
  frameMs: number
  /** GPU time per frame (p95), when the timer extension is available in bench mode. */
  gpuMs: number | null
  calls: number
  triangles: number
  hires: number
  lowres: number
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
const REGION = 512
const MAX_HIRES_INFLIGHT = 6
const MAX_LOWRES_INFLIGHT = 4
const FADE_MS = 450
/** Hires tiles dissolve in over the lowres, which stays drawn until they finish. */
const HIRES_FADE_MS = 200
const UPLOAD_BYTES_PER_FRAME = 12e6
const LEAF_NEAR = 150

interface HiresTile {
  tx: number
  tz: number
  state: 'loading' | 'ready' | 'empty'
  req: number
  group: THREE.Group | null
  gpuBytes: number
  stale: boolean
}

interface LowresTile {
  x: number
  z: number
  level: 1 | 2
  state: 'loading' | 'ready' | 'empty'
  req: number
  mesh: THREE.Mesh | null
  material: THREE.ShaderMaterial | null
  texture: THREE.DataTexture | null
  /** Max height per 8×8 pixels, raw encoded heights. */
  coarse: Uint16Array | null
  /** Bits of finer tiles currently drawn over this tile. */
  cover: [number, number]
  born: number
  stale: boolean
}

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

  request(kind: 'hires' | 'lowres', url: string): { id: number; done: Promise<Response> } {
    const id = ++this.seq
    const worker = this.workers[this.next++ % this.workers.length]
    const done = new Promise<Response>((resolve) => this.pending.set(id, { resolve, worker }))
    worker.postMessage({ kind, id, url } satisfies Request)
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
  private generation = 0
  private blockTex: THREE.DataArrayTexture | null = null
  private materials: THREE.ShaderMaterial[] = []
  private sharedIndex: THREE.BufferAttribute | null = null
  private lowresGrid: THREE.BufferGeometry

  private hires = new Map<string, HiresTile>()
  private lowres = new Map<string, LowresTile>()
  /** Hires tiles dissolving in, with their start times. */
  private fading = new Map<HiresTile, number>()

  hiresDistance = 192
  /** Tiles waiting for their GPU upload, closest first. Each returns the bytes it uploads. */
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
    this.scene.fog = null

    this.camera = new THREE.PerspectiveCamera(55, 1, 2, 60000)
    this.controls = new MapControls(this.camera, canvas)
    this.controls.enableDamping = true
    this.controls.dampingFactor = 0.12
    this.controls.screenSpacePanning = false
    this.controls.maxPolarAngle = 1.35
    this.controls.minDistance = 6
    this.controls.maxDistance = 12000
    this.controls.zoomSpeed = 1.3
    this.controls.zoomToCursor = true
    this.controls.addEventListener('start', () => {
      this.autoView = false
    })
    this.controls.addEventListener('change', () => {
      this.dirty = true
      this.scheduleHash()
    })

    if (this.bench) this.gpuTimer = GpuTimer.create(this.renderer.getContext() as WebGL2RenderingContext)
    this.lowresGrid = makeGrid(128, REGION)
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
    this.manifest = manifest
    this.setupMaterials(texData, manifest.textures)

    if (view) this.placeCamera(view.x, view.z, view.dist, view.az, view.polar)
    else this.placeDefault()
    if (this.pendingUpdate) this.refreshManifest(gen)
    this.dirty = true
  }

  /** Starts over the region closest to the middle of the mapped area. */
  private placeDefault() {
    const tiles = this.manifest?.lowres ?? []
    const n = Math.max(tiles.length, 1)
    const cx = tiles.reduce((a, [x]) => a + x, 0) / n
    const cz = tiles.reduce((a, [, z]) => a + z, 0) / n
    const mid = [...tiles].sort((a, b) => (a[0] - cx) ** 2 + (a[1] - cz) ** 2 - ((b[0] - cx) ** 2 + (b[1] - cz) ** 2))[0]
    const [x, z] = mid ? [mid[0] * REGION + REGION / 2, mid[1] * REGION + REGION / 2] : [0, 0]
    this.placeCamera(x, z, 520, 35, 50)
  }

  private async refreshManifest(gen: number) {
    this.pendingUpdate = false
    const m: Manifest = await (await fetch(`/maps/${this.map!.id}/map.json`, { cache: 'no-cache' })).json()
    if (gen !== this.generation || !this.manifest) return
    const wasEmpty = this.manifest.lowres.length === 0
    this.manifest.lowres = m.lowres
    this.manifest.lowres2 = m.lowres2
    this.manifest.hires = m.hires
    // A map opened before its first render finished: frame it once tiles exist.
    if (wasEmpty && m.lowres.length && this.autoView) this.placeDefault()
    this.lastPlan = 0
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

  /** Debug summary of tile states. */
  debug() {
    const count = (m: Map<string, { state: string }>) =>
      [...m.values()].reduce<Record<string, number>>((a, t) => ((a[t.state] = (a[t.state] ?? 0) + 1), a), {})
    return { hires: count(this.hires), lowres: count(this.lowres), target: this.controls.target.toArray() }
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

  setHiresDistance(d: number) {
    this.hiresDistance = d
    this.lastPlan = 0
    this.dirty = true
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
        uniforms: { ...shared, uAppear: { value: 1 } },
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
    return new THREE.Vector2(d * 2.2 + 900, d * 5 + 3200)
  }

  private listen() {
    this.events?.close()
    const es = new EventSource('/api/events')
    es.addEventListener('update', (e) => {
      const msg = JSON.parse((e as MessageEvent).data) as {
        map: string
        update: { hires: [number, number][]; lowres: [number, number][]; lowres2: [number, number][] }
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
      for (const t of this.hires.values()) t.stale = true
      for (const t of this.lowres.values()) t.stale = true
      this.refreshManifest(this.generation)
    }
    this.events = es
  }

  private async applyUpdate(up: { hires: [number, number][]; lowres: [number, number][]; lowres2: [number, number][] }) {
    if (!this.manifest) {
      // Still loading: refresh the manifest once it's in.
      this.pendingUpdate = true
      return
    }
    // Hires tiles can appear or vanish, which changes the masks in the manifest.
    if (up.hires.length || up.lowres.length || up.lowres2.length) {
      await this.refreshManifest(this.generation)
    }
    for (const [x, z] of up.hires) {
      const t = this.hires.get(`${x},${z}`)
      if (t) t.stale = true
    }
    for (const [x, z] of up.lowres) {
      const t = this.lowres.get(`1:${x},${z}`)
      if (t) t.stale = true
    }
    for (const [x, z] of up.lowres2) {
      const t = this.lowres.get(`2:${x},${z}`)
      if (t) t.stale = true
    }
    this.lastPlan = 0
    this.dirty = true
  }

  // ---------------------------------------------------------------- planning

  private plan(now: number) {
    const m = this.manifest
    if (!m || !this.blockTex) return
    const t = this.controls.target
    const dist = this.camera.position.distanceTo(t)
    const hiresR = dist > 1400 ? 0 : this.hiresDistance
    // Level 2 covers the far field, so level 1 only needs to reach a little past the view center.
    const l1R = THREE.MathUtils.clamp(dist * 2 + 500, 800, 4000)

    const rectDist = (x0: number, z0: number, size: number) => {
      const dx = Math.max(x0 - t.x, 0, t.x - (x0 + size))
      const dz = Math.max(z0 - t.z, 0, t.z - (z0 + size))
      return Math.hypot(dx, dz)
    }

    // Level 2 everywhere, level 1 within range.
    const wantLow = new Map<string, { x: number; z: number; level: 1 | 2; d: number }>()
    for (const [x, z] of m.lowres2) wantLow.set(`2:${x},${z}`, { x, z, level: 2, d: rectDist(x * 4096, z * 4096, 4096) })
    for (const [x, z] of m.lowres) {
      const d = rectDist(x * REGION, z * REGION, REGION)
      if (d <= l1R) wantLow.set(`1:${x},${z}`, { x, z, level: 1, d })
    }

    // Hires tiles the manifest says exist; no need to wait for lowres data.
    const wantHi = new Map<string, { tx: number; tz: number; d: number }>()
    if (hiresR > 0) {
      for (const [rx, rz, lo, hi] of m.hires ?? []) {
        if (rectDist(rx * REGION, rz * REGION, REGION) > hiresR) continue
        for (let bit = 0; bit < 64; bit++) {
          if (!(((bit < 32 ? lo : hi) >>> (bit & 31)) & 1)) continue
          const tx = rx * 8 + (bit & 7)
          const tz = rz * 8 + (bit >> 3)
          const d = rectDist(tx * TILE, tz * TILE, TILE)
          if (d <= hiresR) wantHi.set(`${tx},${tz}`, { tx, tz, d })
        }
      }
    }

    // Unload what's no longer wanted (with some slack to avoid thrashing),
    // and anything in a region the manifest no longer lists.
    const regions = new Set(m.lowres.map(([x, z]) => `${x},${z}`))
    const masks = new Map((m.hires ?? []).map(([rx, rz, lo, hi]) => [`${rx},${rz}`, [lo, hi]]))
    for (const [key, tile] of this.hires) {
      const rx = Math.floor(tile.tx / 8)
      const rz = Math.floor(tile.tz / 8)
      const bit = (tile.tz - rz * 8) * 8 + (tile.tx - rx * 8)
      const mask = masks.get(`${rx},${rz}`)
      const listed = !!mask && ((mask[bit >> 5] >>> (bit & 31)) & 1) === 1
      const near = wantHi.has(key) || rectDist(tile.tx * TILE, tile.tz * TILE, TILE) <= hiresR * 1.2 + 32
      if (!listed || !near) this.dropHires(key, tile)
    }
    for (const [key, tile] of this.lowres) {
      const keep =
        tile.level === 2
          ? wantLow.has(key)
          : regions.has(`${tile.x},${tile.z}`) && rectDist(tile.x * REGION, tile.z * REGION, REGION) <= l1R * 1.25
      if (!keep) this.dropLowres(key, tile)
    }

    // Frustum-aware priority: visible tiles first, then by distance.
    const frustum = new THREE.Frustum().setFromProjectionMatrix(
      new THREE.Matrix4().multiplyMatrices(this.camera.projectionMatrix, this.camera.matrixWorldInverse),
    )
    const box = new THREE.Box3()
    const visible = (x0: number, z0: number, size: number) =>
      frustum.intersectsBox(box.set(new THREE.Vector3(x0, m.minY, z0), new THREE.Vector3(x0 + size, m.maxY, z0 + size)))

    let lowInflight = [...this.lowres.values()].filter((l) => l.state === 'loading').length
    const lowQueue = [...wantLow.entries()]
      .filter(([k]) => {
        const cur = this.lowres.get(k)
        return !cur || (cur.stale && cur.state !== 'loading')
      })
      .map(([k, v]) => ({ k, ...v, score: v.d - (visible(v.x * (v.level === 1 ? REGION : 4096), v.z * (v.level === 1 ? REGION : 4096), v.level === 1 ? REGION : 4096) ? 1e6 : 0) - (v.level === 2 ? 2e6 : 0) }))
      .sort((a, b) => a.score - b.score)
    for (const q of lowQueue) {
      if (lowInflight >= MAX_LOWRES_INFLIGHT) break
      this.loadLowres(q.k, q.x, q.z, q.level, now)
      lowInflight++
    }

    let hiInflight = [...this.hires.values()].filter((h) => h.state === 'loading').length
    const hiQueue = [...wantHi.entries()]
      .filter(([k]) => {
        const cur = this.hires.get(k)
        return !cur || (cur.stale && cur.state !== 'loading')
      })
      .map(([k, v]) => ({ k, ...v, score: v.d - (visible(v.tx * TILE, v.tz * TILE, TILE) ? 1e6 : 0) }))
      .sort((a, b) => a.score - b.score)
    for (const q of hiQueue) {
      if (hiInflight >= MAX_HIRES_INFLIGHT) break
      this.loadHires(q.k, q.tx, q.tz)
      hiInflight++
    }
  }

  // ---------------------------------------------------------------- hires

  private loadHires(key: string, tx: number, tz: number) {
    const gen = this.generation
    const existing = this.hires.get(key)
    const rx = Math.floor(tx / 8)
    const rz = Math.floor(tz / 8)
    const url = `/maps/${this.map!.id}/hires/${rx}_${rz}/${tx}_${tz}.tmh`
    const { id, done } = this.workers.request('hires', url)
    const tile: HiresTile = existing ?? { tx, tz, state: 'loading', req: id, group: null, gpuBytes: 0, stale: false }
    tile.req = id
    tile.stale = false
    if (!existing) this.hires.set(key, tile)
    const wasReady = tile.state === 'ready'
    tile.state = 'loading'
    done.then((res) => {
      if (gen !== this.generation || this.hires.get(key) !== tile || tile.req !== id) return
      // Refill the request slot on the next frame instead of the next plan tick.
      this.lastPlan = 0
      if (res.kind === 'error') {
        console.warn(`tile ${key}: ${res.message}`)
        tile.state = wasReady ? 'ready' : 'empty'
        return
      }
      if (res.kind !== 'hires') return
      this.downloaded += res.bytes
      if (!res.tile) {
        this.fading.delete(tile)
        if (tile.group) this.disposeGroup(tile.group)
        tile.group = null
        tile.gpuBytes = 0
        tile.state = 'empty'
        this.setCover(tx, tz, false)
        this.dirty = true
        return
      }
      const { group, bytes } = this.buildHires(res.tile)
      // Three uploads geometry the first time it's drawn. Queue the tile so each
      // frame uploads only a bounded amount; it counts as loading until then.
      this.uploads.push(() => {
        if (gen !== this.generation || this.hires.get(key) !== tile || tile.req !== id) {
          this.disposeGroup(group)
          return 0
        }
        const old = tile.group
        this.fading.delete(tile)
        tile.group = group
        tile.gpuBytes = bytes
        tile.state = 'ready'
        this.scene.add(group)
        // Three uploads geometry only for meshes it draws, so draw this one
        // once regardless of the frustum to keep its upload in this frame's budget.
        for (const c of group.children) c.frustumCulled = false
        this.justAdded.push(group)
        if (old) {
          // Reloads swap in place.
          this.disposeGroup(old)
          this.setCover(tx, tz, true)
        } else {
          // A first load dissolves in over the lowres, which stays until the fade ends.
          this.setAppear(group, 0)
          this.fading.set(tile, performance.now())
        }
        this.lastPlan = 0
        this.dirty = true
        return bytes
      })
    })
  }

  private dropHires(key: string, tile: HiresTile) {
    if (tile.state === 'loading') this.workers.abort(tile.req)
    this.fading.delete(tile)
    if (tile.group) this.disposeGroup(tile.group)
    this.hires.delete(key)
    this.setCover(tile.tx, tile.tz, false)
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

  private setAppear(g: THREE.Group, k: number) {
    for (const m of g.userData.materials as THREE.ShaderMaterial[]) m.uniforms.uAppear.value = k
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

  private buildHires(t: DecodedTile): { group: THREE.Group; bytes: number } {
    const dataTex = (data: Uint32Array | Uint16Array | Uint8Array, w: number, h: number, format: THREE.PixelFormat, type: THREE.TextureDataType, internal: THREE.PixelFormatGPU | null) => {
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
      m.uniforms = { ...base.uniforms, uQuads: { value: quads }, uTpl: { value: templates }, uPal: { value: colors }, uAppear: { value: 1 } }
      return m
    })
    const [minQ, maxQ] = [t.minQ, t.maxQ]
    const sphere = new THREE.Sphere(
      new THREE.Vector3(33 * 64, (minQ + maxQ) / 2, 33 * 64),
      Math.hypot(33 * 64, 33 * 64, (maxQ - minQ) / 2),
    )
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
        mesh.position.set(t.origin[0], t.origin[1], t.origin[2])
        mesh.scale.setScalar(1 / 64)
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

  // ---------------------------------------------------------------- lowres

  private loadLowres(key: string, x: number, z: number, level: 1 | 2, now: number) {
    const gen = this.generation
    const url = `/maps/${this.map!.id}/lowres/${level}/${x}_${z}.png`
    const { id, done } = this.workers.request('lowres', url)
    const existing = this.lowres.get(key)
    const tile: LowresTile = existing ?? {
      x,
      z,
      level,
      state: 'loading',
      req: id,
      mesh: null,
      material: null,
      texture: null,
      coarse: null,
      cover: [0, 0],
      born: now,
      stale: false,
    }
    tile.req = id
    tile.stale = false
    if (!existing) this.lowres.set(key, tile)
    const wasReady = tile.state === 'ready'
    tile.state = 'loading'
    done.then((res) => {
      if (gen !== this.generation || this.lowres.get(key) !== tile || tile.req !== id) return
      if (res.kind === 'error') {
        console.warn(`lowres ${key}: ${res.message}`)
        tile.state = wasReady ? 'ready' : 'empty'
        return
      }
      if (res.kind !== 'lowres') return
      this.downloaded += res.bytes
      if (!res.rgba) {
        this.disposeLowres(tile)
        tile.state = 'empty'
        this.dirty = true
        return
      }
      const tex = new THREE.DataTexture(res.rgba, REGION, REGION * 2, THREE.RGBAFormat, THREE.UnsignedByteType)
      tex.magFilter = tex.minFilter = THREE.NearestFilter
      tex.generateMipmaps = false
      tex.flipY = false
      tex.colorSpace = THREE.NoColorSpace
      tex.needsUpdate = true
      const scale = level === 1 ? 1 : 8
      if (tile.material && tile.texture) {
        tile.texture.dispose()
        tile.material.uniforms.uTex.value = tex
        tile.texture = tex
      } else {
        const material = new THREE.ShaderMaterial({
          glslVersion: THREE.GLSL3,
          vertexShader: lowresVertex,
          fragmentShader: lowresFragment,
          uniforms: {
            uTex: { value: tex },
            // uvec2: three uploads typed uint arrays with uniform2uiv.
            uMask: { value: new Uint32Array([tile.cover[0] >>> 0, tile.cover[1] >>> 0]) },
            uScale: { value: scale },
            uMinY: { value: this.manifest!.minY },
            uAppear: { value: 0 },
            uFogColor: { value: BG },
            uFogRange: { value: new THREE.Vector2(4000, 9000) },
          },
        })
        const mesh = new THREE.Mesh(this.lowresGrid, material)
        const size = level === 1 ? REGION : 4096
        mesh.position.set(x * size, 0, z * size)
        mesh.scale.set(scale, 1, scale)
        // Coarser levels draw first and sit slightly lower to stay behind finer ones.
        mesh.renderOrder = level === 2 ? -2 : -1
        if (level === 2) mesh.position.y = -0.6
        mesh.matrixAutoUpdate = false
        mesh.updateMatrix()
        mesh.frustumCulled = false
        tile.mesh = mesh
        tile.material = material
        tile.texture = tex
        tile.born = performance.now()
        this.recomputeCover(tile)
        this.scene.add(mesh)
      }
      tile.coarse = res.coarse
      tile.state = 'ready'
      if (level === 1) this.setCoverLevel2(x, z, true)
      this.lastPlan = 0
      this.dirty = true
    })
  }

  private dropLowres(key: string, tile: LowresTile) {
    if (tile.state === 'loading') this.workers.abort(tile.req)
    this.disposeLowres(tile)
    this.lowres.delete(key)
    if (tile.level === 1) this.setCoverLevel2(tile.x, tile.z, false)
    this.dirty = true
  }

  private disposeLowres(tile: LowresTile) {
    if (tile.mesh) this.scene.remove(tile.mesh)
    tile.material?.dispose()
    tile.texture?.dispose()
    tile.mesh = null
    tile.material = null
    tile.texture = null
  }

  /** Marks a hires tile as drawn (or not) over its level 1 tile. */
  private setCover(tx: number, tz: number, on: boolean) {
    const rx = Math.floor(tx / 8)
    const rz = Math.floor(tz / 8)
    const bit = (tz - rz * 8) * 8 + (tx - rx * 8)
    const lt = this.lowres.get(`1:${rx},${rz}`)
    if (lt) setBit(lt, bit, on)
  }

  private setCoverLevel2(rx: number, rz: number, on: boolean) {
    const qx = Math.floor(rx / 8)
    const qz = Math.floor(rz / 8)
    const bit = (rz - qz * 8) * 8 + (rx - qx * 8)
    const lt = this.lowres.get(`2:${qx},${qz}`)
    if (lt) setBit(lt, bit, on)
  }

  /** Rebuilds a lowres tile's cover bits from the finer tiles already drawn. */
  private recomputeCover(tile: LowresTile) {
    tile.cover = [0, 0]
    if (tile.level === 1) {
      for (const h of this.hires.values()) {
        // Tiles still dissolving in don't cover yet; their fade sets the bit when it ends.
        if (h.state === 'ready' && !this.fading.has(h) && Math.floor(h.tx / 8) === tile.x && Math.floor(h.tz / 8) === tile.z) {
          setBit(tile, (h.tz - tile.z * 8) * 8 + (h.tx - tile.x * 8), true)
        }
      }
    } else {
      for (const l of this.lowres.values()) {
        if (l.level === 1 && l.state === 'ready' && Math.floor(l.x / 8) === tile.x && Math.floor(l.z / 8) === tile.z) {
          setBit(tile, (l.z - tile.z * 8) * 8 + (l.x - tile.x * 8), true)
        }
      }
    }
  }

  /** Terrain height under a point, from level 1 data. */
  heightAt(x: number, z: number): number | null {
    const rx = Math.floor(x / REGION)
    const rz = Math.floor(z / REGION)
    const lt = this.lowres.get(`1:${rx},${rz}`)
    if (!lt?.coarse || !this.manifest) return null
    const cx = Math.floor((x - rx * REGION) / 8)
    const cz = Math.floor((z - rz * REGION) / 8)
    const h = lt.coarse[cz * 64 + cx]
    return h ? h + this.manifest.minY - 1 : null
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

    // Fades. Once a hires tile is fully in, the lowres under it can go.
    for (const [tile, start] of this.fading) {
      const k = Math.min(1, (now - start) / HIRES_FADE_MS)
      this.setAppear(tile.group!, k)
      if (k >= 1) {
        this.fading.delete(tile)
        this.setCover(tile.tx, tile.tz, true)
      }
      this.dirty = true
    }
    for (const lt of this.lowres.values()) {
      if (!lt.material) continue
      const k = Math.min(1, (now - lt.born) / FADE_MS)
      const u = lt.material.uniforms
      if (u.uAppear.value !== k) {
        u.uAppear.value = easeOut(k)
        this.dirty = true
      }
      const mask = u.uMask.value as Uint32Array
      if (mask[0] !== lt.cover[0] >>> 0 || mask[1] !== lt.cover[1] >>> 0) {
        mask[0] = lt.cover[0] >>> 0
        mask[1] = lt.cover[1] >>> 0
        this.dirty = true
      }
    }

    if (moved || this.dirty || this.bench) {
      this.gateLeaves()
      const fog = this.fogRange()
      for (const m of this.materials) (m.uniforms.uFogRange.value as THREE.Vector2).copy(fog)
      for (const lt of this.lowres.values()) lt.material?.uniforms.uFogRange.value.copy(fog)
      const t0 = performance.now()
      const q = this.gpuTimer?.begin()
      this.renderer.render(this.scene, this.camera)
      for (const g of this.justAdded) for (const c of g.children) c.frustumCulled = true
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
      let hi = 0
      for (const h of this.hires.values()) {
        gpu += h.gpuBytes
        if (h.state === 'ready') hi++
      }
      let lo = 0
      for (const l of this.lowres.values()) {
        if (l.texture) {
          gpu += REGION * REGION * 2 * 4
          lo++
        }
      }
      const pending =
        [...this.hires.values()].filter((h) => h.state === 'loading').length +
        [...this.lowres.values()].filter((l) => l.state === 'loading').length
      this.onStats({
        fps: (this.frames * 1000) / (now - this.lastStats),
        frameMs: this.frames ? this.frameMsAcc / this.frames : 0,
        gpuMs: this.gpuTimer?.p95() ?? null,
        calls: info.calls,
        triangles: info.triangles,
        hires: hi,
        lowres: lo,
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
    for (const h of this.hires.values()) {
      const mesh = h.group?.userData.leafInterior as THREE.Mesh | undefined
      if (!mesh) continue
      const dx = Math.max(h.tx * TILE - cam.x, 0, cam.x - (h.tx + 1) * TILE)
      const dz = Math.max(h.tz * TILE - cam.z, 0, cam.z - (h.tz + 1) * TILE)
      mesh.visible = Math.hypot(dx, dz) < LEAF_NEAR
    }
  }

  private clearTiles() {
    for (const [k, t] of this.hires) this.dropHires(k, t)
    for (const [k, t] of this.lowres) this.dropLowres(k, t)
    this.hires.clear()
    this.lowres.clear()
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

function setBit(lt: LowresTile, bit: number, on: boolean) {
  const w = bit >> 5
  const b = 1 << (bit & 31)
  lt.cover[w] = on ? lt.cover[w] | b : lt.cover[w] & ~b
}

function easeOut(k: number) {
  return 1 - (1 - k) ** 3
}

/** A flat grid of `n`×`n` cells spanning `size` units in x and z. */
function makeGrid(n: number, size: number): THREE.BufferGeometry {
  const pos = new Float32Array((n + 1) * (n + 1) * 3)
  const step = size / n
  for (let z = 0, i = 0; z <= n; z++) {
    for (let x = 0; x <= n; x++, i += 3) {
      pos[i] = x * step
      pos[i + 2] = z * step
    }
  }
  const idx = new Uint16Array(n * n * 6)
  for (let z = 0, i = 0; z < n; z++) {
    for (let x = 0; x < n; x++) {
      const a = z * (n + 1) + x
      const b = a + n + 1
      idx[i++] = a
      idx[i++] = b
      idx[i++] = a + 1
      idx[i++] = a + 1
      idx[i++] = b
      idx[i++] = b + 1
    }
  }
  const g = new THREE.BufferGeometry()
  g.setAttribute('position', new THREE.BufferAttribute(pos, 3))
  g.setIndex(new THREE.BufferAttribute(idx, 1))
  return g
}
