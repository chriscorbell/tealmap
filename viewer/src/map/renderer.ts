// Draws tiles with raw WebGL2. All tile data lives in a few big textures that
// tiles rent rows from, so a frame binds once and issues a handful of multi-draw
// calls however many tiles it shows.

import { fragmentShader, vertexShader } from './shaders'
import { type DecodedTile, DIRS, GROUPS, LEAF_INTERIOR, QUAD_ROW, TABLE_ROW } from './tile'

export const Pass = { Opaque: 0, OpaqueFade: 1, Cutout: 2, Translucent: 3 } as const
export type Pass = (typeof Pass)[keyof typeof Pass]
const PASSES = 4
/** Rows and columns of the per-draw table: 64 draws of two texels per row. */
const DRAWS_PER_ROW = 64
const DRAW_ROWS = 256
const MAX_DRAWS = DRAWS_PER_ROW * DRAW_ROWS
const DOWN = 0
const UP = 1
const NORTH = 2
const SOUTH = 3
const WEST = 4
const EAST = 5
const OTHER = 6

/** Rows of a pool texture rented to one tile. */
interface Rows {
  start: number
  rows: number
}

/** Rents out contiguous rows of a texture, first fit. */
class RowPool {
  private free: Rows[]
  used = 0

  constructor(rows: number) {
    this.free = [{ start: 0, rows }]
  }

  alloc(rows: number): Rows | null {
    for (let i = 0; i < this.free.length; i++) {
      const f = this.free[i]
      if (f.rows < rows) continue
      const got = { start: f.start, rows }
      f.start += rows
      f.rows -= rows
      if (f.rows === 0) this.free.splice(i, 1)
      this.used += rows
      return got
    }
    return null
  }

  release(r: Rows) {
    this.used -= r.rows
    // Keep the list sorted and merge neighbors, so space doesn't fragment for good.
    let i = 0
    while (i < this.free.length && this.free[i].start < r.start) i++
    this.free.splice(i, 0, { start: r.start, rows: r.rows })
    const next = this.free[i + 1]
    if (next && r.start + r.rows === next.start) {
      this.free[i].rows += next.rows
      this.free.splice(i + 1, 1)
    }
    const prev = this.free[i - 1]
    if (prev && prev.start + prev.rows === this.free[i].start) {
      prev.rows += this.free[i].rows
      this.free.splice(i, 1)
    }
  }
}

/** A tile's data on the GPU. */
export interface Mesh {
  quads: Rows
  tables: Rows
  /** Where colors end and templates start, in texels into the table pool. */
  tableBase: number
  /** First quad, in the quad pool, and quad count per stream. */
  first: Int32Array
  counts: Int32Array
  /** Blocks per voxel, and the origin and y bounds in blocks. */
  scale: number
  origin: [number, number, number]
  minY: number
  maxY: number
}

const inPass = (group: number, pass: Pass) => {
  switch (pass) {
    case Pass.Opaque:
    case Pass.OpaqueFade:
      return group === 0
    case Pass.Cutout:
      return group === 1 || group === LEAF_INTERIOR
    default:
      return group === 3
  }
}

function compile(gl: WebGL2RenderingContext, type: number, source: string) {
  const s = gl.createShader(type)!
  gl.shaderSource(s, source)
  gl.compileShader(s)
  if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(`shader: ${gl.getShaderInfoLog(s)}`)
  return s
}

export class Renderer {
  readonly gl: WebGL2RenderingContext
  private multiDraw: WEBGL_multi_draw | null
  private programs: WebGLProgram[] = []
  private uniforms: Record<'viewProj' | 'cam' | 'firstDraw' | 'fogColor' | 'fogRange', WebGLUniformLocation | null>[] = []
  private blockTex: WebGLTexture | null = null
  private index: WebGLBuffer
  private indexQuads = 0
  private quadRows: RowPool
  private tableRows: RowPool
  private blockBytes = 0

  private drawData = new Float32Array(MAX_DRAWS * 8)
  private drawCounts = new Int32Array(MAX_DRAWS)
  private drawOffsets = new Int32Array(MAX_DRAWS)
  /** Index of each pass's first draw, then one past the last. */
  private passStart = new Int32Array(PASSES + 1)
  private drawn = 0
  private triangles = 0

  constructor(gl: WebGL2RenderingContext) {
    this.gl = gl
    this.multiDraw = gl.getExtension('WEBGL_multi_draw')
    // A bigger screen shows more tiles at full detail. These are 64 and 16 MB on
    // a 1080p screen or smaller, and up to twice that on anything larger.
    const maxSize = gl.getParameter(gl.MAX_TEXTURE_SIZE) as number
    const screenScale = Math.min(2, Math.max(1, (screen.width * screen.height) / (1920 * 1080)))
    const quadHeight = Math.min(Math.ceil(4096 * screenScale), maxSize)
    const tableHeight = Math.min(Math.ceil(2048 * screenScale), maxSize)
    const storage = (unit: number, internal: number, w: number, h: number) => {
      const t = gl.createTexture()!
      gl.activeTexture(gl.TEXTURE0 + unit)
      gl.bindTexture(gl.TEXTURE_2D, t)
      gl.texStorage2D(gl.TEXTURE_2D, 1, internal, w, h)
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST)
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST)
    }
    storage(0, gl.RG32UI, QUAD_ROW, quadHeight)
    storage(1, gl.RGBA16UI, TABLE_ROW, tableHeight)
    storage(2, gl.RGBA32F, DRAWS_PER_ROW * 2, DRAW_ROWS)
    this.quadRows = new RowPool(quadHeight)
    this.tableRows = new RowPool(tableHeight)

    this.index = gl.createBuffer()!
    gl.bindVertexArray(gl.createVertexArray())
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this.index)
    this.growIndex(1 << 16)

    const variants = [['OPAQUE'], ['OPAQUE', 'FADE'], ['CUTOUT', 'FADE'], ['TRANSLUCENT', 'FADE']]
    for (const defines of variants) {
      const head = [
        '#version 300 es',
        this.multiDraw ? '#extension GL_ANGLE_multi_draw : require' : '#define NO_MULTI_DRAW',
        ...defines.map((d) => `#define ${d}`),
      ].join('\n')
      const p = gl.createProgram()!
      gl.attachShader(p, compile(gl, gl.VERTEX_SHADER, `${head}\n${vertexShader}`))
      gl.attachShader(p, compile(gl, gl.FRAGMENT_SHADER, `${head}\n${fragmentShader}`))
      gl.linkProgram(p)
      if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error(`program: ${gl.getProgramInfoLog(p)}`)
      gl.useProgram(p)
      gl.uniform1i(gl.getUniformLocation(p, 'uQuads'), 0)
      gl.uniform1i(gl.getUniformLocation(p, 'uTables'), 1)
      gl.uniform1i(gl.getUniformLocation(p, 'uDraws'), 2)
      gl.uniform1i(gl.getUniformLocation(p, 'uTex'), 3)
      this.programs.push(p)
      this.uniforms.push({
        viewProj: gl.getUniformLocation(p, 'uViewProj'),
        cam: gl.getUniformLocation(p, 'uCam'),
        firstDraw: gl.getUniformLocation(p, 'uFirstDraw'),
        fogColor: gl.getUniformLocation(p, 'uFogColor'),
        fogRange: gl.getUniformLocation(p, 'uFogRange'),
      })
    }
    gl.enable(gl.DEPTH_TEST)
    gl.enable(gl.CULL_FACE)
    gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1)
  }

  private growIndex(quads: number) {
    const idx = new Uint32Array(quads * 6)
    for (let q = 0, v = 0, i = 0; q < quads; q++, v += 4) {
      idx[i++] = v
      idx[i++] = v + 1
      idx[i++] = v + 2
      idx[i++] = v
      idx[i++] = v + 2
      idx[i++] = v + 3
    }
    this.gl.bufferData(this.gl.ELEMENT_ARRAY_BUFFER, idx, this.gl.STATIC_DRAW)
    this.indexQuads = quads
  }

  setBlockTextures(data: Uint8Array, layers: number) {
    const gl = this.gl
    if (this.blockTex) gl.deleteTexture(this.blockTex)
    const t = gl.createTexture()!
    gl.activeTexture(gl.TEXTURE3)
    gl.bindTexture(gl.TEXTURE_2D_ARRAY, t)
    gl.texStorage3D(gl.TEXTURE_2D_ARRAY, 5, gl.RGBA8, 16, 16, layers)
    gl.texSubImage3D(gl.TEXTURE_2D_ARRAY, 0, 0, 0, 0, 16, 16, layers, gl.RGBA, gl.UNSIGNED_BYTE, data)
    gl.generateMipmap(gl.TEXTURE_2D_ARRAY)
    gl.texParameteri(gl.TEXTURE_2D_ARRAY, gl.TEXTURE_MAG_FILTER, gl.NEAREST)
    gl.texParameteri(gl.TEXTURE_2D_ARRAY, gl.TEXTURE_MIN_FILTER, gl.NEAREST_MIPMAP_LINEAR)
    gl.texParameteri(gl.TEXTURE_2D_ARRAY, gl.TEXTURE_WRAP_S, gl.REPEAT)
    gl.texParameteri(gl.TEXTURE_2D_ARRAY, gl.TEXTURE_WRAP_T, gl.REPEAT)
    const aniso = gl.getExtension('EXT_texture_filter_anisotropic')
    if (aniso) {
      const max = gl.getParameter(aniso.MAX_TEXTURE_MAX_ANISOTROPY_EXT) as number
      gl.texParameterf(gl.TEXTURE_2D_ARRAY, aniso.TEXTURE_MAX_ANISOTROPY_EXT, Math.min(8, max))
    }
    this.blockTex = t
    this.blockBytes = layers * 1024 * 1.34
  }

  /** Copies a decoded tile to the GPU, or returns null if the pools are full. */
  upload(t: DecodedTile, level: number): Mesh | null {
    const gl = this.gl
    const quads = this.quadRows.alloc(t.quadRows)
    if (!quads) return null
    const tables = this.tableRows.alloc(t.tableRows)
    if (!tables) {
      this.quadRows.release(quads)
      return null
    }
    gl.activeTexture(gl.TEXTURE0)
    gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, quads.start, QUAD_ROW, t.quadRows, gl.RG_INTEGER, gl.UNSIGNED_INT, t.quads)
    gl.activeTexture(gl.TEXTURE1)
    gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, tables.start, TABLE_ROW, t.tableRows, gl.RGBA_INTEGER, gl.UNSIGNED_SHORT, t.tables)

    const first = new Int32Array(GROUPS * DIRS)
    const counts = new Int32Array(t.counts)
    let at = quads.start * QUAD_ROW
    let biggest = 0
    for (let s = 0; s < GROUPS * DIRS; s++) {
      first[s] = at
      at += counts[s]
      if (counts[s] > biggest) biggest = counts[s]
    }
    if (biggest > this.indexQuads) this.growIndex(Math.max(biggest, this.indexQuads * 2))
    const scale = 1 << level
    // Heights are in 1/64 voxel from the origin.
    const y = (q: number) => (t.origin[1] + q / 64) * scale
    return {
      quads,
      tables,
      tableBase: tables.start * TABLE_ROW + t.tableBase,
      first,
      counts,
      scale,
      origin: [t.origin[0] * scale, t.origin[1] * scale, t.origin[2] * scale],
      minY: y(t.minQ),
      maxY: y(t.maxQ),
    }
  }

  release(m: Mesh) {
    this.quadRows.release(m.quads)
    this.tableRows.release(m.tables)
  }

  /** Bytes rented out of the pools, plus the block textures. */
  get gpuBytes() {
    return this.quadRows.used * QUAD_ROW * 8 + this.tableRows.used * TABLE_ROW * 8 + this.blockBytes
  }

  /** Starts a frame's draw list. */
  begin() {
    this.drawn = 0
    this.triangles = 0
  }

  /** Starts a pass. Passes go in order; a skipped pass is simply empty. */
  startPass(pass: Pass) {
    for (let p = pass; p <= PASSES; p++) this.passStart[p] = this.drawn
  }

  /**
   * Adds a tile's visible streams to the current pass. Streams that face away
   * from the camera are skipped: an east-facing wall can only be seen from east
   * of it. `dissolve` is 1 when settled, 0..1 coming in, and -1..0 going out.
   */
  add(pass: Pass, m: Mesh, cam: ArrayLike<number>, leafInterior: boolean, dissolve: number, plants = true) {
    // Faces can reach past their tile; two voxels of slack cover it.
    const slack = 2 * m.scale
    const [ox, oy, oz] = m.origin
    const size = 66 * m.scale
    for (let g = 0; g < GROUPS; g++) {
      if (!inPass(g, pass) || (g === LEAF_INTERIOR && !leafInterior)) continue
      for (let d = 0; d < DIRS; d++) {
        const s = g * DIRS + d
        const n = m.counts[s]
        if (n === 0) continue
        if (d === DOWN && cam[1] > m.maxY + slack) continue
        if (d === UP && cam[1] < m.minY - slack) continue
        if (d === NORTH && cam[2] > oz + size + slack) continue
        if (d === SOUTH && cam[2] < oz - slack) continue
        if (d === WEST && cam[0] > ox + size + slack) continue
        if (d === EAST && cam[0] < ox - slack) continue
        if (d === OTHER && !plants) continue
        const i = this.drawn
        if (i >= MAX_DRAWS) return
        this.drawCounts[i] = n * 6
        const o = i * 8
        const dd = this.drawData
        // Pool coordinates pass through floats; they stay well under 2^24.
        dd[o] = ox
        dd[o + 1] = oy
        dd[o + 2] = oz
        dd[o + 3] = m.scale / 64
        dd[o + 4] = m.first[s]
        dd[o + 5] = m.tableBase
        dd[o + 6] = dissolve
        this.drawn++
        this.triangles += n * 2
        this.passStart[pass + 1] = this.drawn
      }
    }
  }

  /** Draws the list. `viewProj` has no translation; positions are camera-relative. */
  draw(viewProj: Float32Array, cam: ArrayLike<number>, fogColor: ArrayLike<number>, fogRange: ArrayLike<number>) {
    const gl = this.gl
    const n = this.drawn
    if (n > 0) {
      const rows = Math.ceil(n / DRAWS_PER_ROW)
      gl.activeTexture(gl.TEXTURE2)
      gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, DRAWS_PER_ROW * 2, rows, gl.RGBA, gl.FLOAT, this.drawData)
    }
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT)
    let calls = 0
    for (let pass = 0; pass < PASSES; pass++) {
      const start = this.passStart[pass]
      const count = this.passStart[pass + 1] - start
      if (count <= 0) continue
      gl.useProgram(this.programs[pass])
      const u = this.uniforms[pass]
      gl.uniformMatrix4fv(u.viewProj, false, viewProj)
      gl.uniform3f(u.cam, cam[0], cam[1], cam[2])
      gl.uniform3f(u.fogColor, fogColor[0], fogColor[1], fogColor[2])
      gl.uniform2f(u.fogRange, fogRange[0], fogRange[1])
      if (pass === Pass.Translucent) {
        gl.enable(gl.BLEND)
        gl.blendFuncSeparate(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA, gl.ONE, gl.ONE_MINUS_SRC_ALPHA)
        gl.depthMask(false)
      }
      if (this.multiDraw) {
        gl.uniform1i(u.firstDraw, start)
        this.multiDraw.multiDrawElementsWEBGL(gl.TRIANGLES, this.drawCounts, start, gl.UNSIGNED_INT, this.drawOffsets, start, count)
        calls++
      } else {
        for (let i = 0; i < count; i++) {
          gl.uniform1i(u.firstDraw, start + i)
          gl.drawElements(gl.TRIANGLES, this.drawCounts[start + i], gl.UNSIGNED_INT, 0)
          calls++
        }
      }
      if (pass === Pass.Translucent) {
        gl.disable(gl.BLEND)
        gl.depthMask(true)
      }
    }
    return { calls, triangles: this.triangles }
  }
}
