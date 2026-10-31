// Decodes a tile (see src/mesh.rs) into the arrays the vertex pulling shader
// reads as textures: one RG32UI texel per quad, and a table of RGBA16UI texels
// with colors below a base, counting down, and six texels per template above it.

export const GROUPS = 4
export const DIRS = 7
/** Group index of leaf faces that touch other leaves. */
export const LEAF_INTERIOR = 2
export const QUAD_ROW = 2048
export const TABLE_ROW = 1024
const TEMPLATE_TEXELS = 6
const UP = 1
const OTHER = 6
const SIDE = 64
/** A column with nothing in it. */
export const NO_TOP = -32768

export interface DecodedTile {
  origin: [number, number, number]
  minQ: number
  maxQ: number
  /** Quads per stream: group * DIRS + direction. */
  counts: number[]
  /** Per quad: x | z << 7 | y << 14 | (w - 1) << 24, then template | color << 13 | (h - 1) << 26. */
  quads: Uint32Array
  quadRows: number
  /** Colors (r, g, b, 255), counting down from tableBase, then templates: 4 corners (x, y, z, layer) and UV pairs for corners 0–1 and 2–3. */
  tables: Uint16Array
  tableRows: number
  tableBase: number
  /** Top of each of the 64×64 columns, in the tile's own units (blocks or voxels), or NO_TOP. */
  tops: Int16Array
}

const rows = (texels: number, width: number) => Math.max(1, Math.ceil(texels / width))

/**
 * A corner's bits beside its 12-bit offset. x: which axes the quad stretches
 * along (12-13), and whether this corner is at the far end of u (14) and v (15).
 * y and z: the texture step per block along u and along v (12-15).
 */
function writeTemplates(tables: Uint16Array, base: number, nT: number, field: (k: number, t: number) => number) {
  for (let t = 0; t < nT; t++) {
    const d = (base + t * TEMPLATE_TEXELS) * 4
    const layer = field(20, t)
    const stretch = field(21, t)
    let axes = 0
    if (stretch & 0xff) {
      // Which block axis the far corners differ along.
      let near = 0
      while ((stretch >> near) & 1) near++
      const far = (stretch & 1) ? 0 : (stretch & 2) ? 1 : (stretch & 4) ? 2 : 3
      const alongX = field(far * 3, t) !== field(near * 3, t)
      // Along u: x unless the face looks along x. Along v: z on top and bottom faces.
      let nearV = 0
      while ((stretch >> (4 + nearV)) & 1) nearV++
      const farV = (stretch >> 4) & 1 ? 0 : (stretch >> 4) & 2 ? 1 : (stretch >> 4) & 4 ? 2 : 3
      const vAlongY = field(farV * 3 + 1, t) !== field(nearV * 3 + 1, t)
      axes = !alongX ? 2 : vAlongY ? 1 : 0
    }
    for (let k = 0; k < 4; k++) {
      tables[d + k * 4] = field(k * 3, t) | (axes << 12) | (((stretch >> k) & 1) << 14) | (((stretch >> (4 + k)) & 1) << 15)
      tables[d + k * 4 + 1] = field(k * 3 + 1, t) | (((stretch >> 8) & 15) << 12)
      tables[d + k * 4 + 2] = field(k * 3 + 2, t) | (((stretch >> 12) & 15) << 12)
      tables[d + k * 4 + 3] = layer
      tables[d + 16 + k * 2] = field(12 + k * 2, t)
      tables[d + 17 + k * 2] = field(13 + k * 2, t)
    }
  }
}

export function decodeTile(buf: ArrayBuffer): DecodedTile {
  const b = new Uint8Array(buf)
  const dv = new DataView(buf)
  const magic = String.fromCharCode(b[0], b[1], b[2], b[3])
  if (magic === 'TMH2') return decodeLegacy(b, dv)
  if (magic !== 'TMH3') throw new Error('bad tile header')
  const origin: [number, number, number] = [dv.getInt32(4, true), dv.getInt32(8, true), dv.getInt32(12, true)]
  const minQ = dv.getUint16(16, true)
  const maxQ = dv.getUint16(18, true)
  const counts: number[] = []
  let total = 0
  for (let s = 0; s < GROUPS * DIRS; s++) {
    const n = dv.getUint32(20 + s * 4, true)
    counts.push(n)
    total += n
  }
  const nT = dv.getUint32(132, true)
  const nP = dv.getUint32(136, true)
  let o = 140

  const tableRows = rows(nP + nT * TEMPLATE_TEXELS, TABLE_ROW)
  const tables = new Uint16Array(TABLE_ROW * tableRows * 4)
  const base = o
  writeTemplates(tables, nP, nT, (k, t) => b[base + k * 2 * nT + t] | (b[base + k * 2 * nT + nT + t] << 8))
  o += nT * 44
  for (let c = 0; c < nP; c++) {
    const d = (nP - 1 - c) * 4
    tables[d] = b[o + c]
    tables[d + 1] = b[o + nP + c]
    tables[d + 2] = b[o + 2 * nP + c]
    tables[d + 3] = 255
  }
  o += nP * 3

  const quadRows = rows(total, QUAD_ROW)
  const quads = new Uint32Array(QUAD_ROW * quadRows * 2)
  const tops = new Int16Array(SIDE * SIDE).fill(NO_TOP)
  let w = 0
  for (let s = 0; s < GROUPS * DIRS; s++) {
    const n = counts[s]
    const dir = s % DIRS
    const pl = o
    const vp = o + 2 * n
    const up = o + 4 * n
    const wp = o + 5 * n
    const hp = o + 6 * n
    const tp = o + 7 * n
    const cp = o + 9 * n
    o += 11 * n
    let plane = 0
    let v = 0
    let u = 0
    for (let i = 0; i < n; i++) {
      const dPlane = b[pl + i] | (b[pl + n + i] << 8)
      const dV = b[vp + i] | (b[vp + n + i] << 8)
      const newPlane = dPlane !== 0 || i === 0
      plane += dPlane
      v = newPlane ? dV : v + dV
      u = newPlane || dV !== 0 ? b[up + i] : u + b[up + i]
      const x = dir === 4 || dir === 5 ? plane : u
      const y = dir === 2 || dir === 3 || dir === 4 || dir === 5 ? v : plane
      const z = dir === 2 || dir === 3 ? plane : dir === 4 || dir === 5 ? u : v
      const qw = b[wp + i]
      const qh = b[hp + i]
      quads[w++] = x | (z << 7) | (y << 14) | (qw << 24)
      quads[w++] = ((b[tp + i] | (b[tp + n + i] << 8)) | ((b[cp + i] | (b[cp + n + i] << 8)) << 13) | (qh << 26)) >>> 0
      if (dir === UP || dir === OTHER) {
        // Column 1 is the tile's first; 0 and 65 are the halo.
        const top = origin[1] + y + 1
        for (let cz = Math.max(z, 1); cz <= Math.min(z + qh, SIDE); cz++) {
          for (let cx = Math.max(x, 1); cx <= Math.min(x + qw, SIDE); cx++) {
            const c = (cz - 1) * SIDE + (cx - 1)
            if (top > tops[c]) tops[c] = top
          }
        }
      }
    }
  }
  return { origin, minQ, maxQ, counts, quads, quadRows, tables, tableRows, tableBase: nP, tops }
}

/** Tiles from before merged quads, still on disk while a new render pass replaces them. */
function decodeLegacy(b: Uint8Array, dv: DataView): DecodedTile {
  const W = 66
  const origin: [number, number, number] = [dv.getInt32(4, true), dv.getInt32(8, true), dv.getInt32(12, true)]
  const groupCounts = [0, 1, 2, 3].map((g) => dv.getUint32(20 + g * 4, true))
  const nT = dv.getUint32(36, true)
  const nP = dv.getUint32(40, true)
  let o = 44
  const tableRows = rows(nP + nT * TEMPLATE_TEXELS, TABLE_ROW)
  const tables = new Uint16Array(TABLE_ROW * tableRows * 4)
  const base = o
  writeTemplates(tables, nP, nT, (k, t) => (k === 21 ? 0 : dv.getUint16(base + t * 42 + k * 2, true)))
  o += nT * 42
  for (let c = 0; c < nP; c++) {
    const d = (nP - 1 - c) * 4
    tables[d] = b[o + c * 4]
    tables[d + 1] = b[o + c * 4 + 1]
    tables[d + 2] = b[o + c * 4 + 2]
    tables[d + 3] = 255
  }
  o += nP * 4
  const total = groupCounts.reduce((a, n) => a + n, 0)
  const quadRows = rows(total, QUAD_ROW)
  const quads = new Uint32Array(QUAD_ROW * quadRows * 2)
  const tops = new Int16Array(SIDE * SIDE).fill(NO_TOP)
  const counts = new Array<number>(GROUPS * DIRS).fill(0)
  let w = 0
  groupCounts.forEach((n, g) => {
    counts[g * DIRS + DIRS - 1] = n
    const cLo = o
    const cHi = o + W * W
    o += 2 * W * W
    const sLo = o
    const sHi = o + n
    const tLo = o + 2 * n
    const tHi = o + 3 * n
    const pLo = o + 4 * n
    const pHi = o + 5 * n
    o += 6 * n
    let q = 0
    for (let z = 0; z < W; z++) {
      for (let x = 0; x < W; x++) {
        const cnt = b[cLo + z * W + x] | (b[cHi + z * W + x] << 8)
        let y = 1023
        for (let j = 0; j < cnt; j++, q++) {
          y -= b[sLo + q] | (b[sHi + q] << 8)
          quads[w++] = x | (z << 7) | (y << 14)
          if (x >= 1 && x <= SIDE && z >= 1 && z <= SIDE) {
            const c = (z - 1) * SIDE + (x - 1)
            const top = origin[1] + y + 1
            if (top > tops[c]) tops[c] = top
          }
          quads[w++] = (b[tLo + q] | (b[tHi + q] << 8) | ((b[pLo + q] | (b[pHi + q] << 8)) << 13)) >>> 0
        }
      }
    }
  })
  return { origin, minQ: dv.getUint16(16, true), maxQ: dv.getUint16(18, true), counts, quads, quadRows, tables, tableRows, tableBase: nP, tops }
}
