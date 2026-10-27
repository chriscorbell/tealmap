// Decodes a TMH2 hires tile (see src/mesh.rs) into the arrays the vertex
// pulling shader reads as textures: one RG32UI texel per quad, eight RGBA16UI
// texels per template, and one RGBA8 texel per color.

export const GROUPS = 4
/** Group index of leaf faces that touch other leaves. */
export const LEAF_INTERIOR = 2
export const QUAD_ROW = 2048
export const TABLE_ROW = 4096
const W = 66

export interface DecodedTile {
  origin: [number, number, number]
  minQ: number
  maxQ: number
  /** Quads per group: opaque, cutout, leaf interior, translucent. */
  counts: number[]
  quads: Uint32Array
  quadRows: number
  templates: Uint16Array
  templateRows: number
  colors: Uint8Array
  colorRows: number
}

const rows = (texels: number, width: number) => Math.max(1, Math.ceil(texels / width))

export function decodeTile(buf: ArrayBuffer): DecodedTile {
  const b = new Uint8Array(buf)
  const dv = new DataView(buf)
  if (String.fromCharCode(b[0], b[1], b[2], b[3]) !== 'TMH2') throw new Error('bad tile header')
  const origin: [number, number, number] = [dv.getInt32(4, true), dv.getInt32(8, true), dv.getInt32(12, true)]
  const minQ = dv.getUint16(16, true)
  const maxQ = dv.getUint16(18, true)
  const counts = [0, 1, 2, 3].map((g) => dv.getUint32(20 + g * 4, true))
  const nT = dv.getUint32(36, true)
  const nP = dv.getUint32(40, true)
  let o = 44

  // Templates: per corner (x, y, z, layer), then UV pairs for corners 0–1 and 2–3.
  const templateRows = rows(nT * 8, TABLE_ROW)
  const templates = new Uint16Array(TABLE_ROW * templateRows * 4)
  for (let t = 0; t < nT; t++, o += 42) {
    const d = t * 32
    const layer = dv.getUint16(o + 40, true)
    for (let k = 0; k < 4; k++) {
      templates[d + k * 4] = dv.getUint16(o + k * 6, true)
      templates[d + k * 4 + 1] = dv.getUint16(o + k * 6 + 2, true)
      templates[d + k * 4 + 2] = dv.getUint16(o + k * 6 + 4, true)
      templates[d + k * 4 + 3] = layer
      templates[d + 16 + k * 2] = dv.getUint16(o + 24 + k * 4, true)
      templates[d + 17 + k * 2] = dv.getUint16(o + 26 + k * 4, true)
    }
  }

  const colorRows = rows(nP, TABLE_ROW)
  const colors = new Uint8Array(TABLE_ROW * colorRows * 4)
  colors.set(b.subarray(o, o + nP * 4))
  o += nP * 4

  const total = counts.reduce((a, n) => a + n, 0)
  const quadRows = rows(total, QUAD_ROW)
  const quads = new Uint32Array(QUAD_ROW * quadRows * 2)
  let w = 0
  for (const n of counts) {
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
          quads[w++] = (b[tLo + q] | (b[tHi + q] << 8) | ((b[pLo + q] | (b[pHi + q] << 8)) << 16)) >>> 0
        }
      }
    }
  }
  return { origin, minQ, maxQ, counts, quads, quadRows, templates, templateRows, colors, colorRows }
}
