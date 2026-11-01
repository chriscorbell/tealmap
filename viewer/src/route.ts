// The URL hash: #map/x/z/distance/azimuth/polar, optionally with a pinned block
// as /px/py/pz. Links made before pins existed have no pin, and still work.

export interface View {
  x: number
  z: number
  dist: number
  az: number
  polar: number
}

export interface Pin {
  x: number
  y: number
  z: number
}

export interface Route {
  id: string | null
  view?: View
  pin?: Pin
}

/** Further than any Minecraft world goes. */
const MAX_COORD = 30_000_000
const MAX_Y = 4096
const MAX_DIST = 1_000_000
const DECIMAL = /^-?\d+(\.\d+)?$/
const INTEGER = /^-?\d+$/

export function parseHash(hash: string): Route {
  const [id, ...parts] = hash.slice(1).split('/')
  if (!id) return { id: null }
  const isPin = parts.length === 8
  const valid =
    (parts.length === 5 || isPin) &&
    parts.slice(0, 5).every((p) => DECIMAL.test(p)) &&
    parts.slice(5).every((p) => INTEGER.test(p))
  if (!valid) return { id }
  const v = parts.map(Number)
  const within = Math.abs(v[0]) <= MAX_COORD && Math.abs(v[1]) <= MAX_COORD
  const camera = v[2] > 0 && v[2] <= MAX_DIST && Math.abs(v[3]) <= 3600 && v[4] >= 0 && v[4] <= 180
  const pinned = !isPin || (Math.abs(v[5]) <= MAX_COORD && Math.abs(v[6]) <= MAX_Y && Math.abs(v[7]) <= MAX_COORD)
  if (!within || !camera || !pinned) return { id }
  return {
    id,
    view: { x: v[0], z: v[1], dist: v[2], az: v[3], polar: v[4] },
    pin: isPin ? { x: v[5], y: v[6], z: v[7] } : undefined,
  }
}

export function formatHash(id: string, v: View, pin?: Pin | null): string {
  const view = `#${id}/${v.x}/${v.z}/${v.dist}/${v.az}/${v.polar}`
  return pin ? `${view}/${pin.x}/${pin.y}/${pin.z}` : view
}

/** A coordinate typed into one field, or null if it isn't one. */
export function parseCoord(text: string): number | null {
  const t = text.trim()
  if (!DECIMAL.test(t)) return null
  const n = Number(t)
  return Math.abs(n) <= MAX_COORD ? n : null
}

/**
 * Reads a position out of pasted or typed text, such as `120 64 -340`, `120, -340`,
 * `XYZ: 120.5 / 64 / -340.5` or `/tp @s 120 64 -340`. Two numbers are x and z; with
 * more, the first three are x, y, z. Words with digits in them aren't numbers, and
 * neither are relative coordinates like `~10`.
 */
export function parseCoords(text: string): { x: number; z: number } | null {
  const t = text.trim()
  // A link has plenty of numbers in it, and none of them are a position to go to.
  if (t.startsWith('#') || /^[a-z]+:\/\//i.test(t)) return null
  const n = t
    .split(/[\s,;:/()[\]{}]+/)
    .map(parseCoord)
    .filter((v) => v !== null)
  if (n.length === 2) return { x: n[0], z: n[1] }
  if (n.length >= 3) return { x: n[0], z: n[2] }
  return null
}
