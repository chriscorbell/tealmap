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

export function parseHash(hash: string): Route {
  const [id, ...parts] = hash.slice(1).split('/')
  const v = parts.map(Number)
  const valid = (v.length === 5 || v.length === 8) && v.every(Number.isFinite)
  if (!id || !valid) return { id: id || null }
  return {
    id,
    view: { x: v[0], z: v[1], dist: v[2], az: v[3], polar: v[4] },
    pin: v.length === 8 ? { x: v[5], y: v[6], z: v[7] } : undefined,
  }
}

export function formatHash(id: string, v: View, pin?: Pin | null): string {
  const view = `#${id}/${v.x}/${v.z}/${v.dist}/${v.az}/${v.polar}`
  return pin ? `${view}/${pin.x}/${pin.y}/${pin.z}` : view
}

/**
 * Reads coordinates out of pasted or typed text, such as `120 64 -340`, `120, -340`
 * or `/tp @s 120 64 -340`. Two numbers are x and z; with more, the first three are x, y, z.
 */
export function parseCoords(text: string): { x: number; z: number } | null {
  const n = (text.match(/-?\d+(\.\d+)?/g) ?? []).map(Number)
  if (n.length === 2) return { x: n[0], z: n[1] }
  if (n.length >= 3) return { x: n[0], z: n[2] }
  return null
}
