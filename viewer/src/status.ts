// The one line the viewer shows when the map isn't simply there.

/** What the server says a map is doing. */
export type MapStatus =
  | { state: 'starting' }
  | { state: 'rendering'; done: number; total: number }
  | { state: 'ready' }
  | { state: 'error' }

/** What's stopping the viewer from showing a map, if anything. */
export type Problem = 'unreachable' | 'waiting' | 'lost' | null

export interface StatusLine {
  text: string
  alert: boolean
}

/** A pass this big takes a while, so it's worth saying so even over a map that's already there. */
const BIG_PASS_TILES = 256

export function statusLine(
  name: string | undefined,
  problem: Problem,
  status: MapStatus | undefined,
  /** Whether the map has opened, and whether it has any tiles if so. */
  opened: boolean,
  hasTiles: boolean,
): StatusLine | null {
  if (problem === 'lost') return { text: 'Restoring the map’s graphics…', alert: false }
  if (problem === 'unreachable') return { text: 'Can’t reach the server. Retrying…', alert: true }
  const map = name ?? 'this map'
  switch (status?.state) {
    case 'error':
      return {
        text: hasTiles ? 'This map isn’t updating. Showing its last render.' : `Couldn’t render ${map}. The server log says why.`,
        alert: true,
      }
    case 'rendering':
      if (hasTiles && status.total < BIG_PASS_TILES) return null
      return { text: `Rendering ${map}: ${Math.floor((status.done / Math.max(status.total, 1)) * 100)}%`, alert: false }
    case 'starting':
      return hasTiles ? null : { text: `Starting ${map}…`, alert: false }
    case 'ready':
      return opened && !hasTiles ? { text: `Nothing has been explored on ${map} yet.`, alert: false } : null
    default:
      // A server too old to say, or one that hasn't said yet.
      return problem === 'waiting' ? { text: `Waiting for the first render of ${map}…`, alert: false } : null
  }
}
