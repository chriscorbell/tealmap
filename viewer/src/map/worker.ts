// Fetches and decodes tiles off the main thread. Tiles arrive gzip-encoded
// over HTTP, so the browser inflates them; the worker unpacks them into the
// texture data the vertex-pulling shader reads.

import { type DecodedTile, decodeTile } from './tile'

export type Request = { kind: 'tile'; id: number; url: string } | { kind: 'abort'; id: number }

export type Response =
  | { kind: 'tile'; id: number; tile: DecodedTile | null; bytes: number }
  | { kind: 'error'; id: number; message: string }

const inflight = new Map<number, AbortController>()

self.onmessage = async (e: MessageEvent<Request>) => {
  const req = e.data
  if (req.kind === 'abort') {
    inflight.get(req.id)?.abort()
    inflight.delete(req.id)
    return
  }
  const ctl = new AbortController()
  inflight.set(req.id, ctl)
  try {
    const res = await fetch(req.url, { signal: ctl.signal })
    if (res.status === 404) {
      post({ kind: 'tile', id: req.id, tile: null, bytes: 0 })
      return
    }
    if (!res.ok) throw new Error(`${res.status} ${res.statusText}`)
    const buf = await res.arrayBuffer()
    const tile = decodeTile(buf)
    // Count bytes on the wire: Content-Length is the gzipped size.
    const wire = Number(res.headers.get('content-length')) || buf.byteLength
    post({ kind: 'tile', id: req.id, tile, bytes: wire }, [tile.quads.buffer, tile.templates.buffer, tile.colors.buffer, tile.tops.buffer])
  } catch (err) {
    if ((err as Error).name === 'AbortError') return
    post({ kind: 'error', id: req.id, message: String(err) })
  } finally {
    inflight.delete(req.id)
  }
}

function post(msg: Response, transfer: Transferable[] = []) {
  ;(self as unknown as Worker).postMessage(msg, transfer)
}
