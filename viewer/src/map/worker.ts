// Fetches and decodes tiles off the main thread.
//
// Hires tiles arrive gzip-encoded over HTTP, so the browser inflates them;
// the worker only checks the header. Lowres PNGs are decoded to raw RGBA
// here, along with a hires-tile mask and a coarse height grid.

export type Request =
  | { kind: 'hires'; id: number; url: string }
  | { kind: 'lowres'; id: number; url: string }
  | { kind: 'abort'; id: number }

export type Response =
  | { kind: 'hires'; id: number; buf: ArrayBuffer | null; bytes: number }
  | {
      kind: 'lowres'
      id: number
      rgba: Uint8Array | null
      mask: [number, number]
      coarse: Uint16Array | null
      bytes: number
    }
  | { kind: 'error'; id: number; message: string }

const SIZE = 512
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
      post(empty(req))
      return
    }
    if (!res.ok) throw new Error(`${res.status} ${res.statusText}`)
    if (req.kind === 'hires') {
      const buf = await res.arrayBuffer()
      const magic = new Uint8Array(buf, 0, 4)
      if (String.fromCharCode(...magic) !== 'TMH1') throw new Error('bad tile header')
      post({ kind: 'hires', id: req.id, buf, bytes: buf.byteLength }, [buf])
    } else {
      const blob = await res.blob()
      const out = await decodeLowres(blob)
      post({ kind: 'lowres', id: req.id, ...out, bytes: blob.size }, [out.rgba.buffer, out.coarse.buffer])
    }
  } catch (err) {
    if ((err as Error).name === 'AbortError') return
    post({ kind: 'error', id: req.id, message: String(err) })
  } finally {
    inflight.delete(req.id)
  }
}

function empty(req: Request): Response {
  if (req.kind === 'hires') return { kind: 'hires', id: req.id, buf: null, bytes: 0 }
  return { kind: 'lowres', id: req.id, rgba: null, mask: [0, 0], coarse: null, bytes: 0 }
}

function post(msg: Response, transfer: Transferable[] = []) {
  ;(self as unknown as Worker).postMessage(msg, transfer)
}

async function decodeLowres(blob: Blob) {
  const bmp = await createImageBitmap(blob, { premultiplyAlpha: 'none', colorSpaceConversion: 'none' })
  const canvas = new OffscreenCanvas(bmp.width, bmp.height)
  const ctx = canvas.getContext('2d', { willReadFrequently: true })!
  ctx.drawImage(bmp, 0, 0)
  bmp.close()
  const rgba = new Uint8Array(ctx.getImageData(0, 0, SIZE, SIZE * 2).data.buffer)
  // Mask: which 64×64 blocks have data. Coarse: max height per 8×8 pixels.
  const mask: [number, number] = [0, 0]
  const coarse = new Uint16Array(64 * 64)
  const base = SIZE * SIZE * 4
  for (let z = 0; z < SIZE; z++) {
    for (let x = 0; x < SIZE; x++) {
      const i = base + (z * SIZE + x) * 4
      const h = (rgba[i] << 8) | rgba[i + 1]
      if (h === 0) continue
      const bit = (z >> 6) * 8 + (x >> 6)
      mask[bit >> 5] |= 1 << (bit & 31)
      const c = (z >> 3) * 64 + (x >> 3)
      if (h > coarse[c]) coarse[c] = h
    }
  }
  mask[0] >>>= 0
  mask[1] >>>= 0
  return { rgba, mask, coarse }
}
