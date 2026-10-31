#!/usr/bin/env node
// Drives the viewer in headless Chrome on a real GPU and reports load, frame
// and GPU numbers as JSON. It instruments WebGL and rAF from outside the app, so
// the same run measures any engine that keeps window.tealmap.goTo() and .debug().
//
//   node bench/viewer.mjs --url http://127.0.0.1:8200 --map overworld \
//     --views 0,0,900,35,50 [--dpr 1] [--out dir] [--label base] [--warm]

import { spawn } from 'node:child_process'
import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { parseArgs } from 'node:util'

const { values: args } = parseArgs({
  options: {
    url: { type: 'string', default: 'http://127.0.0.1:8200' },
    map: { type: 'string', default: 'overworld' },
    // Views as x,z,dist,azimuth,polar separated by ';'. The first one is the cold load.
    views: { type: 'string', default: '0,0,900,35,50' },
    dpr: { type: 'string', default: '1' },
    width: { type: 'string', default: '1920' },
    height: { type: 'string', default: '1080' },
    out: { type: 'string' },
    label: { type: 'string', default: 'run' },
    frames: { type: 'string', default: '180' },
    settle: { type: 'string', default: '120000' },
    software: { type: 'boolean', default: false },
    warm: { type: 'boolean', default: false },
    nopath: { type: 'boolean', default: false },
    // Extra query parameters for the viewer, like farpx=2&msaa=0.
    query: { type: 'string', default: '' },
  },
})

const views = args.views.split(';').map((v) => v.split(',').map(Number))
const gpuFlags = args.software
  ? ['--ignore-gpu-blocklist']
  : ['--use-angle=gl-egl', '--use-gl=angle', '--ignore-gpu-blocklist']

// Runs in the page before anything else. Wraps the WebGL2 context so every frame
// (one rAF callback) records its CPU time, GPU time and GL call counts.
const instrument = () => {
  const P = (window.__bench = {
    frames: [],
    current: null,
    longTasks: [],
    uploadBytes: 0,
    uploadMs: 0,
    recording: false,
    ext: null,
    gl: null,
    queries: [],
    gpu: [],
    pixel: new Uint8Array(4),
  })
  new PerformanceObserver((l) => {
    for (const e of l.getEntries()) P.longTasks.push({ start: e.startTime, ms: e.duration })
  }).observe({ type: 'longtask', buffered: true })

  const newFrame = () => ({ calls: 0, tris: 0, instances: 0, uniforms: 0, binds: 0, programs: 0, buffers: 0, states: 0 })
  const count = (proto, names, field) => {
    for (const n of names) {
      const f = proto[n]
      if (!f) continue
      proto[n] = function (...a) {
        if (P.current) P.current[field]++
        return f.apply(this, a)
      }
    }
  }
  const proto = WebGL2RenderingContext.prototype
  const triangles = (mode, n) => (mode === 4 ? n / 3 : mode === 5 ? Math.max(0, n - 2) : 0)
  const wrapDraw = (name, tris) => {
    const f = proto[name]
    proto[name] = function (...a) {
      if (P.current) {
        P.current.calls++
        P.current.tris += tris(a)
      }
      return f.apply(this, a)
    }
  }
  wrapDraw('drawArrays', (a) => triangles(a[0], a[2]))
  wrapDraw('drawElements', (a) => triangles(a[0], a[1]))
  wrapDraw('drawArraysInstanced', (a) => triangles(a[0], a[2]) * a[3])
  wrapDraw('drawElementsInstanced', (a) => triangles(a[0], a[1]) * a[4])
  wrapDraw('drawRangeElements', (a) => triangles(a[0], a[3]))
  count(proto, ['uniform1f', 'uniform1i', 'uniform1ui', 'uniform2f', 'uniform2fv', 'uniform3fv', 'uniform4fv', 'uniform1fv', 'uniform1iv', 'uniformMatrix3fv', 'uniformMatrix4fv', 'uniform2ui', 'uniform3f', 'uniform4f'], 'uniforms')
  count(proto, ['bindTexture', 'activeTexture'], 'binds')
  count(proto, ['useProgram'], 'programs')
  count(proto, ['bindBuffer', 'bindVertexArray', 'bindBufferBase', 'bindBufferRange'], 'buffers')
  count(proto, ['enable', 'disable', 'depthMask', 'blendFunc', 'blendFuncSeparate', 'cullFace', 'frontFace'], 'states')

  // Texture and buffer uploads, timed: this is the stall a burst of tiles causes.
  const wrapUpload = (name) => {
    const f = proto[name]
    if (!f) return
    proto[name] = function (...a) {
      const t = performance.now()
      const r = f.apply(this, a)
      P.uploadMs += performance.now() - t
      const buf = a.find((x) => ArrayBuffer.isView(x))
      if (buf) P.uploadBytes += buf.byteLength
      return r
    }
  }
  for (const n of ['texImage2D', 'texImage3D', 'texSubImage2D', 'texSubImage3D', 'texStorage2D', 'texStorage3D', 'bufferData', 'bufferSubData']) wrapUpload(n)

  // The timer belongs to the harness; apps see none, so queries don't nest.
  const getExtension = proto.getExtension
  proto.getExtension = function (name) {
    if (name === 'WEBGL_multi_draw') {
      const e = getExtension.call(this, name)
      if (!e) return e
      const wrap = (fn, tris) => {
        const f = e[fn]
        e[fn] = function (...a) {
          if (P.current) {
            P.current.calls++
            P.current.tris += tris(a)
          }
          return f.apply(e, a)
        }
      }
      const sum = (arr, off, n) => {
        let s = 0
        for (let i = 0; i < n; i++) s += arr[off + i]
        return s
      }
      wrap('multiDrawArraysWEBGL', (a) => triangles(a[0], sum(a[3], a[4], a[5])))
      wrap('multiDrawElementsWEBGL', (a) => triangles(a[0], sum(a[1], a[2], a[6])))
      wrap('multiDrawArraysInstancedWEBGL', (a) => {
        let s = 0
        for (let i = 0; i < a[7]; i++) s += triangles(a[0], a[3][a[4] + i]) * a[5][a[6] + i]
        return s
      })
      wrap('multiDrawElementsInstancedWEBGL', (a) => {
        let s = 0
        for (let i = 0; i < a[8]; i++) s += triangles(a[0], a[1][a[2] + i]) * a[6][a[7] + i]
        return s
      })
      return e
    }
    if (name === 'EXT_disjoint_timer_query_webgl2') return null
    return getExtension.call(this, name)
  }
  const realGetContext = HTMLCanvasElement.prototype.getContext
  HTMLCanvasElement.prototype.getContext = function (type, opts) {
    const gl = realGetContext.call(this, type, opts)
    if (gl && type === 'webgl2' && !P.gl) {
      P.gl = gl
      P.ext = getExtension.call(gl, 'EXT_disjoint_timer_query_webgl2')
    }
    return gl
  }

  const collect = () => {
    const gl = P.gl
    if (!gl || !P.ext) return
    if (gl.getParameter(P.ext.GPU_DISJOINT_EXT)) {
      P.queries.forEach((q) => gl.deleteQuery(q))
      P.queries = []
      return
    }
    while (P.queries.length && gl.getQueryParameter(P.queries[0], gl.QUERY_RESULT_AVAILABLE)) {
      const q = P.queries.shift()
      P.gpu.push(gl.getQueryParameter(q, gl.QUERY_RESULT) / 1e6)
      gl.deleteQuery(q)
    }
  }

  const raf = window.requestAnimationFrame.bind(window)
  window.requestAnimationFrame = (cb) =>
    raf((now) => {
      const f = newFrame()
      P.current = f
      let q = null
      if (P.gl && P.ext) {
        collect()
        q = P.gl.createQuery()
        P.gl.beginQuery(P.ext.TIME_ELAPSED_EXT, q)
      }
      const t = performance.now()
      try {
        cb(now)
      } finally {
        f.ms = performance.now() - t
        if (q) {
          P.gl.endQuery(P.ext.TIME_ELAPSED_EXT)
          // Only frames that draw count; this also skips the harness's own rAFs.
          if (f.calls > 0) P.queries.push(q)
          else P.gl.deleteQuery(q)
        }
        P.current = null
        if (f.calls > 0) {
          // Waiting for the GPU keeps frames from queueing up and stalling in bursts,
          // so each frame's time is its own.
          P.gl.readPixels(0, 0, 1, 1, P.gl.RGBA, P.gl.UNSIGNED_BYTE, P.pixel)
          f.total = performance.now() - t
          f.at = t
          if (P.recording) P.frames.push(f)
        }
      }
    })
}

function launch(profile) {
  const chrome = spawn(
    'google-chrome',
    [
      '--headless=new',
      '--no-sandbox',
      '--remote-debugging-port=0',
      `--user-data-dir=${profile}`,
      '--no-first-run',
      '--disable-gpu-vsync',
      '--disable-frame-rate-limit',
      '--disable-background-timer-throttling',
      '--disable-renderer-backgrounding',
      '--disable-backgrounding-occluded-windows',
      `--window-size=${args.width},${args.height}`,
      `--force-device-scale-factor=${args.dpr}`,
      ...gpuFlags,
      'about:blank',
    ],
    { stdio: ['ignore', 'ignore', 'pipe'] },
  )
  const ws = new Promise((resolve, reject) => {
    let b = ''
    chrome.stderr.on('data', (d) => {
      b += d
      const m = b.match(/ws:\/\/[^\s]+/)
      if (m) resolve(m[0])
    })
    setTimeout(() => reject(new Error(`chrome did not start: ${b}`)), 20000)
  })
  return { chrome, ws }
}

async function connect(browserWs) {
  const port = new URL(browserWs).port
  const target = (await (await fetch(`http://127.0.0.1:${port}/json/list`)).json()).find((t) => t.type === 'page')
  const ws = new WebSocket(target.webSocketDebuggerUrl)
  await new Promise((r) => (ws.onopen = r))
  let id = 0
  const pending = new Map()
  const handlers = []
  ws.onmessage = (e) => {
    const m = JSON.parse(e.data)
    if (m.id && pending.has(m.id)) {
      const { resolve, reject } = pending.get(m.id)
      pending.delete(m.id)
      if (m.error) reject(new Error(JSON.stringify(m.error)))
      else resolve(m.result)
    } else for (const h of handlers) h(m)
  }
  const send = (method, params = {}, sessionId) =>
    new Promise((resolve, reject) => {
      const i = ++id
      pending.set(i, { resolve, reject })
      ws.send(JSON.stringify({ id: i, method, params, ...(sessionId ? { sessionId } : {}) }))
    })
  return { send, on: (h) => handlers.push(h), close: () => ws.close() }
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms))
const pct = (a, p) => {
  if (!a.length) return null
  const s = [...a].sort((x, y) => x - y)
  return s[Math.min(s.length - 1, Math.floor(s.length * p))]
}
const round = (v, d = 2) => (v === null || v === undefined ? v : Number(v.toFixed(d)))
const summarize = (a) => ({ mean: round(a.reduce((x, y) => x + y, 0) / Math.max(a.length, 1)), p50: round(pct(a, 0.5)), p95: round(pct(a, 0.95)), p99: round(pct(a, 0.99)), max: round(a.length ? Math.max(...a) : null) })

async function main() {
  const profile = mkdtempSync(join(tmpdir(), 'tealmap-bench-'))
  const { chrome, ws: wsUrl } = launch(profile)
  const cdp = await connect(await wsUrl)
  const { send } = cdp
  const net = { requests: 0, bytes: 0, tiles: 0, tileBytes: 0, inflight: new Set(), byUrl: new Map() }

  const onNet = (m) => {
    if (m.method === 'Network.requestWillBeSent') {
      net.requests++
      // The live-update stream never finishes.
      if (!m.params.request.url.includes('/api/events')) net.inflight.add(m.params.requestId)
      net.byUrl.set(m.params.requestId, m.params.request.url)
    }
    if (m.method === 'Network.loadingFinished' || m.method === 'Network.loadingFailed') {
      net.inflight.delete(m.params.requestId)
      const bytes = m.params.encodedDataLength ?? 0
      net.bytes += bytes
      const url = net.byUrl.get(m.params.requestId) ?? ''
      if (url.includes('.tmh') || url.includes('/tiles')) {
        net.tiles++
        net.tileBytes += bytes
      }
    }
    if (m.method === 'Target.attachedToTarget') {
      send('Network.enable', {}, m.params.sessionId).catch(() => {})
    }
  }
  cdp.on(onNet)
  await send('Target.setAutoAttach', { autoAttach: true, flatten: true, waitForDebuggerOnStart: false })
  await send('Network.enable')
  await send('Page.enable')
  await send('Runtime.enable')
  await send('Page.addScriptToEvaluateOnNewDocument', { source: `(${instrument.toString()})()` })

  const ev = async (expression) => {
    const r = await send('Runtime.evaluate', { expression, awaitPromise: true, returnByValue: true })
    if (r.exceptionDetails) throw new Error(r.exceptionDetails.exception?.description ?? r.exceptionDetails.text)
    return r.result.value
  }

  const isSettled = async () =>
    ev(`(() => {
      const t = window.tealmap
      if (!t || !t.manifest) return false
      const d = t.debug()
      let loading = 0, ready = 0
      for (const k of Object.keys(d)) if (k[0] === 'L') { loading += d[k].loading ?? 0; ready += d[k].ready ?? 0 }
      const queued = t.uploads?.length ?? t.uploadQueueLength ?? 0
      return loading === 0 && queued === 0 && ready > 0
    })()`)

  // Waits until no tile has been loading for a while. Returns the time spent.
  const settle = async (limit = Number(args.settle)) => {
    const t0 = performance.now()
    let quiet = 0
    while (performance.now() - t0 < limit) {
      if ((await isSettled()) && net.inflight.size === 0) {
        if (++quiet >= 8) return performance.now() - t0 - 8 * 125
      } else quiet = 0
      await sleep(125)
    }
    return null
  }

  const result = { label: args.label, dpr: Number(args.dpr), size: [Number(args.width), Number(args.height)] }
  const [x0, z0, d0, a0, p0] = views[0]
  const url = `${args.url}/?bench${args.query ? `&${args.query}` : ''}#${args.map}/${x0}/${z0}/${d0}/${a0}/${p0}`
  const t0 = performance.now()
  await send('Page.navigate', { url })

  if (args.warm) {
    await settle()
    await send('Page.reload', { ignoreCache: false })
    net.requests = net.bytes = net.tiles = net.tileBytes = 0
    await sleep(100)
  }
  const tLoad = args.warm ? performance.now() : t0
  const settled = await settle()
  result.load = {
    settledMs: settled === null ? null : round(performance.now() - tLoad - 1000, 0),
    requests: net.requests,
    wireMB: round(net.bytes / 1e6),
    tiles: net.tiles,
    tileMB: round(net.tileBytes / 1e6),
    ...(await ev(`(() => {
      const P = window.__bench
      const tasks = P.longTasks
      return {
        longTasks: tasks.length,
        longestTaskMs: Math.round(Math.max(0, ...tasks.map((t) => t.ms))),
        blockingMs: Math.round(tasks.reduce((a, t) => a + Math.max(0, t.ms - 50), 0)),
        uploadMB: +(P.uploadBytes / 1e6).toFixed(1),
        uploadMs: Math.round(P.uploadMs),
        heapMB: +((performance.memory?.usedJSHeapSize ?? 0) / 1e6).toFixed(0),
      }
    })()`)),
    debug: await ev('JSON.stringify(window.tealmap?.debug?.() ?? {})').then(JSON.parse),
  }

  const shoot = async (name) => {
    if (!args.out) return
    mkdirSync(args.out, { recursive: true })
    const { data } = await send('Page.captureScreenshot', { format: 'png' })
    writeFileSync(join(args.out, `${args.label}-${name}.png`), Buffer.from(data, 'base64'))
  }

  // Records n frames, nudging the camera so an on-demand engine keeps drawing.
  const record = async (n, driver) =>
    ev(`(async () => {
      const P = window.__bench
      P.gpu = []
      P.frames = []
      P.recording = true
      const t = window.tealmap
      const driver = ${driver}
      for (let i = 0; i < ${n}; i++) {
        driver(i)
        await new Promise((r) => requestAnimationFrame(r))
      }
      // Let the last GPU queries come back.
      for (let i = 0; i < 12; i++) await new Promise((r) => requestAnimationFrame(r))
      P.recording = false
      const f = P.frames
      return { frames: f, gpu: P.gpu }
    })()`)

  const report = (raw) => {
    const f = raw.frames
    const intervals = []
    for (let i = 1; i < f.length; i++) intervals.push(f[i].at - f[i - 1].at)
    const avg = (k) => round(f.reduce((a, x) => a + x[k], 0) / Math.max(f.length, 1), 1)
    return {
      drew: f.length,
      cpuMs: summarize(f.map((x) => x.ms)),
      totalMs: summarize(f.map((x) => x.total)),
      gpuMs: summarize(raw.gpu),
      intervalMs: summarize(intervals),
      calls: avg('calls'),
      trisM: round(f.reduce((a, x) => a + x.tris, 0) / Math.max(f.length, 1) / 1e6),
      uniforms: avg('uniforms'),
      binds: avg('binds'),
      programs: avg('programs'),
      buffers: avg('buffers'),
    }
  }

  result.views = []
  for (let i = 0; i < views.length; i++) {
    const [x, z, d, a, p] = views[i]
    const set = `{x:${x},z:${z},dist:${d},az:${a},polar:${p}}`
    const before = net.bytes
    const tv = performance.now()
    await ev(`window.tealmap.goTo(${set})`)
    const s = await settle()
    await sleep(600)
    await shoot(`view${i}`)
    const raw = await record(Number(args.frames), `(i) => { t.goTo({x:${x},z:${z},dist:${d},az:${a} + (i % 2) * 0.002,polar:${p}}) }`)
    result.views.push({ view: views[i], settleMs: s === null ? null : round(s, 0), wireMB: round((net.bytes - before) / 1e6), ...report(raw) })
  }

  // A flight over the map: tiles load while it moves, which is what people feel.
  if (!args.nopath) {
    const [x, z] = views[0]
    await ev(`(window.__bench.longTasks = [], window.__bench.uploadMs = 0, window.__bench.uploadBytes = 0, 0)`)
    const before = net.bytes
    const raw = await record(
      900,
      `(i) => {
        const k = i / 900
        const s = k * Math.PI * 2
        const dist = 400 + 2600 * (0.5 + 0.5 * Math.cos(s))
        t.goTo({ x: ${x} + k * 6000, z: ${z} + 1500 * Math.sin(s), dist, az: 35 + 90 * k, polar: 40 + 15 * Math.sin(s * 2) })
      }`,
    )
    result.flight = {
      ...report(raw),
      wireMB: round((net.bytes - before) / 1e6),
      ...(await ev(`(() => {
        const P = window.__bench
        return {
          longTasks: P.longTasks.length,
          longestTaskMs: Math.round(Math.max(0, ...P.longTasks.map((t) => t.ms))),
          blockingMs: Math.round(P.longTasks.reduce((a, t) => a + Math.max(0, t.ms - 50), 0)),
          uploadMB: +(P.uploadBytes / 1e6).toFixed(1),
          uploadMs: Math.round(P.uploadMs),
        }
      })()`)),
    }
  }

  result.heapMB = await ev('+((performance.memory?.usedJSHeapSize ?? 0) / 1e6).toFixed(0)')
  result.renderer = await ev(`(() => { const g = window.__bench.gl; const d = g.getExtension('WEBGL_debug_renderer_info'); return d ? g.getParameter(d.UNMASKED_RENDERER_WEBGL) : null })()`)
  console.log(JSON.stringify(result, null, 1))
  cdp.close()
  chrome.kill()
  await sleep(300)
  rmSync(profile, { recursive: true, force: true })
  process.exit(0)
}

main().catch((e) => {
  console.error(e)
  process.exit(1)
})
