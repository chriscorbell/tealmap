# tealmap plan

Agreed by the design council (Claude Opus 5.5, GPT-6.1 Sol) on 2026-10-26.

## Stack
- **Renderer + server:** one Rust binary `tealmap` (edition 2024), run as a sidecar container. It reads the world read-only and never runs in the Minecraft JVM. No FFM.
  - Subcommands: `render` (full or incremental) and `serve` (HTTP + watch + incremental re-render).
  - Crates: a small custom NBT reader that skips unused tags; flate2 with the zlib-rs backend; lz4_flex; rayon; axum/tokio; notify with a poll fallback; zip; serde_json; png.
- **Storage:** flat files, pre-gzipped. Each write goes to a temp file and is renamed into place. axum serves them with `Content-Encoding: gzip` and an ETag. Reason: simple, static-hostable, and with 64×64 tiles the inode count stays reasonable. No SQLite.
- **Mesh format:** a small versioned little-endian binary.
  - Header + u16-quantized positions in 1/64-block steps, measured from the tile's min corner minus a 1-block margin and from the dimension's `min_y` (model elements can reach past their block, and rotated elements need finer than 1/16) + u16 UVs + u16 texture layer + packed normal/light/tint + u16/u32 indices.
  - Separate opaque, cutout, and translucent groups. gzip only. No meshopt or Draco in the MVP.
- **Viewer:** pnpm + Vite + React + TS + three.js (WebGL2), embedded in the binary.
  - React handles the UI chrome only, never the render loop.
  - A worker does fetch and decode, then transfers buffers. Textures live in a `DataArrayTexture` with mipmaps, drawn by a custom ShaderMaterial.
  - Lucide icons, dark, restrained, light motion.
- **CI/CD:** GitHub Actions runs cargo fmt/clippy/test and pnpm typecheck/build. A multi-stage Docker build pushes the image to GHCR. A Compose stack on minicore mounts the Crafty world dirs read-only, sets a `cpus:` limit, and listens on a new port behind Tailscale Serve, next to BlueMap.

## MVP scope
- **Config:** TOML listing maps. Each has an id, a name, an explicit dimension dir (works for both the 26.x `dimensions/minecraft/<dim>` and the 1.21.1 `world/region` layouts), and a Minecraft version. Targets: VoidCraft (26.2) and Create (1.21.1) overworlds.
- **Assets:** the matching client jar is downloaded from Mojang's piston-meta on first run, checked by SHA-1, and cached in the data dir. Never redistributed.
- **Generic JSON model resolver:**
  - blockstate variants (deterministic weighted pick) and multipart
  - parent inheritance and texture refs
  - element from/to and element rotation
  - variant x/y and uvlock
  - cullface, and tintindex mapped to grass/foliage/water tint from biome colormaps
  - Unresolvable blocks become an averaged-color cube (magenta if no texture) and are counted in a coverage report.
- **Special cases:** water and lava as simple fluids with level-based height, and waterlogging. No block entities and no entities.
- **Lighting:** directional face shade plus the saved sky/block light when the chunk has it. Otherwise full bright.
- **Culling:** hidden-face culling across chunk and tile borders using a 1-block halo.
- **Hires tiles:** 64×64 blocks, full height.
- **Lowres:** 512×512-block tiles, at most 2 levels (1:1 and 1:8). Color is an RGB PNG. Height is a separate PNG with 16-bit values (R and G hold the high and low bytes), decoded with `premultiplyAlpha: 'none'` and no color-space conversion. The vertex shader displaces a grid with them.
- **Hires/lowres handoff:** the lowres shader discards fragments inside the footprint of loaded hires tiles, so the layers never z-fight or leave gaps.
- **Incremental:**
  - The watcher, plus a periodic reconcile pass, diffs region-header timestamps as a hint, then hashes each chunk's compressed payload to confirm a real change.
  - Only dirty tiles and the neighbors that border needs are re-rendered. Each changed chunk also marks its covering lowres tiles dirty at both levels, and they regenerate in the same pass.
  - Torn reads are validated, retried on the next pass, and the last-good tile is kept.
  - SSE notifies the viewer, which refetches the changed hires and lowres tiles.
- **Viewer:**
  - BlueMap-like map controls (pan/rotate/zoom).
  - Closest-first tile loading with cancellation, and frustum culling.
  - Hires radius with lowres beyond it, and bounded GPU residency.
  - A stats overlay (fps/frame ms, draw calls, triangles, bytes) and a minimal map switcher.
- **Out of scope:**
  - AO, greedy meshing, animated textures (first frame only)
  - block entities, entities, players, markers
  - any Fabric/NeoForge/Paper mod
  - modded assets, resource packs, auth
  - Nether/End-specific polish

## Done criteria
1. Unit and fixture tests on real 26.2 and 1.21.1 region data: negative coords, negative section Y, single-entry palettes, tile boundaries, and padded palette longs.
2. Determinism. The same input produces byte-identical tiles. After changing one chunk, only affected tiles change.
3. Screenshots of both worlds at ground level and from overview, with no holes or seams at hires tile borders or at the hires/lowres handoff, plus BlueMap screenshots at similar cameras for comparison.
4. A benchmark against BlueMap CLI (5.29 for VoidCraft, 5.7 for Create) on identical world snapshot copies on minicore with the same thread count. It records wall time, CPU-seconds, peak RSS, and output bytes, and reports fidelity differences honestly. The goal is faster and smaller, without claiming quality parity.
5. Viewer: frame times from the stats overlay, gzipped bundle size, and bytes transferred for a standard view.
6. CI green, image on GHCR, and the sidecar running on minicore at a new tailnet port.

## Build order

The order is set so that a cut at any point still leaves something usable. Deploy as soon as the first world renders.

1. Render one world (VoidCraft): region/NBT reader, assets and models, hires mesher, lowres.
2. Viewer.
3. CI, image, and deploy to minicore.
4. Second world (Create, 1.21.1).
5. Incremental updates and SSE.
6. Benchmark against BlueMap CLI.
