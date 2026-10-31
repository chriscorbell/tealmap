# tealmap

A 3D web map for Minecraft worlds, in the spirit of [BlueMap](https://github.com/BlueMap-Minecraft/BlueMap), built to be fast and light.

- **Zero server impact.** tealmap runs as its own process next to the server and reads region files directly from a read-only mount. Nothing runs inside the Minecraft JVM, so it can't slow ticks, grow the heap, or crash the server.
- **Fast renders.** It's a Rust renderer with a zero-copy NBT reader, palette-level block resolution, and one tile per worker thread.
- **Small tiles.** Each quad is 8 bytes: a block anchor plus indices into small per-tile tables of quad shapes and colors. Quads are column-coded so gzip gets them to about 1 byte each, and the GPU draws them straight from those records.
- **Full detail at any distance.** Past the hires tiles, a voxel LOD pyramid takes over: each level halves the resolution by picking the most common visible block in every 2×2×2, and meshes that with the same mesher at a larger scale. The viewer switches to a finer level where a voxel would cover more than 4 pixels on screen, so distant terrain keeps its real shapes, textures and water.
- **A light viewer.** The viewer is three.js on WebGL2. Tiles load in a worker, all block textures share one texture array, and frames render on demand.
- **Live updates.** tealmap watches region files and hashes each chunk's compressed bytes to find real changes. It re-renders only the tiles affected and pushes them to open viewers over SSE. A big pass renders in batches from the middle of the map outward, so the map fills in as it goes, and a restart picks up where the pass stopped.

## How it works

```
world/region/*.mca ──► tealmap render ──► data/maps/<id>/
  (read-only)          (Rust, rayon)        hires/<rx>_<rz>/<tx>_<tz>.tmh   64×64-block tiles (gzip)
                                            lod/<k>/<x>_<z>.tmh             LOD tiles, voxels of 2^k blocks
                                            lod/src<k>/<x>_<z>.bin          the grids each LOD level is built from
                                            textures.bin, map.json
                       tealmap serve ───► HTTP + SSE, viewer embedded in the binary
```

Block models and textures come from the Minecraft client jar for the configured version. tealmap downloads the jar from Mojang on first run and checks its SHA-1. It's never bundled or redistributed.

## Run it

```yaml
services:
  tealmap:
    image: ghcr.io/chriscorbell/tealmap:latest
    restart: unless-stopped
    ports:
      - "8200:8200"
    volumes:
      - ./tealmap.toml:/config/tealmap.toml:ro
      - ./data:/data
      - /path/to/server:/worlds/main:ro
```

Copy [`tealmap.example.toml`](tealmap.example.toml) to `tealmap.toml` and point each map's `world` at a dimension directory. To render once without serving, run `tealmap render`. Add `--full` to ignore saved state.

## Develop

```sh
cargo run --release -- -c tealmap.toml serve   # API on :8200
cd viewer && pnpm install && pnpm dev           # viewer with hot reload, proxies /api and /maps
```

Append `?bench` to the viewer URL to render every frame and show the stats panel. The URL hash (`#map/x/z/distance/azimuth/polar`) stores the camera.

## Supported

- Worlds from Minecraft 1.18 through 26.x. Both the `world/region` and `world/dimensions/<ns>/<dim>/region` layouts work.
- Vanilla block models: variants, multipart, parent models, element and variant rotations, uvlock, and biome tints. Mod jars can supply biome colors.
- Water and lava, waterlogging, saved sky and block light, and BlueMap-style cave removal.

Not yet supported: block entities (chests, signs, beds), entities and players, markers, ambient occlusion, animated textures, resource packs, and mod block models. Blocks without a model render as magenta-checkered cubes.

See [`docs/plan.md`](docs/plan.md) for the design and roadmap.
