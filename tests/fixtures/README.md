Small region files for tests. `subregion.py` cut the first two from real worlds:

- `26.2/region/`: a 4×4-chunk corner of each of the four regions that meet at
  the origin of a Minecraft 26.2 (Fabric, Terralith) world. Tiles that cross
  region files and negative coordinates are covered.
- `1.21.1/region/r.0.-1.mca`: chunks 8–11 × 8–11 of a 1.21.1 (NeoForge,
  Create) world, at negative z.
- `nether/region/r.0.0.mca`: a 4×4-chunk Nether made up by `nether/make.py`, with a
  bedrock roof at y=127, no saved sky light, and caves and lava under the roof.
