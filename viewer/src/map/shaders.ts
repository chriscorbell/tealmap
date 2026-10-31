// GLSL for tiles, hires and LOD alike. Colors stay in texture space end to end,
// the way Minecraft shades blocks.

// Vertex pulling: each quad is two words in uQuads (block anchor and merged
// width; template, color and merged height). Quad q's four corners are vertices
// 4q..4q+3 through a shared index buffer, and each corner reads its offset and
// UV from the template. A merged quad stretches its far corners along the two
// axes its face lies in, and its texture repeats across them.
//
// Tile tables hold colors below their base, counting down, and templates above it.
// The per-draw table says where a draw's quads and tables are, its origin and
// scale, and its dissolve.
export const vertexShader = /* glsl */ `
  precision highp float;
  precision highp int;
  precision highp usampler2D;
  precision highp sampler2D;
  uniform usampler2D uQuads;
  uniform usampler2D uTables;
  uniform sampler2D uDraws;
  uniform mat4 uViewProj;
  uniform vec3 uCam;
  uniform int uFirstDraw;
  out vec2 vUv;
  flat out float vLayer;
  flat out vec3 vColor;
  #ifdef FADE
    flat out vec2 vDissolve;
  #endif
  #ifdef NO_MULTI_DRAW
    #define DRAW_ID 0
  #else
    #define DRAW_ID gl_DrawID
  #endif
  void main() {
    int id = uFirstDraw + DRAW_ID;
    ivec2 at = ivec2((id & 63) * 2, id >> 6);
    vec4 d0 = texelFetch(uDraws, at, 0);
    vec4 d1 = texelFetch(uDraws, at + ivec2(1, 0), 0);
    int q = int(d1.x) + (gl_VertexID >> 2);
    int c = gl_VertexID & 3;
    uvec2 w = texelFetch(uQuads, ivec2(q & 2047, q >> 11), 0).xy;
    vec3 anchor = vec3(float(w.x & 127u), float((w.x >> 14) & 1023u), float((w.x >> 7) & 127u));
    int base = int(d1.y);
    int ti = base + int(w.y & 8191u) * 6;
    int tc = ti + c;
    uvec4 corner = texelFetch(uTables, ivec2(tc & 1023, tc >> 10), 0);
    int tu = ti + 4 + (c >> 1);
    uvec4 uvs = texelFetch(uTables, ivec2(tu & 1023, tu >> 10), 0);
    vec2 uv = (c & 1) == 0 ? vec2(uvs.xy) : vec2(uvs.zw);
    float farU = float((corner.x >> 14) & 1u) * float((w.x >> 24) & 63u);
    float farV = float(corner.x >> 15) * float(w.y >> 26);
    uint axes = (corner.x >> 12) & 3u;
    vec3 stretch = axes == 0u ? vec3(farU, 0.0, farV) : axes == 1u ? vec3(farU, farV, 0.0) : vec3(0.0, farV, farU);
    vec3 pos = anchor * 64.0 + vec3(corner.xyz & 4095u) - 64.0 + stretch * 64.0;
    uv += (vec2((corner.y >> 12) & 3u, corner.y >> 14) - 1.0) * farU * 4096.0;
    uv += (vec2((corner.z >> 12) & 3u, corner.z >> 14) - 1.0) * farV * 4096.0;
    // Relative to the camera, so large coordinates keep their precision.
    vec3 rel = (d0.xyz - uCam) + pos * d0.w;
    gl_Position = uViewProj * vec4(rel, 1.0);
    vUv = uv / 4096.0;
    vLayer = float(corner.w);
    int tp = base - 1 - int((w.y >> 13) & 8191u);
    vColor = vec3(texelFetch(uTables, ivec2(tp & 1023, tp >> 10), 0).rgb) / 255.0;
    #ifdef FADE
      vDissolve = d1.zw;
    #endif
  }
`

export const fragmentShader = /* glsl */ `
  precision highp float;
  precision highp sampler2DArray;
  uniform sampler2DArray uTex;
  uniform vec3 uFogColor;
  uniform vec2 uFogRange;
  in vec2 vUv;
  flat in float vLayer;
  flat in vec3 vColor;
  #ifdef FADE
    flat in vec2 vDissolve;
  #endif
  out vec4 fragColor;
  void main() {
    vec4 t = texture(uTex, vec3(vUv, vLayer));
    #ifdef CUTOUT
      if (t.a < 0.4) discard;
    #endif
    // Dissolve: a stable per-pixel threshold, so no transparency sorting. A tile
    // coming in draws pixels below appear; one going out drops pixels below fade,
    // so during a swap each pixel belongs to exactly one of them. Settled opaque
    // tiles use a program without this, because any discard costs early depth writes.
    #ifdef FADE
      if (vDissolve.x < 1.0 || vDissolve.y > 0.0) {
        float n = fract(52.9829189 * fract(dot(gl_FragCoord.xy, vec2(0.06711056, 0.00583715))));
        if (n >= vDissolve.x || n < vDissolve.y) discard;
      }
    #endif
    // Distance along the view direction: 1 / w, which the rasterizer already has.
    vec3 c = mix(t.rgb * vColor, uFogColor, smoothstep(uFogRange.x, uFogRange.y, 1.0 / gl_FragCoord.w));
    #ifdef TRANSLUCENT
      fragColor = vec4(c, t.a);
    #else
      fragColor = vec4(c, 1.0);
    #endif
  }
`
