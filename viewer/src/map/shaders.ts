// GLSL for tile meshes, hires and LOD alike. Colors stay in texture space end
// to end, the way Minecraft shades blocks.

const fog = /* glsl */ `
  uniform vec3 uFogColor;
  uniform vec2 uFogRange;
  float fogFactor(float dist) {
    return smoothstep(uFogRange.x, uFogRange.y, dist);
  }
`

// Vertex pulling: each quad is two words in uQuads (block anchor; template and
// color indices). Quad q's four corners are vertices 4q..4q+3 through a shared
// index buffer, and each corner reads its offset and UV from the template.
export const hiresVertex = /* glsl */ `
  precision highp usampler2D;
  uniform highp usampler2D uQuads;
  uniform highp usampler2D uTpl;
  uniform sampler2D uPal;
  out vec3 vUv;
  out vec3 vColor;
  out float vDist;
  void main() {
    int q = gl_VertexID >> 2;
    int c = gl_VertexID & 3;
    uvec2 w = texelFetch(uQuads, ivec2(q & 2047, q >> 11), 0).xy;
    vec3 anchor = vec3(float(w.x & 127u), float((w.x >> 14) & 1023u), float((w.x >> 7) & 127u));
    int tid = int(w.y & 0xffffu);
    int ci = int(w.y >> 16);
    int ti = tid * 8 + c;
    uvec4 corner = texelFetch(uTpl, ivec2(ti & 4095, ti >> 12), 0);
    int tu = tid * 8 + 4 + (c >> 1);
    uvec4 uvs = texelFetch(uTpl, ivec2(tu & 4095, tu >> 12), 0);
    vec2 uv = (c & 1) == 0 ? vec2(uvs.xy) : vec2(uvs.zw);
    vec4 mv = modelViewMatrix * vec4(anchor * 64.0 + vec3(corner.xyz) - 64.0, 1.0);
    gl_Position = projectionMatrix * mv;
    vUv = vec3(uv / 4096.0, float(corner.w));
    vColor = texelFetch(uPal, ivec2(ci & 4095, ci >> 12), 0).rgb;
    vDist = length(mv.xyz);
  }
`

export const hiresFragment = /* glsl */ `
  precision highp sampler2DArray;
  uniform sampler2DArray uTex;
  uniform float uAppear;
  uniform float uFade;
  ${fog}
  in vec3 vUv;
  in vec3 vColor;
  in float vDist;
  out vec4 fragColor;
  void main() {
    vec4 t = texture(uTex, vUv);
    #ifdef CUTOUT
      if (t.a < 0.4) discard;
    #endif
    // Dissolve: a stable per-pixel threshold, so no transparency sorting. A tile
    // coming in draws pixels below uAppear; one going out drops pixels below
    // uFade, so during a swap each pixel belongs to exactly one of them.
    if (uAppear < 1.0 || uFade > 0.0) {
      float n = fract(52.9829189 * fract(dot(gl_FragCoord.xy, vec2(0.06711056, 0.00583715))));
      if (n >= uAppear || n < uFade) discard;
    }
    vec3 c = t.rgb * vColor;
    c = mix(c, uFogColor, fogFactor(vDist));
    #ifdef TRANSLUCENT
      fragColor = vec4(c, t.a);
    #else
      fragColor = vec4(c, 1.0);
    #endif
  }
`
