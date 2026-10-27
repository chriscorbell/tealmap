// GLSL for hires block meshes and lowres heightfields. Colors stay in
// texture space end to end, the way Minecraft shades blocks.

const fog = /* glsl */ `
  uniform vec3 uFogColor;
  uniform vec2 uFogRange;
  float fogFactor(float dist) {
    return smoothstep(uFogRange.x, uFogRange.y, dist);
  }
`

export const hiresVertex = /* glsl */ `
  in vec3 uvl;
  in vec4 tmColor;
  out vec3 vUv;
  out vec3 vColor;
  out float vDist;
  void main() {
    vec4 mv = modelViewMatrix * vec4(position, 1.0);
    gl_Position = projectionMatrix * mv;
    vUv = vec3(uvl.xy / 4096.0, uvl.z);
    vColor = tmColor.rgb;
    vDist = length(mv.xyz);
  }
`

export const hiresFragment = /* glsl */ `
  precision highp sampler2DArray;
  uniform sampler2DArray uTex;
  uniform float uAppear;
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
    // Dissolve in: a stable per-pixel threshold, so no transparency sorting.
    if (uAppear < 1.0) {
      float n = fract(52.9829189 * fract(dot(gl_FragCoord.xy, vec2(0.06711056, 0.00583715))));
      if (n >= uAppear) discard;
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

// Lowres tiles: a shared grid in pixel units, displaced by the height half of
// the tile texture. Pixels under loaded finer tiles are discarded.
const heightFns = /* glsl */ `
  uniform sampler2D uTex;
  float heightAt(ivec2 p) {
    p = clamp(p, ivec2(0), ivec2(511));
    vec2 rg = texelFetch(uTex, ivec2(p.x, p.y + 512), 0).rg;
    return rg.r * 65280.0 + rg.g * 255.0;
  }
`

export const lowresVertex = /* glsl */ `
  uniform float uMinY;
  ${heightFns}
  out vec2 vPx;
  out float vDist;
  void main() {
    ivec2 p = ivec2(position.xz);
    float h = heightAt(p);
    if (h == 0.0) {
      // Borrow a neighbor's height so edges don't plunge to the world floor.
      h = max(max(heightAt(p + ivec2(-1, 0)), heightAt(p + ivec2(0, -1))),
              max(heightAt(p + ivec2(-1, -1)), heightAt(p + ivec2(1, 1))));
    }
    vec4 world = modelMatrix * vec4(position.x, 0.0, position.z, 1.0);
    world.y = h > 0.0 ? h + uMinY - 1.0 : uMinY;
    vec4 mv = viewMatrix * world;
    gl_Position = projectionMatrix * mv;
    vPx = position.xz;
    vDist = length(mv.xyz);
  }
`

export const lowresFragment = /* glsl */ `
  uniform uvec2 uMask;
  uniform float uScale;
  uniform float uAppear;
  ${heightFns}
  ${fog}
  in vec2 vPx;
  in float vDist;
  out vec4 fragColor;
  void main() {
    ivec2 p = clamp(ivec2(floor(vPx)), ivec2(0), ivec2(511));
    float h = heightAt(p);
    if (h == 0.0) discard;
    ivec2 cell = p / 64;
    int bit = cell.y * 8 + cell.x;
    uint m = bit < 32 ? uMask.x : uMask.y;
    if (((m >> uint(bit & 31)) & 1u) != 0u) discard;
    vec3 c = texelFetch(uTex, p, 0).rgb;
    float hl = heightAt(p + ivec2(-1, 0)); hl = hl == 0.0 ? h : hl;
    float hr = heightAt(p + ivec2(1, 0));  hr = hr == 0.0 ? h : hr;
    float hu = heightAt(p + ivec2(0, -1)); hu = hu == 0.0 ? h : hu;
    float hd = heightAt(p + ivec2(0, 1));  hd = hd == 0.0 ? h : hd;
    vec3 n = normalize(vec3(hl - hr, 2.0 * uScale, hu - hd));
    float shade = 0.62 + 0.38 * clamp(dot(n, normalize(vec3(-0.35, 1.0, -0.55))), 0.0, 1.0);
    c *= shade;
    c = mix(c, uFogColor, max(fogFactor(vDist), 1.0 - uAppear));
    fragColor = vec4(c, 1.0);
  }
`
