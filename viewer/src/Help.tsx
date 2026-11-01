/** What the map's controls are, for whichever kind of pointer this device has. */
export function Help() {
  const touch = matchMedia('(pointer: coarse)').matches
  return (
    <dl className="help">
      {touch ? (
        <>
          <h2>Touch</h2>
          <dt>Drag</dt>
          <dd>Pan</dd>
          <dt>Pinch</dt>
          <dd>Zoom</dd>
          <dt>Twist</dt>
          <dd>Rotate</dd>
          <dt>Two-finger drag</dt>
          <dd>Rotate and tilt</dd>
          <dt>Tap</dt>
          <dd>Pin a block</dd>
        </>
      ) : (
        <>
          <h2>Mouse</h2>
          <dt>Drag</dt>
          <dd>Pan</dd>
          <dt>Right-drag</dt>
          <dd>Rotate and tilt</dd>
          <dt>Scroll</dt>
          <dd>Zoom</dd>
          <dt>Click</dt>
          <dd>Pin a block</dd>
        </>
      )}
      <h2>Keyboard</h2>
      <dt>W A S D, arrows</dt>
      <dd>Pan</dd>
      <dt>Q E</dt>
      <dd>Rotate</dd>
      <dt>R F</dt>
      <dd>Tilt</dd>
      <dt>+ −</dt>
      <dd>Zoom</dd>
      <dt>Esc</dt>
      <dd>Remove pin</dd>
    </dl>
  )
}
