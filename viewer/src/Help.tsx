const MOUSE = [
  ['Drag', 'Pan'],
  ['Right-drag', 'Rotate and tilt'],
  ['Scroll', 'Zoom'],
  ['Click', 'Pin a block'],
]

const TOUCH = [
  ['Drag', 'Pan'],
  ['Pinch', 'Zoom'],
  ['Twist', 'Rotate'],
  ['Two-finger drag', 'Rotate and tilt'],
  ['Tap', 'Pin a block'],
]

const KEYBOARD = [
  ['W A S D, arrows', 'Pan'],
  ['Q E', 'Rotate'],
  ['R F', 'Tilt'],
  ['+ −', 'Zoom'],
  ['Esc', 'Remove pin'],
]

function Controls({ title, controls }: { title: string; controls: string[][] }) {
  return (
    <>
      <h2>{title}</h2>
      <dl>
        {controls.map(([control, action]) => (
          <div key={control}>
            <dt>{control}</dt>
            <dd>{action}</dd>
          </div>
        ))}
      </dl>
    </>
  )
}

/** What the map's controls are, for whichever kind of pointer this device has. */
export function Help() {
  const touch = matchMedia('(pointer: coarse)').matches
  return (
    <div className="help">
      <Controls title={touch ? 'Touch' : 'Mouse'} controls={touch ? TOUCH : MOUSE} />
      <Controls title="Keyboard" controls={KEYBOARD} />
    </div>
  )
}
