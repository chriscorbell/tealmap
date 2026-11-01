import { Link } from 'lucide-react'
import type { ClipboardEvent, FocusEvent, KeyboardEvent, RefObject } from 'react'
import { CopyButton } from './CopyButton'
import { parseCoord, parseCoords } from './route'

/** Where the camera is looking. x and z are also where you type to go somewhere. */
export function Coords({
  x,
  y,
  z,
  go,
  restore,
  link,
}: {
  x: RefObject<HTMLInputElement | null>
  y: RefObject<HTMLElement | null>
  z: RefObject<HTMLInputElement | null>
  go: (to: { x?: number; z?: number }) => void
  /** Puts the camera's position back in the fields. */
  restore: () => void
  link: () => string
}) {
  const key = (e: KeyboardEvent<HTMLInputElement>) => {
    // The map's own keys mustn't fire while typing.
    e.stopPropagation()
    const input = e.currentTarget
    if (e.key === 'Escape') {
      input.blur()
    } else if (e.key === 'Enter') {
      // Both fields count: typing x, then Tab, then z is one move. One field can
      // hold a whole position too. A field that isn't a number leaves that axis alone.
      const number = parseCoord(input.value)
      const pair = number === null ? parseCoords(input.value) : null
      go(
        pair ?? {
          x: parseCoord(x.current?.value ?? '') ?? undefined,
          z: parseCoord(z.current?.value ?? '') ?? undefined,
        },
      )
      input.blur()
    }
  }

  // Leaving the fields gives up on what was typed. Moving between them doesn't.
  const leave = (e: FocusEvent<HTMLInputElement>) => {
    if (e.relatedTarget !== x.current && e.relatedTarget !== z.current) restore()
  }

  // A pasted position usually has both axes, so it goes straight there.
  const paste = (e: ClipboardEvent<HTMLInputElement>) => {
    const pair = parseCoords(e.clipboardData.getData('text'))
    if (!pair) return
    e.preventDefault()
    go(pair)
    e.currentTarget.blur()
  }

  const field = (axis: 'x' | 'z') => (
    <label>
      {axis}
      <input
        ref={axis === 'x' ? x : z}
        type="text"
        autoComplete="off"
        spellCheck={false}
        aria-label={`Go to ${axis}`}
        defaultValue="0"
        onKeyDown={key}
        onPaste={paste}
        onFocus={(e) => e.currentTarget.select()}
        onBlur={leave}
      />
    </label>
  )

  return (
    <footer className="bar bottom-left coords">
      {field('x')}
      <span>
        y <b ref={y}>0</b>
      </span>
      {field('z')}
      <CopyButton text={link} label="Copy link to this view">
        <Link size={14} />
      </CopyButton>
    </footer>
  )
}
