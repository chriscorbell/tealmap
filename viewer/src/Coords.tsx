import { Link } from 'lucide-react'
import type { ClipboardEvent, KeyboardEvent, RefObject } from 'react'
import { CopyButton } from './CopyButton'
import { parseCoords } from './route'

/** Where the camera is looking. x and z are also where you type to go somewhere. */
export function Coords({
  x,
  y,
  z,
  go,
  link,
}: {
  x: RefObject<HTMLInputElement | null>
  y: RefObject<HTMLElement | null>
  z: RefObject<HTMLInputElement | null>
  go: (to: { x?: number; z?: number }) => void
  link: () => string
}) {
  const key = (axis: 'x' | 'z') => (e: KeyboardEvent<HTMLInputElement>) => {
    // The map's own keys mustn't fire while typing.
    e.stopPropagation()
    const input = e.currentTarget
    if (e.key === 'Escape') {
      input.blur()
    } else if (e.key === 'Enter') {
      const pair = parseCoords(input.value)
      const n = Number(input.value)
      if (pair) go(pair)
      else if (input.value.trim() !== '' && Number.isFinite(n)) go({ [axis]: n })
      input.blur()
    }
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
        onKeyDown={key(axis)}
        onPaste={paste}
        onFocus={(e) => e.currentTarget.select()}
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
