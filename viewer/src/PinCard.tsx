import { Copy, Link, MapPin, X } from 'lucide-react'
import { type Ref, useLayoutEffect, useState } from 'react'
import { CopyButton } from './CopyButton'
import type { Pin } from './route'

/** A marker on a block, with its coordinates. The engine's camera updates move it, not React. */
export function PinCard({
  pin,
  marker,
  link,
  close,
  place,
}: {
  pin: Pin
  marker: Ref<HTMLDivElement>
  link: () => string
  close: () => void
  /** Positions the marker and fits the card on screen. */
  place: () => void
}) {
  const [open, setOpen] = useState(true)
  // The card's size decides how it fits, and a closed card has none.
  useLayoutEffect(place, [open, place])
  return (
    <div ref={marker} className="pin">
      <button className="pin-mark" onClick={() => setOpen((v) => !v)} aria-label="Pinned block" aria-expanded={open}>
        <MapPin size={22} strokeWidth={2.2} />
      </button>
      {open && (
        <div className="pin-card">
          <span className="pin-coords">
            x <b>{pin.x}</b> y <b>{pin.y}</b> z <b>{pin.z}</b>
          </span>
          <CopyButton text={() => `${pin.x} ${pin.y} ${pin.z}`} label="Copy coordinates">
            <Copy size={14} />
          </CopyButton>
          <CopyButton text={link} label="Copy link to this pin">
            <Link size={14} />
          </CopyButton>
          <button className="icon" onClick={close} aria-label="Remove pin" title="Remove pin">
            <X size={14} />
          </button>
        </div>
      )}
    </div>
  )
}
