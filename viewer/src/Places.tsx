import { useEffect, useMemo, useRef, useState } from 'react'

export interface Place {
  name: string
  x: number
  z: number
}

/** A name as far as searching goes: what it looks like on screen, in any case. */
const comparable = (name: string) => name.normalize('NFC').toLowerCase().replace(/\s+/g, ' ').trim()

/** The map's named places, narrowed down by typing. */
export function Places({ places, open, go }: { places: Place[]; open: boolean; go: (place: Place) => void }) {
  const [query, setQuery] = useState('')
  const input = useRef<HTMLInputElement>(null)
  useEffect(() => {
    // On a touch screen that would raise a keyboard over the list.
    if (open && !matchMedia('(pointer: coarse)').matches) input.current?.focus()
  }, [open])
  const found = useMemo(() => {
    const q = comparable(query)
    return q ? places.filter((p) => comparable(p.name).includes(q)) : places
  }, [places, query])
  const choose = (place: Place) => {
    setQuery('')
    go(place)
  }
  return (
    <div className="places">
      <input
        ref={input}
        type="search"
        placeholder="Find a place"
        aria-label="Find a place"
        autoComplete="off"
        spellCheck={false}
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        onKeyDown={(e) => {
          // Enter in the middle of composing text is part of the composing.
          if (e.key === 'Enter' && open && !e.repeat && !e.nativeEvent.isComposing && found[0]) choose(found[0])
        }}
      />
      {found.length > 0 ? (
        <ul>
          {found.map((p) => (
            <li key={p.name}>
              <button onClick={() => choose(p)}>{p.name}</button>
            </li>
          ))}
        </ul>
      ) : (
        <p>No place called that.</p>
      )}
    </div>
  )
}
