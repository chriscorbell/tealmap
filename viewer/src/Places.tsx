import { useEffect, useMemo, useRef, useState } from 'react'

export interface Place {
  name: string
  x: number
  z: number
}

/** The map's named places, narrowed down by typing. */
export function Places({ places, open, go }: { places: Place[]; open: boolean; go: (place: Place) => void }) {
  const [query, setQuery] = useState('')
  const input = useRef<HTMLInputElement>(null)
  useEffect(() => {
    if (open) input.current?.focus()
  }, [open])
  const found = useMemo(() => {
    const q = query.trim().toLowerCase()
    return q ? places.filter((p) => p.name.toLowerCase().includes(q)) : places
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
          if (e.key === 'Enter' && found[0]) choose(found[0])
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
