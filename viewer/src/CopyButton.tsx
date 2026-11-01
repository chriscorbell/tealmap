import { Check, X } from 'lucide-react'
import { type ReactNode, useEffect, useRef, useState } from 'react'
import { copyText } from './clipboard'

/** A button that copies text when pressed, and shows for a moment whether that worked. */
export function CopyButton({ text, label, children }: { text: () => string; label: string; children: ReactNode }) {
  const [result, setResult] = useState<'copied' | 'failed' | null>(null)
  const timer = useRef(0)
  useEffect(() => () => clearTimeout(timer.current), [])
  const copy = async () => {
    const ok = await copyText(text())
    setResult(ok ? 'copied' : 'failed')
    clearTimeout(timer.current)
    timer.current = window.setTimeout(() => setResult(null), 1500)
  }
  const shown = result === 'copied' ? 'Copied' : result === 'failed' ? "Couldn't copy" : label
  return (
    <button className={`icon ${result ?? ''}`} onClick={copy} aria-label={shown} title={shown}>
      {result === 'copied' ? <Check size={14} /> : result === 'failed' ? <X size={14} /> : children}
    </button>
  )
}
