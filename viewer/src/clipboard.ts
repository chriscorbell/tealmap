/** Copies text. The clipboard API only exists on HTTPS, and maps often run on plain HTTP. */
export async function copyText(text: string): Promise<boolean> {
  try {
    if (navigator.clipboard) {
      await navigator.clipboard.writeText(text)
      return true
    }
  } catch {
    // Denied or unavailable: the older way may still work.
  }
  const focused = document.activeElement as HTMLElement | null
  const area = document.createElement('textarea')
  area.value = text
  area.style.position = 'fixed'
  area.style.opacity = '0'
  document.body.append(area)
  area.select()
  let copied = false
  try {
    copied = document.execCommand('copy')
  } catch {
    copied = false
  }
  area.remove()
  focused?.focus()
  return copied
}
