import { createContext, useContext, useLayoutEffect, useMemo, useState, type Dispatch, type ReactNode, type SetStateAction } from 'react';

// All site colors live here. CSS and SVGs consume these keys as --custom-properties.
const graphiteColors = {
  bg: 'oklch(14.1% 0.005 285.823)',
  surface: 'oklch(21% 0.006 285.885)',
  surface2: 'oklch(27.4% 0.006 286.033)',
  'surface-hover': 'oklch(37% 0.013 285.805)',
  line: 'oklch(27.4% 0.006 286.033)',
  'line-strong': 'oklch(44.2% 0.017 285.786)',
  dim: 'oklch(55.2% 0.016 285.938)',
  muted: 'oklch(70.5% 0.015 286.067)',
  'text-soft': 'oklch(87.1% 0.006 286.286)',
  text: 'oklch(92% 0.004 286.32)',
  'text-strong': 'oklch(96.7% 0.001 286.375)',
  accent: 'var(--text)',
  'accent-hover': 'var(--text-strong)',
  'on-accent': 'var(--surface)',
  'brand-accent': '#c5ef82',
  'brand-ink': '#172016',
  'status-dot': '#7e9c69',
  'bullet-marker': '#a7c384',
  'diagram-dot': '#90ad70',
  'interview-bg': '#292719',
  'interview-border': '#665a38',
  'interview-divider': '#514b2e',
  'interview-label': '#decd93',
  'interview-heading': '#eee7c8',
  'interview-text': '#d6d4ad',
  'grade-retry-bg': '#30271c',
  'grade-retry-border': '#6b513c',
  'grade-retry-text': '#e3c4a0',
  'review-retry-bg': '#513b29',
  'review-retry-text': '#e6c397',
  'error-bg': '#35291e',
  'error-border': '#896146',
  'error-text': '#f3d4ad',
  'error-action': '#f1d6a6',
  'visual-shadow': '#00000012',
  'motion-note': '#d4c5e7',
  'chart-primary': 'var(--accent)',
  'chart-blue': '#94c9ea',
  'chart-purple': '#d0aceb',
  'chart-amber': '#efbd84',
  'chart-negative': '#f29f97',
  'category-foundations': '#b7d9a0',
  'category-alignment': '#ddb58c',
  'category-architecture': '#b6b3ee',
  'category-inference': '#8acbc6',
  'category-applications': '#a4c6e9',
  'category-evaluation': '#e1a6ae',
  'category-ai-farm': 'var(--brand-accent)',
  'category-system-design': '#e5cc87',
};

export type ThemeColors = { [Key in keyof typeof graphiteColors]: string };
export type Theme = { name: string; colorScheme: 'dark' | 'light'; colors: ThemeColors };

export const themes = {
  graphite: { name: 'Graphite', colorScheme: 'dark', colors: graphiteColors },
  midnight: {
    name: 'Midnight',
    colorScheme: 'dark',
    colors: {
      ...graphiteColors,
      bg: '#0b1220',
      surface: '#111c30',
      surface2: '#1b2b43',
      'surface-hover': '#2c405d',
      line: '#1b2b43',
      'line-strong': '#4c6585',
      dim: '#7286a3',
      muted: '#a2b2c9',
      'text-soft': '#c8d5e8',
      text: '#e2eaf6',
      'text-strong': '#f4f8ff',
      accent: '#9acbff',
      'accent-hover': '#c2dfff',
      'on-accent': '#10213a',
      'brand-accent': '#9acbff',
      'brand-ink': '#10213a',
      'status-dot': '#83afd9',
      'bullet-marker': '#a8c9ed',
      'diagram-dot': '#83afd9',
    },
  },
} satisfies Record<string, Theme>;

// Change this one line to swap the site's starting palette.
export const defaultTheme: Theme = themes.graphite;

type ThemeContextValue = { theme: Theme; setTheme: Dispatch<SetStateAction<Theme>> };
const ThemeContext = createContext<ThemeContextValue | null>(null);

export function ThemeProvider({ children, initialTheme = defaultTheme }: { children: ReactNode; initialTheme?: Theme }) {
  const [theme, setTheme] = useState<Theme>(initialTheme);

  // Apply before paint, without adding a wrapper that could affect page layout.
  useLayoutEffect(() => {
    const root = document.documentElement;
    const wasDark = root.classList.contains('dark');
    root.classList.toggle('dark', theme.colorScheme === 'dark');
    const properties = { ...Object.fromEntries(Object.entries(theme.colors).map(([key, value]) => [`--${key}`, value])), 'color-scheme': theme.colorScheme };
    const previous = Object.keys(properties).map(key => [key, root.style.getPropertyValue(key), root.style.getPropertyPriority(key)]);
    for (const [key, value] of Object.entries(properties)) root.style.setProperty(key, value);

    const computed = getComputedStyle(root);
    const meta = document.querySelector<HTMLMetaElement>('meta[name="theme-color"]');
    const previousMeta = meta?.getAttribute('content');
    meta?.setAttribute('content', computed.backgroundColor);
    const icon = document.querySelector<HTMLLinkElement>('link[rel="icon"]');
    const previousIcon = icon?.getAttribute('href');
    const svg = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64"><rect width="64" height="64" rx="16" fill="${computed.getPropertyValue('--brand-accent').trim()}"/><path d="M18 14v36h30v-8H27V14z" fill="${computed.getPropertyValue('--brand-ink').trim()}"/><circle cx="44" cy="21" r="7" fill="${computed.getPropertyValue('--brand-ink').trim()}"/></svg>`;
    icon?.setAttribute('href', `data:image/svg+xml,${encodeURIComponent(svg)}`);

    return () => {
      root.classList.toggle('dark', wasDark);
      for (const [key, value, priority] of previous) {
        if (value) root.style.setProperty(key, value, priority);
        else root.style.removeProperty(key);
      }
      if (meta) {
        if (previousMeta == null) meta.removeAttribute('content');
        else meta.setAttribute('content', previousMeta);
      }
      if (icon) {
        if (previousIcon == null) icon.removeAttribute('href');
        else icon.setAttribute('href', previousIcon);
      }
    };
  }, [theme]);

  const value = useMemo(() => ({ theme, setTheme }), [theme]);
  return <ThemeContext.Provider value={value}>{children}</ThemeContext.Provider>;
}

export function useTheme() {
  const context = useContext(ThemeContext);
  if (!context) throw new Error('useTheme must be used within a ThemeProvider');
  return context;
}
