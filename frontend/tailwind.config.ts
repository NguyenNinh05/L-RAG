/** @type {import('tailwindcss').Config} */
export default {
  darkMode: 'class',
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        border: 'hsl(var(--border) / <alpha-value>)',
        background: 'hsl(var(--background) / <alpha-value>)',
        foreground: 'hsl(var(--foreground) / <alpha-value>)',
        primary: {
          DEFAULT: 'hsl(var(--primary) / <alpha-value>)',
          foreground: 'hsl(var(--primary-foreground) / <alpha-value>)',
        },
        secondary: {
          DEFAULT: 'hsl(var(--secondary) / <alpha-value>)',
          foreground: 'hsl(var(--secondary-foreground) / <alpha-value>)',
        },
        accent: {
          DEFAULT: 'hsl(var(--accent) / <alpha-value>)',
          foreground: 'hsl(var(--accent-foreground) / <alpha-value>)',
        },
        muted: {
          DEFAULT: 'hsl(var(--muted) / <alpha-value>)',
          foreground: 'hsl(var(--muted-foreground) / <alpha-value>)',
        },
        destructive: {
          DEFAULT: 'hsl(var(--destructive) / <alpha-value>)',
          foreground: 'hsl(var(--destructive-foreground) / <alpha-value>)',
        },
        ring: 'hsl(var(--ring) / <alpha-value>)',
        // Semantic change-type colors (diff vocabulary)
        'diff-addition': 'hsl(var(--diff-addition) / <alpha-value>)',
        'diff-deletion': 'hsl(var(--diff-deletion) / <alpha-value>)',
        'diff-numerical': 'hsl(var(--diff-numerical) / <alpha-value>)',
        'diff-terminology': 'hsl(var(--diff-terminology) / <alpha-value>)',
        'diff-structural': 'hsl(var(--diff-structural) / <alpha-value>)',
        'diff-reorder': 'hsl(var(--diff-reorder) / <alpha-value>)',
      },
      fontFamily: {
        sans: ['Lato', 'system-ui', 'sans-serif'],
        serif: ['"EB Garamond"', 'Georgia', 'serif'],
        mono: ['"JetBrains Mono"', 'ui-monospace', 'monospace'],
      },
      borderRadius: {
        sm: '6px',
        md: '8px',
        lg: '12px',
        xl: '16px',
      },
      fontSize: {
        '2xs': ['12px', { lineHeight: '1.25' }],
        xs: ['13px', { lineHeight: '1.25' }],
        sm: ['14px', { lineHeight: '1.43' }],
        base: ['16px', { lineHeight: '1.5' }],
        lg: ['18px', { lineHeight: '1.5' }],
        xl: ['20px', { lineHeight: '1.4' }],
        '2xl': ['24px', { lineHeight: '1.33' }],
        '3xl': ['32px', { lineHeight: '1.25' }],
      },
      spacing: {
        18: '4.5rem',
        88: '22rem',
      },
      zIndex: {
        dropdown: '10',
        sticky: '20',
        drawer: '40',
        modal: '50',
        toast: '60',
        overlay: '100',
      },
      boxShadow: {
        card: '0 1px 3px 0 rgb(0 0 0 / 0.06)',
        'card-hover': '0 4px 12px 0 rgb(0 0 0 / 0.08)',
      },
    },
  },
  plugins: [],
}
