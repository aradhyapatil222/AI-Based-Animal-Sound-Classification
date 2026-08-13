/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        obsidian: {
          950: '#060709',
          900: '#090A0F',
          850: '#0D0E12',
          800: '#12141C',
          750: '#171A24',
          700: '#1E2230',
        },
        electric: {
          cyan: '#06b6d4',
          'cyan-glow': 'rgba(6, 182, 212, 0.18)',
          indigo: '#6366f1',
          'indigo-glow': 'rgba(99, 102, 241, 0.18)',
          emerald: '#10b981',
          violet: '#8b5cf6',
        },
      },
      fontFamily: {
        sans: ['Geist', 'Inter', '-apple-system', 'BlinkMacSystemFont', 'sans-serif'],
        mono: ['Geist Mono', 'SFMono-Regular', 'Menlo', 'Monaco', 'monospace'],
      },
      boxShadow: {
        'glass-glow': '0 0 32px -8px rgba(99, 102, 241, 0.12)',
        'cyan-glow': '0 0 24px -4px rgba(6, 182, 212, 0.25)',
        'card-hover': '0 20px 40px -15px rgba(0, 0, 0, 0.62)',
      },
      animation: {
        'pulse-subtle': 'pulseSubtle 3s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'radar-ping': 'radarPing 2.5s cubic-bezier(0, 0, 0.2, 1) infinite',
        'fade-in': 'fadeIn 0.2s cubic-bezier(0, 0, 0.2, 1) forwards',
      },
      keyframes: {
        pulseSubtle: {
          '0%, 100%': { opacity: 1 },
          '50%': { opacity: 0.65 },
        },
        radarPing: {
          '0%': { transform: 'scale(0.95)', opacity: 0.8 },
          '50%': { transform: 'scale(1.25)', opacity: 0 },
          '100%': { transform: 'scale(0.95)', opacity: 0 },
        },
        fadeIn: {
          '0%': { opacity: 0, transform: 'translateY(4px)' },
          '100%': { opacity: 1, transform: 'translateY(0)' },
        }
      }
    },
  },
  plugins: [],
}
