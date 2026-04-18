/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,jsx}'],
  darkMode: 'class',
  theme: {
    extend: {
      fontFamily: {
        sans: ['Inter', 'system-ui', '-apple-system', 'sans-serif'],
        mono: ['JetBrains Mono', 'Consolas', 'monospace'],
      },
      animation: {
        'pulse-heart': 'pulseHeart 1s ease-in-out infinite',
      },
      keyframes: {
        pulseHeart: {
          '0%, 100%': { transform: 'scale(1)', opacity: '0.8' },
          '15%': { transform: 'scale(1.3)', opacity: '1' },
          '30%': { transform: 'scale(1)', opacity: '0.8' },
        },
      },
    },
  },
  plugins: [],
};
