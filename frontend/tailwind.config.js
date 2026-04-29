/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,jsx,ts,tsx}'],
  theme: {
    extend: {
      fontFamily: {
        display: ['Playfair Display', 'Georgia', 'serif'],
        body: ['DM Sans', 'system-ui', 'sans-serif'],
        arabic: ['Cairo', 'system-ui', 'sans-serif'],
      },
      colors: {
        coral: {
          50: '#fff5f5',
          100: '#ffe4e4',
          200: '#ffcaca',
          300: '#ffa0a0',
          400: '#ff6b6b',
          500: '#f94040',
          600: '#e51f1f',
          700: '#c11515',
        },
        blush: {
          50: '#fdf6f0',
          100: '#fae8d8',
          200: '#f5cfb0',
          300: '#edaf7d',
          400: '#e5854a',
          500: '#dc6728',
        },
        sage: {
          50: '#f4f7f4',
          100: '#e4ece4',
          200: '#c8dac8',
          300: '#9ebf9e',
          400: '#6d9e6d',
          500: '#4d7e4d',
          600: '#3a633a',
        },
        teal: {
          soft: '#e8f5f3',
          DEFAULT: '#4da6a0',
          deep: '#2d7a74',
        },
        warmgray: {
          50: '#fafaf8',
          100: '#f3f2ef',
          200: '#e8e6e1',
          300: '#d4d1c9',
          400: '#b5b1a7',
          500: '#8f8b80',
        },
      },
      animation: {
        'fade-up': 'fadeUp 0.5s ease forwards',
        'fade-in': 'fadeIn 0.3s ease forwards',
        'pulse-soft': 'pulseSoft 2s ease-in-out infinite',
        'shimmer': 'shimmer 1.5s ease-in-out infinite',
        'bounce-gentle': 'bounceGentle 1.4s ease-in-out infinite',
      },
      keyframes: {
        fadeUp: {
          from: { opacity: '0', transform: 'translateY(16px)' },
          to: { opacity: '1', transform: 'translateY(0)' },
        },
        fadeIn: {
          from: { opacity: '0' },
          to: { opacity: '1' },
        },
        pulseSoft: {
          '0%, 100%': { opacity: '1' },
          '50%': { opacity: '0.6' },
        },
        shimmer: {
          '0%': { backgroundPosition: '-200% 0' },
          '100%': { backgroundPosition: '200% 0' },
        },
        bounceGentle: {
          '0%, 80%, 100%': { transform: 'scale(0)', opacity: '0.4' },
          '40%': { transform: 'scale(1)', opacity: '1' },
        },
      },
    },
  },
  plugins: [],
}
