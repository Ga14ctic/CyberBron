/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        cyber: {
          primary: '#00ff88',
          secondary: '#00d4ff',
          dark: '#0a0e1a',
          darker: '#050810',
          gray: '#1a1f2e',
          lightgray: '#2d3548',
        },
      },
      fontFamily: {
        mono: ['Fira Code', 'monospace'],
      },
      animation: {
        'glow': 'glow 2s ease-in-out infinite alternate',
        'pulse-slow': 'pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite',
      },
      keyframes: {
        glow: {
          '0%': { 
            textShadow: '0 0 10px #00ff88, 0 0 20px #00ff88, 0 0 30px #00ff88',
          },
          '100%': { 
            textShadow: '0 0 20px #00ff88, 0 0 30px #00ff88, 0 0 40px #00ff88',
          },
        },
      },
    },
  },
  plugins: [],
}
