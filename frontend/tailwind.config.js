/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      // Grayscale color palette (minimal accent design)
      colors: {
        accent: {
          50: '#FAFAFA',
          100: '#F5F5F5',
          200: '#E5E5E5',
          300: '#D4D4D4',
          400: '#A3A3A3',
          500: '#737373',
          600: '#525252',
          700: '#404040',
          800: '#262626',
          900: '#171717',
          950: '#0A0A0A',
        },
      },

      // Custom animations for smooth interactions
      animation: {
        'slide-in-right': 'slideInRight 0.3s ease-out',
        'slide-in-left': 'slideInLeft 0.3s ease-out',
        'slide-in-up': 'slideInUp 0.3s ease-out',
        'fade-in': 'fadeIn 0.2s ease-in',
        'scale-in': 'scaleIn 0.2s ease-out',
        'lift-up': 'liftUp 0.15s ease-out',
      },

      // Animation keyframes
      keyframes: {
        slideInRight: {
          '0%': { transform: 'translateX(20px)', opacity: '0' },
          '100%': { transform: 'translateX(0)', opacity: '1' },
        },
        slideInLeft: {
          '0%': { transform: 'translateX(-20px)', opacity: '0' },
          '100%': { transform: 'translateX(0)', opacity: '1' },
        },
        slideInUp: {
          '0%': { transform: 'translateY(20px)', opacity: '0' },
          '100%': { transform: 'translateY(0)', opacity: '1' },
        },
        fadeIn: {
          '0%': { opacity: '0' },
          '100%': { opacity: '1' },
        },
        scaleIn: {
          '0%': { transform: 'scale(0.95)', opacity: '0' },
          '100%': { transform: 'scale(1)', opacity: '1' },
        },
        liftUp: {
          '0%': { transform: 'translateY(0)' },
          '100%': { transform: 'translateY(-2px)' },
        },
      },

      // Transition durations
      transitionDuration: {
        '0': '0ms',
        '150': '150ms',
        '250': '250ms',
        '400': '400ms',
        '600': '600ms',
      },

      // Typography plugin customization (grayscale palette)
      typography: (theme) => ({
        DEFAULT: {
          css: {
            '--tw-prose-body': theme('colors.accent.800'),
            '--tw-prose-headings': theme('colors.accent.900'),
            '--tw-prose-links': theme('colors.accent.900'),
            '--tw-prose-bold': theme('colors.accent.900'),
            '--tw-prose-counters': theme('colors.accent.500'),
            '--tw-prose-bullets': theme('colors.accent.400'),
            '--tw-prose-hr': theme('colors.accent.200'),
            '--tw-prose-quotes': theme('colors.accent.700'),
            '--tw-prose-quote-borders': theme('colors.accent.300'),
            '--tw-prose-code': theme('colors.accent.800'),
            '--tw-prose-th-borders': theme('colors.accent.300'),
            '--tw-prose-td-borders': theme('colors.accent.200'),
            // Invert (text on dark backgrounds: user bubbles in light mode, assistant bubbles in dark mode)
            '--tw-prose-invert-body': theme('colors.accent.200'),
            '--tw-prose-invert-headings': theme('colors.accent.50'),
            '--tw-prose-invert-links': theme('colors.accent.50'),
            '--tw-prose-invert-bold': theme('colors.accent.50'),
            '--tw-prose-invert-counters': theme('colors.accent.400'),
            '--tw-prose-invert-bullets': theme('colors.accent.500'),
            '--tw-prose-invert-hr': theme('colors.accent.700'),
            '--tw-prose-invert-quotes': theme('colors.accent.300'),
            '--tw-prose-invert-quote-borders': theme('colors.accent.600'),
            '--tw-prose-invert-code': theme('colors.accent.200'),
            '--tw-prose-invert-th-borders': theme('colors.accent.600'),
            '--tw-prose-invert-td-borders': theme('colors.accent.700'),
            // Remove link underline, add on hover
            a: {
              textDecoration: 'none',
              '&:hover': {
                textDecoration: 'underline',
              },
            },
            // Tighter heading spacing for chat context
            h2: {
              marginTop: '1em',
              marginBottom: '0.5em',
            },
            h3: {
              marginTop: '0.75em',
              marginBottom: '0.4em',
            },
          },
        },
      }),
    },
  },
  plugins: [
    require('@tailwindcss/typography'),
  ],
  darkMode: 'class',
}
