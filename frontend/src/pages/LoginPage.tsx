/**
 * LoginPage Component - Authentication gateway for SUJBOT
 *
 * Design: Sophisticated, centered layout matching the SUJBOT aesthetic
 * - Cormorant Garamond serif typography for branding
 * - Monochromatic palette with subtle gradients
 * - Smooth 750ms transitions matching theme system
 * - Ceremonial entrance feel for professional legal/technical system
 */

import React, { useState } from 'react';
import { Lock, Mail, AlertCircle } from 'lucide-react';
import { useTranslation } from 'react-i18next';
import { cn } from '../design-system/utils/cn';
import { useAuth } from '../contexts/AuthContext';
import { LanguageSwitcher } from '../components/header/LanguageSwitcher';
import { SujbotLogo } from '../components/common/SujbotLogo';
import { GradientBackground } from '../components/common/GradientBackground';
import './LoginPage.css';

// Pre-computed particle positions to avoid Math.random() during render
const PARTICLES = [
  { width: 120, height: 180, left: 15, top: 25, duration: 28, delay: 1.2 },
  { width: 80, height: 95, left: 72, top: 60, duration: 22, delay: 3.5 },
  { width: 150, height: 130, left: 40, top: 10, duration: 32, delay: 0.8 },
  { width: 65, height: 110, left: 85, top: 75, duration: 18, delay: 4.2 },
  { width: 140, height: 70, left: 5, top: 85, duration: 25, delay: 2.0 },
  { width: 100, height: 160, left: 55, top: 45, duration: 30, delay: 1.5 },
];

export function LoginPage() {
  const { t } = useTranslation();
  const { login } = useAuth();
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [isLoading, setIsLoading] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');
    setIsLoading(true);

    try {
      const result = await login(email, password);
      if (!result.success) {
        // Use specific error message from AuthContext (e.g., "Invalid credentials", "Failed to connect to backend")
        setError(result.error || 'Login failed. Please try again.');
      }
    } catch (err) {
      // Fallback for unexpected errors (shouldn't happen with new error handling)
      setError('An unexpected error occurred. Please try again.');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className={cn(
      'min-h-screen flex items-center justify-center',
      'bg-white dark:bg-accent-950',
      'px-6 py-12',
      'relative overflow-hidden'
    )}>
      <GradientBackground />

      {/* Language switcher - top right */}
      <div className="absolute top-6 right-6 z-10">
        <LanguageSwitcher />
      </div>

      {/* Floating particles effect (subtle decoration) */}
      <div className="absolute inset-0 -z-10 overflow-hidden pointer-events-none">
        {PARTICLES.map((particle, i) => (
          <div
            key={i}
            className={cn(
              'absolute rounded-full',
              'bg-accent-200/20 dark:bg-accent-700/10'
            )}
            style={{
              width: `${particle.width}px`,
              height: `${particle.height}px`,
              left: `${particle.left}%`,
              top: `${particle.top}%`,
              animation: `float ${particle.duration}s ease-in-out infinite`,
              animationDelay: `${particle.delay}s`,
            }}
          />
        ))}
      </div>

      {/* Login card */}
      <div
        className="w-full max-w-md"
        style={{
          animation: 'fadeInUp 0.8s cubic-bezier(0.16, 1, 0.3, 1)',
        }}
      >
        {/* Logo and branding */}
        <div className="text-center mb-8">
          {/* SUJBOT Logo */}
          <div className="flex justify-center mb-4">
            <SujbotLogo
              size={80}
              className="opacity-90"
              style={{ filter: 'drop-shadow(0 2px 8px rgba(0, 0, 0, 0.1))' }}
            />
          </div>

          {/* App name */}
          <h1
            className={cn(
              'text-5xl font-light tracking-tight mb-2',
              'text-accent-950 dark:text-accent-50'
            )}
            style={{ fontFamily: 'var(--font-display)' }}
          >
            SUJBOT
          </h1>

          {/* Tagline */}
          <p className={cn(
            'text-sm font-light',
            'text-accent-500 dark:text-accent-400'
          )}>
            {t('header.tagline')}
          </p>
        </div>

        {/* Login form */}
        <div className={cn(
          'p-8 rounded-2xl',
          'bg-white/80 dark:bg-accent-900/50',
          'backdrop-blur-sm',
          'border border-accent-200 dark:border-accent-800',
          'shadow-xl'
        )}>
          <form onSubmit={handleSubmit} className="space-y-6">
            {/* Email field */}
            <div>
              <label
                htmlFor="email"
                className={cn(
                  'block text-sm font-medium mb-2',
                  'text-accent-700 dark:text-accent-300'
                )}
              >
                {t('login.email')}
              </label>
              <div className="relative">
                <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                  <Mail size={18} className="text-accent-400 dark:text-accent-500" />
                </div>
                <input
                  id="email"
                  type="email"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  className={cn(
                    'w-full pl-10 pr-4 py-3 rounded-lg',
                    'bg-white dark:bg-accent-800',
                    'border border-accent-300 dark:border-accent-700',
                    'text-accent-900 dark:text-accent-100',
                    'placeholder-accent-400 dark:placeholder-accent-500',
                    'focus:outline-none focus:ring-2 focus:ring-accent-500 dark:focus:ring-accent-400',
                    'focus:border-transparent',
                    'transition-all duration-200'
                  )}
                  placeholder="admin@sujbot.local"
                  required
                  autoComplete="email"
                />
              </div>
            </div>

            {/* Password field */}
            <div>
              <label
                htmlFor="password"
                className={cn(
                  'block text-sm font-medium mb-2',
                  'text-accent-700 dark:text-accent-300'
                )}
              >
                {t('login.password')}
              </label>
              <div className="relative">
                <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                  <Lock size={18} className="text-accent-400 dark:text-accent-500" />
                </div>
                <input
                  id="password"
                  type="password"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  className={cn(
                    'w-full pl-10 pr-4 py-3 rounded-lg',
                    'bg-white dark:bg-accent-800',
                    'border border-accent-300 dark:border-accent-700',
                    'text-accent-900 dark:text-accent-100',
                    'placeholder-accent-400 dark:placeholder-accent-500',
                    'focus:outline-none focus:ring-2 focus:ring-accent-500 dark:focus:ring-accent-400',
                    'focus:border-transparent',
                    'transition-all duration-200'
                  )}
                  placeholder={t('login.enterPassword')}
                  required
                  autoComplete="current-password"
                />
              </div>
            </div>

            {/* Error message */}
            {error && (
              <div
                className={cn(
                  'p-3 rounded-lg flex items-start gap-2',
                  'bg-red-50 dark:bg-red-900/20',
                  'border border-red-200 dark:border-red-800',
                  'text-red-800 dark:text-red-200',
                  'text-sm'
                )}
                style={{
                  animation: 'fadeIn 0.3s ease-in',
                }}
              >
                <AlertCircle size={18} className="flex-shrink-0 mt-0.5" />
                <span>{error}</span>
              </div>
            )}

            {/* Submit button */}
            <button
              type="submit"
              disabled={isLoading}
              className={cn(
                'w-full py-3 px-4 rounded-lg',
                'bg-accent-900 dark:bg-accent-100',
                'text-white dark:text-accent-900',
                'font-medium',
                'hover:bg-accent-800 dark:hover:bg-accent-200',
                'focus:outline-none focus:ring-2 focus:ring-accent-500 dark:focus:ring-accent-400',
                'focus:ring-offset-2',
                'disabled:opacity-50 disabled:cursor-not-allowed',
                'transition-all duration-200',
                'shadow-lg hover:shadow-xl'
              )}
            >
              {isLoading ? (
                <span className="flex items-center justify-center gap-2">
                  <svg
                    className="animate-spin h-5 w-5"
                    xmlns="http://www.w3.org/2000/svg"
                    fill="none"
                    viewBox="0 0 24 24"
                  >
                    <circle
                      className="opacity-25"
                      cx="12"
                      cy="12"
                      r="10"
                      stroke="currentColor"
                      strokeWidth="4"
                    />
                    <path
                      className="opacity-75"
                      fill="currentColor"
                      d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                    />
                  </svg>
                  {t('login.authenticating')}
                </span>
              ) : (
                t('login.signIn')
              )}
            </button>
          </form>

          {/* Footer hint */}
          <div className="mt-6 text-center">
            <p className={cn(
              'text-xs',
              'text-accent-400 dark:text-accent-600'
            )}>
              {t('login.authorizedAccess')}
            </p>
          </div>
        </div>

      </div>
    </div>
  );
}
