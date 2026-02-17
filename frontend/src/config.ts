/**
 * Centralized application configuration.
 *
 * API_BASE_URL:
 * - Empty string = relative URLs (same-origin, through Nginx proxy) -- production
 * - 'http://localhost:8000' = direct backend for local development
 */
export const API_BASE_URL = import.meta.env.VITE_API_BASE_URL !== undefined
  ? import.meta.env.VITE_API_BASE_URL
  : 'http://localhost:8000';
