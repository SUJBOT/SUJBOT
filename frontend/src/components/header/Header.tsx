/**
 * Header Component - Top navigation with sidebar control
 */

import { useState } from 'react';
import { Menu, LogOut, FolderOpen } from 'lucide-react';
import { useTranslation } from 'react-i18next';
import { cn } from '../../design-system/utils/cn';
import { useHover } from '../../design-system/animations/hooks/useHover';
import { useAuth } from '../../contexts/AuthContext';
import { AgentVariantSelector } from './AgentVariantSelector';
import { LanguageSwitcher } from './LanguageSwitcher';
import { DocumentBrowser } from './DocumentBrowser';
import { SujbotLogo } from '../common/SujbotLogo';

interface HeaderProps {
  onToggleSidebar: () => void;
  sidebarOpen: boolean;
}

export function Header({
  onToggleSidebar,
  sidebarOpen,
}: HeaderProps) {
  // Translations
  const { t } = useTranslation();

  // Authentication
  const { logout } = useAuth();

  // Document browser state
  const [documentBrowserOpen, setDocumentBrowserOpen] = useState(false);

  // Animation hooks
  const hamburgerHover = useHover({ scale: true });
  const logoutHover = useHover({ scale: true });
  const documentHover = useHover({ scale: true });

  return (
    <header className={cn(
      'bg-white dark:bg-accent-900',
      'border-b border-accent-200 dark:border-accent-800',
      'px-6 py-4',
      'transition-colors duration-200'
    )}>
      <div className="flex items-center justify-between">
        {/* Left side: Hamburger + Logo */}
        <div className="flex items-center gap-3">
          {/* Hamburger button */}
          <button
            onClick={onToggleSidebar}
            {...hamburgerHover.hoverProps}
            style={hamburgerHover.style}
            className={cn(
              'p-2 rounded-lg',
              'text-accent-700 dark:text-accent-300',
              'hover:bg-accent-100 dark:hover:bg-accent-800',
              'transition-colors duration-200'
            )}
            aria-label={sidebarOpen ? 'Close sidebar' : 'Open sidebar'}
          >
            <Menu size={20} />
          </button>

          {/* Logo and title */}
          <div className="flex items-center gap-3">
            <SujbotLogo size={40} className="flex-shrink-0" />
            <div>
              <h1
                className={cn(
                  'text-xl font-light tracking-tight',
                  'text-accent-900 dark:text-accent-100'
                )}
                style={{ fontFamily: 'var(--font-display)' }}
              >
                SUJBOT
              </h1>
              <p className={cn(
                'text-xs font-light',
                'text-accent-500 dark:text-accent-400'
              )}>
                {t('header.tagline')}
              </p>
            </div>
          </div>
        </div>

        {/* Controls */}
        <div className="flex items-center gap-3">
          {/* Language switcher */}
          <LanguageSwitcher />

          {/* Document browser */}
          <div className="relative">
            <button
              onClick={() => setDocumentBrowserOpen(!documentBrowserOpen)}
              {...documentHover.hoverProps}
              style={documentHover.style}
              className={cn(
                'p-2 rounded-lg',
                'text-accent-700 dark:text-accent-300',
                'hover:bg-accent-100 dark:hover:bg-accent-800',
                'transition-colors duration-200',
                documentBrowserOpen && 'bg-accent-100 dark:bg-accent-800'
              )}
              title={t('header.browseDocuments')}
            >
              <FolderOpen size={20} />
            </button>
            <DocumentBrowser
              isOpen={documentBrowserOpen}
              onClose={() => setDocumentBrowserOpen(false)}
            />
          </div>

          {/* Agent variant selector */}
          <AgentVariantSelector />

          {/* Logout button */}
          <button
            onClick={logout}
            {...logoutHover.hoverProps}
            style={logoutHover.style}
            className={cn(
              'p-2 rounded-lg',
              'text-accent-700 dark:text-accent-300',
              'hover:bg-accent-100 dark:hover:bg-accent-800',
              'transition-colors duration-200'
            )}
            title={t('header.signOut')}
          >
            <LogOut size={20} />
          </button>
        </div>
      </div>
    </header>
  );
}
