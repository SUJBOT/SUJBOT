/**
 * WelcomeScreen Component - Displayed when starting a new conversation
 *
 * Features:
 * - Large SUJBOT branding with serif typography
 * - Suggested prompts users can click
 * - Gradient background effects
 * - Smooth fade-in animations
 */

import { FileText, Scale, Shield, FileCheck } from 'lucide-react';
import { useTranslation } from 'react-i18next';
import { cn } from '../../design-system/utils/cn';
import { SujbotLogo } from '../common/SujbotLogo';
import { GradientBackground } from '../common/GradientBackground';

interface WelcomeScreenProps {
  onPromptClick: (prompt: string) => void;
  children?: React.ReactNode;
}

// Prompt templates - titles and prompts use translation keys
const SUGGESTED_PROMPTS = [
  {
    icon: Scale,
    titleKey: 'welcome.regulatoryCompliance',
    promptKey: 'welcome.prompts.regulatoryCompliance',
  },
  {
    icon: Shield,
    titleKey: 'welcome.safetyAnalysis',
    promptKey: 'welcome.prompts.safetyAnalysis',
  },
  {
    icon: FileCheck,
    titleKey: 'welcome.documentComparison',
    promptKey: 'welcome.prompts.documentComparison',
  },
  {
    icon: FileText,
    titleKey: 'welcome.citationLookup',
    promptKey: 'welcome.prompts.citationLookup',
  },
];

export function WelcomeScreen({ onPromptClick, children }: WelcomeScreenProps) {
  const { t } = useTranslation();

  return (
    <div className={cn(
      'flex-1 flex flex-col items-center justify-center',
      'px-6 py-8 overflow-hidden'
    )}>
      <GradientBackground />

      {/* Main content */}
      <div
        className="max-w-4xl w-full flex flex-col items-center gap-8"
        style={{
          animation: 'fadeInUp 0.8s cubic-bezier(0.16, 1, 0.3, 1)',
        }}
      >
        {/* Branding */}
        <div className="text-center space-y-2">
          {/* Icon */}
          <div className="flex justify-center mb-1">
            <SujbotLogo size={64} className="opacity-90" />
          </div>

          <h1
            className={cn(
              'text-6xl font-light tracking-tight',
              'text-accent-950 dark:text-accent-50'
            )}
            style={{ fontFamily: 'var(--font-display)' }}
          >
            SUJBOT
          </h1>
        </div>

        {/* Search bar slot */}
        {children && (
          <div className="w-full max-w-3xl">
            {children}
          </div>
        )}

        {/* Suggested prompts */}
        <div className="w-full space-y-3">
          <p className={cn(
            'text-sm font-medium tracking-wide uppercase',
            'text-accent-500 dark:text-accent-500',
            'text-center'
          )}>
            {t('welcome.suggestedQuestions')}
          </p>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            {SUGGESTED_PROMPTS.map((item, index) => {
              const Icon = item.icon;
              return (
                <button
                  key={index}
                  onClick={() => onPromptClick(t(item.promptKey))}
                  className={cn(
                    'group relative',
                    'p-4 rounded-xl',
                    'border border-accent-200 dark:border-accent-800',
                    'bg-white/80 dark:bg-accent-900/50',
                    'backdrop-blur-sm',
                    'hover:border-accent-400 dark:hover:border-accent-600',
                    'hover:shadow-lg',
                    'transition-all duration-300',
                    'text-left'
                  )}
                  style={{
                    animation: `fadeInUp 0.8s cubic-bezier(0.16, 1, 0.3, 1) ${index * 0.1 + 0.2}s backwards`,
                  }}
                >
                  <div className="flex items-start gap-3">
                    <div className={cn(
                      'flex-shrink-0 w-10 h-10 rounded-lg',
                      'bg-accent-100 dark:bg-accent-800',
                      'flex items-center justify-center',
                      'group-hover:bg-accent-200 dark:group-hover:bg-accent-700',
                      'transition-colors duration-300'
                    )}>
                      <Icon size={20} className="text-accent-700 dark:text-accent-300" />
                    </div>
                    <div className="flex-1 min-w-0">
                      <div className={cn(
                        'text-sm font-medium mb-1',
                        'text-accent-900 dark:text-accent-100'
                      )}>
                        {t(item.titleKey)}
                      </div>
                      <div className={cn(
                        'text-xs line-clamp-2',
                        'text-accent-600 dark:text-accent-400'
                      )}>
                        {t(item.promptKey)}
                      </div>
                    </div>
                  </div>
                </button>
              );
            })}
          </div>
        </div>

      </div>
    </div>
  );
}
