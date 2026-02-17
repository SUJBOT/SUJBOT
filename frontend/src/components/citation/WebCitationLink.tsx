/**
 * WebCitationLink Component
 *
 * Renders a clickable external link badge for web search citations.
 * Matches the blue badge styling of CitationLink but opens URLs in a new tab.
 *
 * Display format: [title] with ExternalLink icon
 */

import { ExternalLink } from 'lucide-react';
import { cn } from '../../design-system/utils/cn';

interface WebCitationLinkProps {
  url: string;
  title: string;
}

export function WebCitationLink({ url, title }: WebCitationLinkProps) {
  // Truncate long titles for display
  const displayTitle = title.length > 30 ? title.slice(0, 27) + '...' : title;

  return (
    <a
      href={url}
      target="_blank"
      rel="noopener noreferrer"
      title={url}
      className={cn(
        'inline-flex items-center gap-1 px-1.5 py-0.5 rounded',
        'bg-blue-100 dark:bg-blue-900/40',
        'text-blue-700 dark:text-blue-300',
        'hover:bg-blue-200 dark:hover:bg-blue-800/60',
        'hover:text-blue-800 dark:hover:text-blue-200',
        'text-xs font-mono',
        'transition-colors duration-150',
        'cursor-pointer',
        'border border-blue-200 dark:border-blue-700/50',
        'focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-1',
        'no-underline'
      )}
    >
      <ExternalLink size={10} />
      {displayTitle}
    </a>
  );
}
