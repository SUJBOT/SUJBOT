/**
 * AttachmentChip - Shared attachment chip display for chat input and messages
 */

import { Image, File, FileText, X } from 'lucide-react';
import { cn } from '../../design-system/utils/cn';

interface AttachmentChipProps {
  filename: string;
  mimeType: string;
  sizeBytes: number;
  /** Called when chip body is clicked (e.g. preview) */
  onClick?: () => void;
  /** Called when remove button is clicked; hides remove button if omitted */
  onRemove?: () => void;
  removeTitle?: string;
}

export function AttachmentChip({ filename, mimeType, sizeBytes, onClick, onRemove, removeTitle }: AttachmentChipProps) {
  const icon = mimeType.startsWith('image/')
    ? <Image size={12} />
    : mimeType === 'application/pdf'
      ? <FileText size={12} />
      : <File size={12} />;

  const content = (
    <>
      {icon}
      <span className="truncate max-w-[150px]" title={filename}>
        {filename}
      </span>
      <span className="text-blue-400 dark:text-blue-500">
        ({(sizeBytes / 1024).toFixed(0)} KB)
      </span>
    </>
  );

  return (
    <div
      className={cn(
        'inline-flex items-center gap-1.5 px-2.5 py-1',
        'bg-blue-50 dark:bg-blue-900/30',
        'text-blue-700 dark:text-blue-300',
        'rounded-lg',
        'border border-blue-200 dark:border-blue-800',
        'text-xs'
      )}
    >
      {onClick ? (
        <button
          type="button"
          onClick={onClick}
          className={cn(
            'inline-flex items-center gap-1.5',
            'cursor-pointer hover:text-blue-900 dark:hover:text-blue-100',
            'transition-colors'
          )}
        >
          {content}
        </button>
      ) : (
        <span className="inline-flex items-center gap-1.5">{content}</span>
      )}
      {onRemove && (
        <button
          type="button"
          onClick={onRemove}
          className={cn(
            'p-0.5 rounded-full -mr-0.5',
            'text-blue-400 dark:text-blue-500',
            'hover:bg-blue-200 dark:hover:bg-blue-800',
            'hover:text-blue-600 dark:hover:text-blue-300',
            'transition-colors duration-150'
          )}
          title={removeTitle}
        >
          <X size={10} />
        </button>
      )}
    </div>
  );
}
