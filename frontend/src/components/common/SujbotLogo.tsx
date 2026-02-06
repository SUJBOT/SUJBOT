import { cn } from '../../design-system/utils/cn';

interface SujbotLogoProps {
  size?: number;
  className?: string;
  style?: React.CSSProperties;
}

export function SujbotLogo({ size = 40, className, style }: SujbotLogoProps) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 512 512"
      xmlns="http://www.w3.org/2000/svg"
      className={cn(
        'text-accent-900 dark:text-accent-100',
        className
      )}
      style={style}
    >
      <g transform="translate(256 256)" stroke="currentColor" fill="none" strokeLinecap="round">
        <ellipse rx="185" ry="110" strokeWidth="16" />
        <ellipse rx="185" ry="110" strokeWidth="16" transform="rotate(60)" />
        <ellipse rx="185" ry="110" strokeWidth="16" transform="rotate(-60)" />
        <circle r="20" cx="185" cy="0" fill="currentColor" stroke="none" />
        <circle r="20" cx="-92.5" cy="160" fill="currentColor" stroke="none" />
        <circle r="20" cx="-92.5" cy="-160" fill="currentColor" stroke="none" />
        <text
          x="0"
          y="35"
          fontSize="140"
          fontWeight="bold"
          fill="currentColor"
          textAnchor="middle"
          fontFamily="serif"
        >§</text>
      </g>
    </svg>
  );
}
