import { cn } from '../../design-system/utils/cn';

export function GradientBackground() {
  return (
    <>
      <div
        className={cn(
          'absolute inset-0 -z-10',
          'bg-white dark:bg-accent-950'
        )}
        style={{ background: 'var(--gradient-mesh-light)' }}
      />
      <div
        className="absolute inset-0 -z-10 dark:block hidden"
        style={{ background: 'var(--gradient-mesh-dark)' }}
      />
      <div
        className="absolute inset-0 -z-10"
        style={{ background: 'var(--gradient-light)' }}
      />
      <div
        className="absolute inset-0 -z-10 dark:block hidden"
        style={{ background: 'var(--gradient-dark)' }}
      />
    </>
  );
}
