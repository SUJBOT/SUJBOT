import React from 'react';
import { Translation } from 'react-i18next';

interface ErrorBoundaryProps {
  children: React.ReactNode;
  fallback?: React.ReactNode;
  onError?: (error: Error, errorInfo: React.ErrorInfo) => void;
}

interface ErrorBoundaryState {
  hasError: boolean;
  error: Error | null;
}

export class ErrorBoundary extends React.Component<ErrorBoundaryProps, ErrorBoundaryState> {
  constructor(props: ErrorBoundaryProps) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error: Error): ErrorBoundaryState {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, errorInfo: React.ErrorInfo) {
    console.error('ErrorBoundary caught:', error, errorInfo);
    this.props.onError?.(error, errorInfo);
  }

  handleReset = () => {
    this.setState({ hasError: false, error: null });
  };

  render() {
    if (this.state.hasError) {
      if (this.props.fallback) {
        return this.props.fallback;
      }

      return (
        <Translation>
          {(t) => (
            <div className="flex items-center justify-center p-8">
              <div className="text-center max-w-md">
                <div className="text-4xl mb-4">&#9888;&#65039;</div>
                <h2 className="text-lg font-semibold text-accent-900 dark:text-accent-100 mb-2">
                  {t('errors.somethingWentWrong')}
                </h2>
                <p className="text-sm text-accent-600 dark:text-accent-400 mb-4">
                  {this.state.error?.message || t('common.error')}
                </p>
                <button
                  onClick={this.handleReset}
                  className="px-4 py-2 rounded-lg bg-accent-900 dark:bg-accent-100 text-white dark:text-accent-900 hover:bg-accent-800 dark:hover:bg-accent-200 transition-colors duration-200 text-sm font-medium"
                >
                  {t('errors.tryAgain')}
                </button>
              </div>
            </div>
          )}
        </Translation>
      );
    }

    return this.props.children;
  }
}
