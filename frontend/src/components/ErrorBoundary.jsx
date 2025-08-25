import React from "react";

export default class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error) {
    return { hasError: true, error };
  }

  componentDidCatch(error, info) {
    console.error("UI ErrorBoundary:", error, info);
  }

  render() {
    if (this.state.hasError) {
      return (
        <div className="m-4 rounded-xl border border-red-500/40 bg-red-500/10 p-4 text-red-200">
          <div className="font-semibold">UI recovered from an error.</div>
          <div className="text-xs mt-1">
            {String(this.state.error?.message || this.state.error)}
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}
