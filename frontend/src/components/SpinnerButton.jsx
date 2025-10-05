import React from "react";

export default function SpinnerButton({ loading, children, ...props }) {
  return (
    <button
      {...props}
      disabled={loading || props.disabled}
      className={`px-4 py-2 rounded-lg font-medium transition ${
        props.className ?? ""
      } ${loading ? "opacity-60 cursor-not-allowed" : ""}`}
    >
      {loading ? "Working…" : children}
    </button>
  );
}
