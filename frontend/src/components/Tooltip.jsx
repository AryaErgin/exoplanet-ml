import React from "react";

export default function Tooltip({ label, children }) {
  return (
    <span className="relative inline-flex items-center group">
      {children}
      <span className="pointer-events-none absolute -top-9 left-1/2 -translate-x-1/2 whitespace-nowrap rounded-md bg-black/80 text-white text-xs px-2 py-1 opacity-0 group-hover:opacity-100 transition">
        {label}
      </span>
    </span>
  );
}
