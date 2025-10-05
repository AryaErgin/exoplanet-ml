import React from "react";

export default function Footer() {
  return (
    <footer className="py-8 border-t border-white/10 text-center text-white/60">
      <div className="mx-auto max-w-7xl px-4">
        <p>
          © {new Date().getFullYear()} ExoVision AI • Built for NASA Space Apps
        </p>
      </div>
    </footer>
  );
}
