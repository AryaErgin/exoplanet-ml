import React from "react";
import { Telescope, Github } from "lucide-react";

export default function Navbar() {
  const items = [
    { href: "#home", label: "Home" },
    { href: "#upload", label: "Upload" },
    { href: "#visualization", label: "Visualization" },
    { href: "#results", label: "Results" },
    { href: "#catalog", label: "Catalog" },
    { href: "#about", label: "About" },
  ];

  return (
    <nav className="fixed top-0 left-0 right-0 backdrop-blur supports-[backdrop-filter]:bg-[#0b0e16]/70 bg-[#0b0e16]/90 border-b border-white/10 z-50">
      <div className="mx-auto max-w-7xl px-4 py-3 flex items-center justify-between">
        <a href="#home" className="flex items-center gap-2 text-white">
          <Telescope className="h-6 w-6 text-blue-400" />
          <span className="font-bold tracking-wide">ExoVision AI</span>
        </a>

        <div className="hidden md:flex gap-6 text-sm">
          {items.map((it) => (
            <a
              key={it.href}
              href={it.href}
              className="text-white/80 hover:text-white transition-colors"
            >
              {it.label}
            </a>
          ))}
        </div>

        <a
          href="https://github.com"
          target="_blank"
          rel="noreferrer"
          className="inline-flex items-center gap-2 text-white/80 hover:text-white"
        >
          <Github className="h-5 w-5" />
          <span className="hidden sm:inline">Repo</span>
        </a>
      </div>
    </nav>
  );
}
