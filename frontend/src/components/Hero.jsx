import React from "react";
import { motion } from "framer-motion";
import { UploadCloud, ChevronRight, CheckCircle2 } from "lucide-react";
import DemoBadge from "./DemoBadge";

export default function Hero({ onCTAClick }) {
  return (
    <section id="home" className="pt-24 md:pt-28">
      <div className="mx-auto max-w-7xl px-4">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
          className="rounded-3xl border border-white/10 bg-gradient-to-b from-[#0b0e16]/60 to-black/40 p-8 md:p-12 shadow-2xl"
        >
          <div className="flex flex-col md:flex-row items-start md:items-center gap-8">
            <div className="flex-1">
              <h1 className="text-3xl md:text-5xl font-extrabold tracking-tight text-white">
                Discover New Worlds with{" "}
                <span className="text-blue-400">AI</span>
              </h1>

              <p className="mt-4 text-white/80 max-w-2xl">
                From photons to planets: upload light curves, visualize
                transits, and let ExoVision highlight exoplanet candidates with
                scientific clarity.
              </p>

              <div className="mt-6 flex flex-wrap items-center gap-3">
                <button
                  onClick={onCTAClick}
                  className="inline-flex items-center gap-2 rounded-2xl bg-blue-600 hover:bg-blue-500 px-5 py-3 text-sm font-semibold text-white shadow-lg"
                >
                  <UploadCloud className="h-5 w-5" /> Upload Data
                </button>

                <a
                  href="#visualization"
                  className="inline-flex items-center gap-2 rounded-2xl bg-white/10 hover:bg-white/20 px-5 py-3 text-sm font-semibold text-white"
                >
                  Preview Visualization <ChevronRight className="h-4 w-4" />
                </a>
              </div>

              <div className="mt-6 flex items-center gap-3 text-white/70 text-sm">
                <CheckCircle2 className="h-4 w-4 text-green-400" /> Client-only
                demo. No data leaves your browser.
              </div>
            </div>

            <div className="w-full md:w-80">
              <DemoBadge />
            </div>
          </div>
        </motion.div>
      </div>
    </section>
  );
}
