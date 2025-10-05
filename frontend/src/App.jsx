import React, { useEffect, useState } from "react";
import ErrorBoundary from "./components/ErrorBoundary.jsx";
import Starfield from "./components/Starfield.jsx";
import Navbar from "./components/Navbar.jsx";
import Hero from "./components/Hero.jsx";
import UploadPanel from "./components/UploadPanel.jsx";
import VisualizationPanel from "./components/VisualizationPanel.jsx";
import PredictionCard from "./components/PredictionCard.jsx";
import CatalogTable from "./components/CatalogTable.jsx";
import EvaluationPanel from "./components/EvaluationPanel.jsx";
import About from "./components/About.jsx";
import Footer from "./components/Footer.jsx";
import HarnessTestSuite from "./components/HarnessTestSuite.jsx";
import { generateSyntheticLightCurve } from "./lib/utils.js";
import { getRuns } from "./lib/api.js";

export default function App() {
  const [data, setData] = useState(null);
  const [catalog, setCatalog] = useState([]);
  const [lastResult, setLastResult] = useState(null);

  useEffect(() => {
    const demo = generateSyntheticLightCurve(1200, true);
    setData({
      ...demo,
      meta: {
        filename: "synthetic_demo.csv",
        rows: demo.time.length,
        columns: [],
      },
    });
  }, []);

  useEffect(() => {
    const ctrl = new AbortController();
    getRuns(ctrl.signal)
      .then((rows) => Array.isArray(rows) && setCatalog(rows))
      .catch(() => {});
    return () => ctrl.abort();
  }, []);

  const refreshRuns = () => {
    const ctrl = new AbortController();
    getRuns(ctrl.signal)
      .then((rows) => Array.isArray(rows) && setCatalog(rows))
      .catch(() => {});
  };

  return (
    <div className="min-h-screen bg-[#0b0e16] text-white selection:bg-blue-600 selection:text-white">
      <ErrorBoundary>
        <Starfield />
        <Navbar />
        <Hero
          onCTAClick={() =>
            document
              .querySelector("#upload")
              ?.scrollIntoView({ behavior: "smooth" })
          }
        />
        <UploadPanel
          onDataLoaded={(d) => {
            setData(d);
            setLastResult(null);
          }}
        />
        <VisualizationPanel data={data} result={lastResult} />
        <PredictionCard
          data={data}
          onAddToCatalog={refreshRuns}
          onResult={setLastResult}
        />
        <EvaluationPanel />
        <CatalogTable items={catalog} />
        <About />
        <Footer />
        <HarnessTestSuite />
      </ErrorBoundary>
    </div>
  );
}
