import React from "react";
import { createRoot } from "react-dom/client"; // used to mount app
import App from "./App.jsx"; // root component
import "./index.css"; // tailwind styles

const root = createRoot(document.getElementById("root"));
root.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
