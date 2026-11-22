import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { BrowserRouter, Route, Routes } from "react-router-dom";
import Home from "./pages/Home";
import Fixtures from "./pages/Fixtures";
import Match from "./pages/Match";
import Golden from "./pages/Golden";
import Ask from "./pages/Ask";
import Desk from "./pages/Desk";
import "./styles.css";

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/fixtures" element={<Fixtures />} />
        <Route path="/match/:id" element={<Match />} />
        <Route path="/golden" element={<Golden />} />
        <Route path="/ask" element={<Ask />} />
        <Route path="/desk" element={<Desk />} />
        <Route path="*" element={<Home />} />
      </Routes>
    </BrowserRouter>
  </StrictMode>,
);
