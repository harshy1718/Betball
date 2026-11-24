import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { BrowserRouter, Route, Routes } from "react-router-dom";
import Home from "./pages/Home";
import Fixtures from "./pages/Fixtures";
import Match from "./pages/Match";
import Golden from "./pages/Golden";
import Ask from "./pages/Ask";
import Desk from "./pages/Desk";
import Scorecard from "./pages/Scorecard";
import Posts from "./pages/Posts";
import "@fontsource/fredoka/500.css";
import "@fontsource/fredoka/600.css";
import "@fontsource/fredoka/700.css";
import "@fontsource/nunito/400.css";
import "@fontsource/nunito/600.css";
import "@fontsource/nunito/700.css";
import "@fontsource/nunito/800.css";
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
        <Route path="/scorecard" element={<Scorecard />} />
        <Route path="/posts" element={<Posts />} />
        <Route path="*" element={<Home />} />
      </Routes>
    </BrowserRouter>
  </StrictMode>,
);
