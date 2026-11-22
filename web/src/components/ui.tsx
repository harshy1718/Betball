import { useEffect, useState, type ReactNode } from "react";
import { NavLink, Link } from "react-router-dom";
import Nox from "./Nox";
import { pct, type Golden, type MatchSummary } from "../api";

export function Star({ size = 14 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" aria-hidden="true" style={{ flex: "none" }}>
      <path d="M12 2.8l2.8 5.7 6.3.9-4.5 4.4 1 6.3L12 17.2l-5.6 2.9 1-6.3L2.9 9.4l6.3-.9z" fill="currentColor" />
    </svg>
  );
}

function ChatIcon() {
  return (
    <svg width={16} height={16} viewBox="0 0 24 24" aria-hidden="true" style={{ flex: "none" }}>
      <path
        d="M4 5.5C4 4.1 5.1 3 6.5 3h11C18.9 3 20 4.1 20 5.5v8c0 1.4-1.1 2.5-2.5 2.5H11l-4.5 4v-4h0C5.1 16 4 14.9 4 13.5z"
        fill="currentColor"
      />
    </svg>
  );
}

export function Header() {
  return (
    <header className="header">
      <Link to="/" className="brand" aria-label="nox home">
        <Nox size={52} ball={false} label="Nox" />
        <span>
          <span className="brand-name">nox</span>
          <span className="brand-tag">football predictions, all night</span>
        </span>
      </Link>
      <nav aria-label="Main" className="nav">
        <NavLink to="/" end className="pill">
          Next 5 predictions
        </NavLink>
        <NavLink to="/fixtures" className="pill">
          Fixtures
        </NavLink>
        <NavLink to="/golden" className="pill">
          <span className="gold-icon">
            <Star size={15} />
          </span>
          Golden calls
        </NavLink>
        <NavLink to="/ask" className="pill pill-ask">
          <ChatIcon />
          Ask Nox
        </NavLink>
        <NavLink to="/desk" className="pill pill-desk" title="Private: only on this PC">
          <span aria-hidden="true">🔒</span>
          Desk
        </NavLink>
      </nav>
    </header>
  );
}

export function Page({ children, trained }: { children: ReactNode; trained?: string }) {
  return (
    <div className="page">
      <div className="wrap">
        <Header />
        <main className="stack-36">{children}</main>
        <footer className="footer">
          Nox reads our model, trained on Premier League, Serie A and La Liga results
          {trained ? ` up to ${trained}` : ""}. Predictions are chances, not certainties.
        </footer>
      </div>
    </div>
  );
}

/** Home / draw / away bar. Segments grow in from zero when it first appears. */
export function Bar({ m, height = 14, labels = false }: { m: MatchSummary; height?: number; labels?: boolean }) {
  const [on, setOn] = useState(false);
  useEffect(() => {
    const t = requestAnimationFrame(() => setOn(true));
    return () => cancelAnimationFrame(t);
  }, []);
  const { home, draw, away } = m.probs;
  const seg = (p: number, cls: string) => <span className={`seg ${cls}`} style={{ flexGrow: on ? p : 0.0001 }} />;
  return (
    <div className="stack-8">
      <div
        className="bar"
        style={{ height }}
        role="img"
        aria-label={`${m.home} ${pct(home)}, draw ${pct(draw)}, ${m.away} ${pct(away)}`}
      >
        {seg(home, "home")}
        {seg(draw, "draw")}
        {seg(away, "away")}
      </div>
      {labels && (
        <div className="bar-labels">
          <span>
            <i className="dot home" />
            {m.home} {pct(home)}
          </span>
          <span>
            <i className="dot draw" />
            Draw {pct(draw)}
          </span>
          <span>
            <i className="dot away" />
            {m.away} {pct(away)}
          </span>
        </div>
      )}
    </div>
  );
}

export function Bubble({ children, tail = "left", className = "" }: { children: ReactNode; tail?: "left" | "top"; className?: string }) {
  return <div className={`bubble tail-${tail} ${className}`}>{children}</div>;
}

export function GoldChip({ g }: { g: Golden }) {
  return (
    <span className="gold-chip">
      <Star size={13} />
      Golden: {g.kind === "cs" ? `${g.call} score` : g.call}
    </span>
  );
}

export function Loading({ what = "the numbers" }: { what?: string }) {
  return (
    <div className="state" role="status">
      <Nox mood="unsure" size={140} animate />
      <p>Nox is reading {what}…</p>
    </div>
  );
}

export function Problem({ error }: { error: string }) {
  const noApi = /fetch|Failed|NetworkError|50\d/.test(error);
  return (
    <div className="state" role="alert">
      <Nox mood="shock" size={140} />
      <p>
        <b>Nox couldn’t get the predictions.</b>
        <br />
        {noApi ? "Is the API running? Start it with: football-edge serve" : error}
      </p>
    </div>
  );
}
