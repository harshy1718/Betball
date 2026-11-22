import { Link, useParams } from "react-router-dom";
import Nox from "../components/Nox";
import { Bar, Bubble, Loading, Page, Problem, Star } from "../components/ui";
import { api, pct, useApi, type MatchDetail } from "../api";
import { sentence } from "../format";

const n1 = (x: number | null) => (x == null ? "–" : x.toFixed(1));
const p0 = (x: number | null) => (x == null ? "–" : pct(x));

function Stat({ label, value, note }: { label: string; value: string; note: string }) {
  return (
    <div className="stat">
      <span className="stat-label">{label}</span>
      <span className="stat-value">{value}</span>
      <span className="stat-note">{note}</span>
    </div>
  );
}

function Grid({ m }: { m: MatchDetail }) {
  const max = Math.max(...m.grid.flat());
  const top = new Set(m.top_scores.map((t) => t.score));
  return (
    <div className="grid" role="table" aria-label={`Chance of each score, ${m.home} goals by ${m.away} goals`}>
      <span />
      {m.grid[0].map((_, j) => (
        <span key={j} className="grid-head">
          {j}
        </span>
      ))}
      {m.grid.map((row, i) => (
        <div key={i} role="row" style={{ display: "contents" }}>
          <span className="grid-head side">{i}</span>
          {row.map((p, j) => {
            const a = 0.08 + (0.92 * p) / max;
            return (
              <span
                key={j}
                role="cell"
                className={`cell${top.has(`${i}-${j}`) ? " top" : ""}`}
                style={{ background: `rgba(255,183,43,${a.toFixed(2)})`, color: a > 0.45 ? "#0F1530" : "#F6F3FF" }}
                title={`${i}-${j}: ${(100 * p).toFixed(1)}%`}
              >
                {(100 * p).toFixed(1)}%
              </span>
            );
          })}
        </div>
      ))}
    </div>
  );
}

function Form({ team, f }: { team: string; f: MatchDetail["form"]["home"] }) {
  return (
    <div className="stack-6">
      <span className="label">{team} · last {f.results.length}, oldest first</span>
      <div className="row-6">
        {f.results.map((r, i) => (
          <span key={i} className={`form form-${r}`} aria-label={{ W: "win", D: "draw", L: "loss" }[r]}>
            {r}
          </span>
        ))}
      </div>
    </div>
  );
}

export default function Match() {
  const { id = "" } = useParams();
  const res = useApi<MatchDetail>(() => api.match(id), id);
  if (res.error)
    return (
      <Page>
        <Problem error={res.error.includes("404") || res.error.includes("no upcoming") ? "That match isn’t in the next few days any more." : res.error} />
      </Page>
    );
  if (res.loading || !res.data) return <Page><Loading what="this match" /></Page>;
  const m = res.data;
  return (
    <Page>
      <Link to="/fixtures" className="back">
        ‹ Back to fixtures
      </Link>
      <section aria-label="Prediction" className="hero match-hero">
        <div className="hero-nox">
          <Nox mood={m.mood} size={250} animate />
        </div>
        <div className="hero-copy stack-16">
          <span className="kicker">
            {m.league_name.toUpperCase()} · {m.day_ist.toUpperCase()} · {m.time_ist} IST
          </span>
          <h1 className="h1 match-title">
            {m.home} v {m.away}
          </h1>
          <div className="bigs">
            <span>
              <b className="t-home">{pct(m.probs.home)}</b>
              {m.home} win
            </span>
            <span>
              <b className="t-draw">{pct(m.probs.draw)}</b>
              Draw
            </span>
            <span>
              <b className="t-away">{pct(m.probs.away)}</b>
              {m.away} win
            </span>
          </div>
          <Bar m={m} height={16} />
          <Bubble className="bubble-sm">
            “{m.line} Confidence {m.pick.confidence}: {sentence(m.pick.why)}”
          </Bubble>
        </div>
      </section>
      <div className="stats">
        <Stat label="Expected goals" value={`${m.exp_goals.home.toFixed(2)} – ${m.exp_goals.away.toFixed(2)}`} note={`${m.home} – ${m.away}`} />
        <Stat label="Over 2.5 goals" value={pct(m.over25)} note={`Goals call: ${m.goals_call.text}, confidence ${m.goals_call.confidence}`} />
        <Stat label="Both teams score" value={pct(m.btts)} note="chance both teams score" />
        <Stat label="Corners" value={n1(m.corners.expected)} note={`expected · over 9.5: ${p0(m.corners.over_9_5)}`} />
        <Stat label="Cards" value={n1(m.cards.expected)} note={`expected · over 3.5: ${p0(m.cards.over_3_5)}`} />
      </div>
      <div className="two-col">
        <section aria-label="Scoreline chances" className="stack-10 col-wide">
          <h2>Every score, its chance</h2>
          <span className="muted small">
            Rows: {m.home} goals · columns: {m.away} goals · outlined: the three most likely (
            {m.top_scores.map((t) => `${t.score} ${pct(t.p)}`).join(", ")})
          </span>
          <Grid m={m} />
        </section>
        <section aria-label="Form and meetings" className="stack-16 col">
          <h2>Form</h2>
          <Form team={m.home} f={m.form.home} />
          <Form team={m.away} f={m.form.away} />
          <div className="stack-6">
            <span className="label">Last {m.meetings.length} meetings</span>
            {m.meetings.length ? (
              <ul className="meetings">
                {m.meetings.map((x, i) => (
                  <li key={i}>
                    {x.date} · {x.home} {x.score} {x.away}
                  </li>
                ))}
              </ul>
            ) : (
              <span className="muted">No meetings in our data.</span>
            )}
          </div>
        </section>
      </div>
      <section aria-label="Ask Nox about this match" className="ask-strip">
        <Nox mood="sure" size={64} ball={false} />
        <div className="stack-10">
          <h2 className="h-sm">Ask Nox about this match</h2>
          <div className="row-8">
            {[
              `Will both teams score in ${m.home} v ${m.away}?`,
              `How many goals do you expect in ${m.home} v ${m.away}?`,
              `How has ${m.away} been playing lately?`,
            ].map((q) => (
              <Link key={q} to={`/ask?q=${encodeURIComponent(q)}`} className="chip chip-link">
                {q}
              </Link>
            ))}
          </div>
        </div>
      </section>
      {m.golden.length ? (
        <section aria-label="Golden calls" className="stack-10">
          <h2 className="row-10 align-center">
            <span className="t-gold">
              <Star size={26} />
            </span>
            Golden calls in this match
          </h2>
          <div className="cards">
            {m.golden.map((g) => (
              <div key={g.kind} className="card golden-card">
                <span className="kicker small">{g.title.toUpperCase()}</span>
                <span className="golden-call">{g.call}</span>
                <span>{pct(g.p)} chance</span>
                <span className="muted small">Why it’s golden: {g.rule}.</span>
              </div>
            ))}
          </div>
        </section>
      ) : (
        <div className="note">
          <Nox mood="unsure" size={56} ball={false} />
          <span className="muted">No golden call in this match.</span>
        </div>
      )}
    </Page>
  );
}
