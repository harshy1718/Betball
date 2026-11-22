import { Link } from "react-router-dom";
import Nox from "../components/Nox";
import { Bubble, Loading, Page, Problem, Star } from "../components/ui";
import { api, pct, useApi, type GoldenPage } from "../api";

export default function Golden() {
  const res = useApi<GoldenPage>(api.golden, "golden");
  if (res.error) return <Page><Problem error={res.error} /></Page>;
  if (res.loading || !res.data) return <Page><Loading what="the golden calls" /></Page>;
  const { calls, bars, tracker } = res.data;
  const days: [string, typeof calls][] = [];
  for (const m of calls) {
    const last = days[days.length - 1];
    if (last && last[0] === m.day_ist) last[1].push(m);
    else days.push([m.day_ist, [m]]);
  }
  return (
    <Page>
      <section aria-label="Golden calls" className="hero">
        <div className="hero-nox">
          <Nox size={210} animate />
        </div>
        <div className="hero-copy stack-12">
          <h1 className="h1 row-14 align-center">
            <span className="t-gold">
              <Star size={44} />
            </span>
            Golden calls
          </h1>
          <Bubble className="bubble-sm">
            “These are the ranges where my model did best in testing. They’re still unproven, so we track every one in
            the open.”
          </Bubble>
        </div>
      </section>
      {days.length === 0 && <p className="muted">No golden calls in the next few days.</p>}
      {days.map(([day, ms]) => (
        <section key={day} aria-label={`Golden calls ${day}`} className="stack-14">
          <h2>
            {day}: {ms.length} {ms.length === 1 ? "match" : "matches"}
          </h2>
          <div className="cards">
            {ms.map((m) => (
              <Link key={m.id} to={`/match/${m.id}`} className="card golden-card wide">
                <span className="kicker small row-8 align-center">
                  <Star size={16} />
                  GOLDEN · {m.time_ist} IST · {m.league_name.toUpperCase()}
                </span>
                <span className="teams-lg">
                  {m.home} v {m.away}
                </span>
                {m.golden.map((g) => (
                  <div key={g.kind} className="stack-4">
                    <span className="golden-call">
                      {g.kind === "cs" ? `${g.call} score` : g.call}
                    </span>
                    <span>
                      {g.title}: {pct(g.p)} chance
                    </span>
                    <span className="muted small">Why it’s golden: {g.rule}.</span>
                  </div>
                ))}
                <span className="muted small">
                  Result chances: {m.home} {pct(m.probs.home)} · Draw {pct(m.probs.draw)} · {m.away} {pct(m.probs.away)}
                </span>
              </Link>
            ))}
          </div>
        </section>
      ))}
      <section aria-label="How a call turns golden" className="stack-14">
        <h2>How a call turns golden</h2>
        <div className="cards">
          {bars.map((b) => (
            <div key={b.kind} className="card">
              <span className="card-title">{b.title}</span>
              <span className="muted">{b.text}</span>
            </div>
          ))}
        </div>
      </section>
      <section aria-label="Tracker" className="stack-14">
        <h2>The tracker</h2>
        <div className="cards">
          {tracker.map((t) => (
            <div key={t.title} className="card">
              <span className="card-title">{t.title}</span>
              <span className="muted">
                {t.logged} logged · {t.settled} settled
                {t.settled ? ` · ${t.came_true} came true (${pct(t.came_true / t.settled)})` : ""}
              </span>
            </div>
          ))}
        </div>
        <span className="muted small">Every golden call is marked right or wrong after the final whistle.</span>
      </section>
    </Page>
  );
}
