import { Link } from "react-router-dom";
import Nox from "../components/Nox";
import { Bar, Bubble, GoldChip, Loading, Page, Problem, Star } from "../components/ui";
import { api, pct, useApi, type Meta, type MatchSummary } from "../api";
import { pickText, sentence, soon } from "../format";

function Hero({ m, golden }: { m: MatchSummary; golden: number }) {
  return (
    <section aria-label="The big match" className="hero">
      <div className="hero-nox">
        <Nox mood={m.mood} size={320} animate />
      </div>
      <div className="hero-copy stack-16">
        <span className="kicker">
          {soon(m.kickoff_utc) ? "TONIGHT'S BIG ONE" : "THE BIG ONE"} · {m.league_name.toUpperCase()} · {m.day_ist.toUpperCase()} {m.time_ist} IST
        </span>
        <Bubble>
          <span className="bubble-lead">
            {m.home} v {m.away}? I’m going
          </span>
          <span className="bubble-big">
            {m.pick.team === "Draw" ? "a draw" : m.pick.team}, {pct(m.pick.p)}
          </span>
          <span className="bubble-small">
            Most likely score {m.top_score.score}. Expected goals {m.exp_goals.home.toFixed(2)} to{" "}
            {m.exp_goals.away.toFixed(2)}. Confidence {m.pick.confidence}: {sentence(m.pick.why)}
          </span>
        </Bubble>
        <Bar m={m} height={16} labels />
        <div className="row-10">
          <Link to={`/match/${m.id}`} className="btn btn-cream">
            Open match preview
          </Link>
          {golden > 0 && (
            <Link to="/golden" className="btn btn-gold-outline">
              <Star size={15} />
              {golden} golden {golden === 1 ? "call" : "calls"} coming up
            </Link>
          )}
        </div>
      </div>
    </section>
  );
}

function Card({ m, selected }: { m: MatchSummary; selected: boolean }) {
  return (
    <Link to={`/match/${m.id}`} className={`card next-card${selected ? " selected" : ""}`}>
      <span className="meta-line">
        {m.time_ist} IST · {m.league_name}
      </span>
      <span className="teams">
        {m.home}
        <br />
        <span className="muted">v</span> {m.away}
      </span>
      <div className="row-10 align-center">
        <Nox mood={m.mood} size={64} ball={false} />
        <span className="quote">“{m.line}”</span>
      </div>
      <Bar m={m} height={10} />
      <span className="pick">{pickText(m)}</span>
      {m.golden.map((g) => (
        <GoldChip key={g.kind} g={g} />
      ))}
    </Link>
  );
}

export default function Home() {
  const meta = useApi<Meta>(api.meta, "meta");
  const next = useApi<MatchSummary[]>(() => api.next(5), "next5");
  if (next.error) return <Page><Problem error={next.error} /></Page>;
  if (next.loading || !next.data) return <Page><Loading /></Page>;
  const list = next.data;
  if (!list.length)
    return (
      <Page trained={meta.data?.trained_up_to}>
        <div className="state">
          <Nox mood="unsure" size={160} />
          <p>No matches in the next few days. Nox is having a rare early night.</p>
        </div>
      </Page>
    );
  const hero = [...list].sort((a, b) => b.big - a.big)[0];
  const golden = list.filter((m) => m.golden.length).length;
  return (
    <Page trained={meta.data?.trained_up_to}>
      {meta.data?.warning && <p className="warning">{meta.data.warning}</p>}
      <Hero m={hero} golden={golden} />
      <section aria-label="Next 5 matches" className="stack-16">
        <div className="section-head">
          <h2>Nox’s next {list.length}</h2>
          <span className="legend">
            Bar: <b className="t-home">home</b> · <b className="t-draw">draw</b> · <b className="t-away">away</b> · Nox
            waves when he’s sure and scratches his head on close calls
          </span>
        </div>
        <div className="cards">
          {list.map((m) => (
            <Card key={m.id} m={m} selected={m.id === hero.id} />
          ))}
        </div>
      </section>
    </Page>
  );
}
