import { useMemo, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import Nox from "../components/Nox";
import { Bar, Bubble, GoldChip, Loading, Page, Problem, Star } from "../components/ui";
import { api, pct, useApi, type MatchSummary } from "../api";
import { pickText, started } from "../format";

const LEAGUES = [
  ["", "All leagues"],
  ["E0", "Premier League"],
  ["SP1", "La Liga"],
  ["I1", "Serie A"],
] as const;

function Preview({ m }: { m: MatchSummary }) {
  return (
    <aside aria-label="Selected match" className="preview stack-14">
      <span className="kicker small">
        SELECTED · {m.day_ist.toUpperCase()} {m.time_ist} IST · {m.league_name.toUpperCase()}
      </span>
      <span className="preview-title">
        {m.home} v {m.away}
      </span>
      <div className="row-14 align-center">
        <Nox mood={m.mood} size={110} ball={false} />
        <Bubble className="bubble-sm">“{m.line}”</Bubble>
      </div>
      <Bar m={m} height={14} labels />
      <div className="mini-grid">
        <span>
          Most likely
          <b>{m.top_score.score}</b>
        </span>
        <span>
          Expected goals
          <b>
            {m.exp_goals.home.toFixed(2)} – {m.exp_goals.away.toFixed(2)}
          </b>
        </span>
        <span>
          Over 2.5 goals
          <b>{pct(m.over25)}</b>
        </span>
        <span>
          Confidence
          <b>{m.pick.confidence}</b>
        </span>
      </div>
      {m.golden.map((g) => (
        <GoldChip key={g.kind} g={g} />
      ))}
      <Link to={`/match/${m.id}`} className="btn btn-gold">
        Open full preview
      </Link>
    </aside>
  );
}

export default function Fixtures() {
  const all = useApi<MatchSummary[]>(api.matches, "matches");
  const [league, setLeague] = useState("");
  const [picked, setPicked] = useState<string | null>(null);
  const go = useNavigate();
  // On a phone the preview sits below a long list, so a tap opens the match page instead.
  const choose = (m: MatchSummary) =>
    window.matchMedia("(max-width: 900px)").matches ? go(`/match/${m.id}`) : setPicked(m.id);
  const shown = useMemo(() => (all.data ?? []).filter((m) => !league || m.league === league), [all.data, league]);
  const days = useMemo(() => {
    const out: [string, MatchSummary[]][] = [];
    for (const m of shown) {
      const last = out[out.length - 1];
      if (last && last[0] === m.day_ist) last[1].push(m);
      else out.push([m.day_ist, [m]]);
    }
    return out;
  }, [shown]);
  if (all.error) return <Page><Problem error={all.error} /></Page>;
  if (all.loading) return <Page><Loading what="the fixtures" /></Page>;
  const selected = shown.find((m) => m.id === picked) ?? shown.find((m) => !started(m)) ?? shown[0];
  return (
    <Page>
      <section aria-label="Fixtures" className="stack-18">
        <div className="section-head">
          <h1 className="h1">Fixtures</h1>
          <div role="group" aria-label="League" className="row-8">
            {LEAGUES.map(([code, name]) => (
              <button key={code} type="button" aria-pressed={league === code} className="chip" onClick={() => setLeague(code)}>
                {name}
              </button>
            ))}
          </div>
        </div>
        <div className="fixtures-layout">
          <div className="fixture-list">
            {days.length === 0 && <p className="muted">No matches for this league in the next few days.</p>}
            {days.map(([day, ms]) => (
              <div key={day} className="stack-10">
                <h2 className="day">{day}</h2>
                {ms.map((m) => (
                  <button
                    key={m.id}
                    type="button"
                    className={`fixture${selected?.id === m.id ? " selected" : ""}`}
                    aria-pressed={selected?.id === m.id}
                    onClick={() => choose(m)}
                  >
                    <span className="f-time">{m.time_ist}</span>
                    <span className="f-teams">
                      <b>
                        {m.home} v {m.away}
                      </b>
                      <small>
                        {m.league_name}
                        {started(m) && <span className="live"> · kicked off</span>}
                      </small>
                    </span>
                    <span className="f-bar">
                      <Bar m={m} height={10} />
                    </span>
                    <span className="f-pick">{pickText(m)}</span>
                    <span className="f-star" aria-label={m.golden.length ? "Golden call" : undefined}>
                      {m.golden.length > 0 && <Star size={20} />}
                    </span>
                  </button>
                ))}
              </div>
            ))}
          </div>
          {selected && <Preview m={selected} />}
        </div>
      </section>
    </Page>
  );
}
