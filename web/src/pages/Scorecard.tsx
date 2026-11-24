// The live scorecard: every prediction logged before kickoff, graded after the match.
// All figures come from /api/desk/scorecard (src/football_edge/tracking/scorecard.py).
import Nox from "../components/Nox";
import { Loading, Page, PrivateTabs, Problem } from "../components/ui";
import { pct, tracking, useApi, type ScoreRow, type Scorecard as Card } from "../api";

const num = (x: number | null, d = 3) => (x == null ? "–" : x.toFixed(d));

function Skill({ r }: { r: ScoreRow }) {
  if (r.skill == null) return <span className="muted">–</span>;
  const s = Math.round(r.skill * 100);
  return <span className={s > 2 ? "t-ok" : s < -2 ? "t-away" : "muted"}>{s > 0 ? `+${s}` : s}%</span>;
}

function Market({ r }: { r: ScoreRow }) {
  if (r.market_brier == null || r.model_brier_same == null) return <span className="muted">no prices</span>;
  const better = r.model_brier_same < r.market_brier;
  return (
    <span className={better ? "t-ok" : "t-away"}>
      {num(r.model_brier_same)} vs {num(r.market_brier)}
      <span className="muted small"> ({r.market_n})</span>
    </span>
  );
}

function Calibration({ card }: { card: Card }) {
  return (
    <div className="calib stack-8">
      {card.calibration.map((b) => (
        <div key={b.band} className="calib-row">
          <span className="calib-band">{b.band}</span>
          <div className="calib-bars" aria-label={`Said ${pct(b.avg_p ?? 0)}, came in ${pct(b.hit ?? 0)}`}>
            <span className="calib-said" style={{ width: `${(b.avg_p ?? 0) * 100}%` }} />
            <span className="calib-hit" style={{ width: `${(b.hit ?? 0) * 100}%` }} />
          </div>
          <span className="small calib-nums">
            said <b>{pct(b.avg_p ?? 0)}</b> · came in <b>{pct(b.hit ?? 0)}</b> <span className="muted">({b.n})</span>
          </span>
        </div>
      ))}
    </div>
  );
}

function Body({ card }: { card: Card }) {
  if (!card.graded_matches) {
    return (
      <section className="card empty-card row-14 align-center">
        <Nox mood="unsure" size={110} ball={false} />
        <div className="stack-6">
          <b>Nothing graded yet.</b>
          <span className="muted">
            {card.since
              ? `The log started on ${card.since} with ${card.logged_matches} matches. Grades arrive once those matches are played and the daily run has fetched the results.`
              : "Nothing logged yet. The daily run (or opening the desk) logs every prediction before kickoff."}
          </span>
        </div>
      </section>
    );
  }
  const few = (n: number) => n < card.min_sample;
  return (
    <>
      <section className="stack-14" aria-label="Accuracy by market">
        <div className="section-head">
          <h2>Accuracy by market</h2>
          <span className="legend">
            Accuracy score: lower is better · vs league = how much better than just knowing the league’s own rate
          </span>
        </div>
        <div className="table-wrap">
          <table className="table">
            <thead>
              <tr>
                <th>Market</th>
                <th>Graded</th>
                <th>Avg chance</th>
                <th>Came in</th>
                <th>Accuracy score</th>
                <th>vs league</th>
                <th>Model vs market</th>
                <th>Verdict</th>
              </tr>
            </thead>
            <tbody>
              {[...card.markets, ...(card.overall ? [card.overall] : [])].map((r) => (
                <tr key={r.family} className={r.family === "all" ? "row-total" : few(r.n) ? "row-few" : ""}>
                  <td>
                    <b>{r.name}</b>
                  </td>
                  <td>
                    {r.n}
                    <span className="muted small"> · {r.matches} matches</span>
                  </td>
                  <td>{r.avg_p == null ? "–" : pct(r.avg_p)}</td>
                  <td>{r.hit == null ? "–" : pct(r.hit)}</td>
                  <td>{num(r.brier)}</td>
                  <td>
                    <Skill r={r} />
                  </td>
                  <td>
                    <Market r={r} />
                  </td>
                  <td className="small">{r.verdict}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      <section className="stack-14" aria-label="Calibration">
        <div className="section-head">
          <h2>When Nox says X%, does it happen X% of the time?</h2>
          <span className="legend">Grey: the chance we gave · amber: how often it came in</span>
        </div>
        <Calibration card={card} />
      </section>

      <div className="cards">
        <section className="card stack-10" aria-label="By confidence">
          <span className="label">By confidence label</span>
          {card.confidence.map((c) => (
            <span key={c.label}>
              <b>{c.label}</b>: said {pct(c.avg_p ?? 0)}, came in {pct(c.hit ?? 0)}{" "}
              <span className="muted small">({c.n})</span>
            </span>
          ))}
        </section>
        <section className="card stack-10" aria-label="Desk slips">
          <span className="label">The desk’s slips</span>
          {card.slips.length ? (
            card.slips.map((s) => (
              <span key={s.title}>
                <b>{s.title}</b>: {s.won} of {s.n} came in ({pct(s.hit ?? 0)}), expected {pct(s.expected ?? 0)}
                {s.floor != null && <span className="muted small"> · promise {pct(s.floor)}+</span>}
              </span>
            ))
          ) : (
            <span className="muted small">No graded slips yet.</span>
          )}
        </section>
        <section className="card stack-10" aria-label="Posted calls">
          <span className="label">Calls posted on Instagram</span>
          <span>
            {card.posted.won} of {card.posted.settled} came in
            {card.posted.n > card.posted.settled && (
              <span className="muted small"> · {card.posted.n - card.posted.settled} waiting</span>
            )}
          </span>
        </section>
      </div>
    </>
  );
}

export default function Scorecard() {
  const card = useApi<Card>(tracking.scorecard, "scorecard");
  return (
    <Page>
      <PrivateTabs />
      <section className="stack-10" aria-label="Scorecard">
        <h1 className="h1">Scorecard</h1>
        <p className="muted desk-intro">
          Every prediction is logged before kickoff and graded after the match, on matches the model had never seen. Only the
          last prediction before kickoff counts. Figures with fewer than {card.data?.min_sample ?? 30} graded predictions are
          too few to judge.
        </p>
        {card.data && (
          <div className="row-8 wrap-row">
            <span className="chip chip-static">Logging since {card.data.since ?? "–"}</span>
            <span className="chip chip-static">{card.data.logged_matches} matches logged</span>
            <span className="chip chip-static">{card.data.graded_matches} graded</span>
            <span className={`chip chip-static ${card.data.log_ok ? "t-ok" : "t-away"}`}>
              {card.data.log_ok ? "✓ Log untouched" : "⚠ Log was edited"}
            </span>
          </div>
        )}
        {card.data && !card.data.log_ok && (
          <p className="warning">
            The log no longer matches its hashes:{" "}
            {card.data.log_problems.map((p) => `${p.file} ${p.taken_at}: ${p.what}`).join("; ")}
          </p>
        )}
      </section>
      {card.error ? <Problem error={card.error} /> : card.loading || !card.data ? <Loading what="the scorecard" /> : <Body card={card.data} />}
    </Page>
  );
}
