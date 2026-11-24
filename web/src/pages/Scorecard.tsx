// The live scorecard: every prediction logged before kickoff, graded after the match.
// Written to be read without a statistics background: every number comes with a sentence
// saying what it means. Figures come from /api/desk/scorecard (tracking/scorecard.py).
import type { ReactNode } from "react";
import Nox from "../components/Nox";
import { Loading, Page, PrivateTabs, Problem } from "../components/ui";
import { pct, tracking, useApi, type GradedCall, type ScoreRow, type Scorecard as Card } from "../api";

const LEAGUES: Record<string, string> = { E0: "Premier League", I1: "Serie A", SP1: "La Liga" };
const n3 = (x: number | null) => (x == null ? "–" : x.toFixed(3));

function day(iso: string) {
  return new Date(iso).toLocaleDateString("en-GB", { weekday: "short", day: "numeric", month: "short", timeZone: "Asia/Kolkata" });
}

function Step({ n, title, children }: { n: number; title: string; children: ReactNode }) {
  return (
    <div className="sc-step">
      <span className="sc-step-n">{n}</span>
      <div className="stack-6">
        <b>{title}</b>
        <span className="small muted">{children}</span>
      </div>
    </div>
  );
}

function HowItWorks({ card }: { card: Card }) {
  return (
    <section className="sc-steps" aria-label="How the scorecard works">
      <Step n={1} title={`${card.logged_matches} matches logged before kickoff`}>
        Every chance Nox gives is saved before the match starts. It can’t be changed afterwards, and anything saved after
        kickoff never counts.
      </Step>
      <Step n={2} title={`${card.upcoming_matches + card.awaiting_results} waiting`}>
        {card.upcoming_matches} still to play
        {card.awaiting_results > 0 && `, ${card.awaiting_results} played and waiting for the result`}. The daily run (10:30)
        fetches results.
      </Step>
      <Step n={3} title={`${card.graded_matches} matches graded`}>
        Each prediction is marked ✓ (it happened) or ✗ (it didn’t), then added up below.
      </Step>
    </section>
  );
}

function Call({ c }: { c?: GradedCall }) {
  if (!c) return null;
  return (
    <span className={`sc-call ${c.won ? "hit" : "miss"}`}>
      {c.won ? "✓" : "✗"} {c.call} <b>{pct(c.p)}</b>
    </span>
  );
}

function Recent({ card }: { card: Card }) {
  return (
    <section className="stack-10" aria-label="Latest graded matches">
      <h2>Latest graded matches</h2>
      <p className="muted sc-explain">
        For each match, Nox’s two main calls: the most likely result and over or under 2.5 goals, with the chance Nox gave.
        One match proves nothing: a 60% call should still miss 4 times in 10. The patterns further down are what matter.
      </p>
      <div className="sc-recent">
        {card.recent.map((r) => (
          <div key={r.id} className="sc-recent-row">
            <span className="sc-recent-when small muted">
              {day(r.kickoff_utc)} · {LEAGUES[r.league] ?? r.league}
            </span>
            <span className="sc-recent-match">
              {r.home} <b className="sc-score">{r.score}</b> {r.away}
            </span>
            <span className="sc-recent-calls">
              <Call c={r.result} />
              <Call c={r.goals} />
            </span>
          </div>
        ))}
      </div>
    </section>
  );
}

function Verdict({ r, min }: { r: ScoreRow; min: number }) {
  if (r.n < min) return <span className="sc-badge wait">Too few to judge</span>;
  const vsMarket = r.market_brier != null && r.model_brier_same != null && r.market_n >= min ? r.model_brier_same - r.market_brier : null;
  if (vsMarket != null && vsMarket < -0.002) return <span className="sc-badge good">Beats the bookmakers</span>;
  if (r.skill != null && r.skill > 0.02) return <span className="sc-badge ok">Beats the league guess</span>;
  if (r.skill != null && r.skill < -0.02) return <span className="sc-badge bad">Worse than the league guess</span>;
  return <span className="sc-badge wait">No better than the league guess</span>;
}

function MarketCard({ r, min }: { r: ScoreRow; min: number }) {
  const gap = r.avg_p != null && r.hit != null ? r.hit - r.avg_p : null;
  const skill = r.skill == null ? null : Math.round(r.skill * 100);
  const market =
    r.market_brier != null && r.model_brier_same != null
      ? { ours: r.model_brier_same, theirs: r.market_brier, rel: Math.round(((r.model_brier_same - r.market_brier) / r.market_brier) * 1000) / 10 }
      : null;
  return (
    <article className="card stack-10 sc-market">
      <div className="row-8 align-center wrap-row sc-market-head">
        <b className="sc-market-name">{r.name}</b>
        <Verdict r={r} min={min} />
      </div>
      <div className="small">
        <b>{r.n}</b> predictions from {r.matches} matches
        {r.n < min && (
          <>
            <div className="sc-progress" aria-label={`${r.n} of ${min} needed`}>
              <span style={{ width: `${Math.min(100, (r.n / min) * 100)}%` }} />
            </div>
            <span className="muted">
              {min - r.n} more needed before this means anything ({min} minimum)
            </span>
          </>
        )}
      </div>
      <dl className="sc-dl">
        <dt>Chance Nox gave, on average</dt>
        <dd>{r.avg_p == null ? "–" : pct(r.avg_p)}</dd>
        <dt>How often it came in</dt>
        <dd>
          {r.hit == null ? "–" : pct(r.hit)}
          {gap != null && (
            <span className="muted small">
              {" "}
              ·{" "}
              {Math.abs(gap) <= 0.03
                ? "close, so the chances were honest"
                : gap < 0
                  ? "lower: Nox was too confident"
                  : "higher: Nox was too cautious"}
            </span>
          )}
        </dd>
        <dt>Accuracy score (lower is better)</dt>
        <dd>{n3(r.brier)}</dd>
        <dt>Just guessing the league’s usual rate</dt>
        <dd>
          {n3(r.base_brier)}
          {skill != null && (
            <span className={`small ${skill > 2 ? "t-ok" : skill < -2 ? "t-away" : "muted"}`}>
              {" "}
              · Nox is {Math.abs(skill)}% {skill >= 0 ? "more" : "less"} accurate
            </span>
          )}
        </dd>
        <dt>The bookmakers’ final prices</dt>
        <dd>
          {market ? (
            <>
              {n3(market.theirs)}{" "}
              <span className={`small ${market.rel < 0 ? "t-ok" : "t-away"}`}>
                · {market.rel < 0 ? `Nox is ${Math.abs(market.rel)}% more accurate` : `bookmakers are ${market.rel}% more accurate`}
              </span>
              <span className="muted small"> (Nox {n3(market.ours)} on the same {r.market_n})</span>
            </>
          ) : (
            <span className="muted small">no bookmaker prices for this market in our data</span>
          )}
        </dd>
      </dl>
    </article>
  );
}

function Calibration({ card }: { card: Card }) {
  const big = [...card.calibration].sort((a, b) => b.n - a.n)[0];
  return (
    <section className="stack-10" aria-label="Calibration">
      <h2>Are Nox’s chances honest?</h2>
      <p className="muted sc-explain">
        All predictions grouped by the chance Nox gave. Grey is the chance, amber is how often it actually happened. When the two
        bars are about the same length, a “70%” from Nox really means 7 in 10.
        {big && big.avg_p != null && big.hit != null && (
          <>
            {" "}
            Example from your log: in the {big.band} row ({big.n} predictions) Nox gave {pct(big.avg_p)} on average, and {pct(big.hit)}{" "}
            came in.
          </>
        )}
      </p>
      <div className="calib stack-8">
        {card.calibration.map((b) => (
          <div key={b.band} className="calib-row">
            <span className="calib-band">{b.band}</span>
            <div className="calib-bars" aria-label={`Said ${pct(b.avg_p ?? 0)}, came in ${pct(b.hit ?? 0)}`}>
              <span className="calib-said" style={{ width: `${(b.avg_p ?? 0) * 100}%` }} />
              <span className="calib-hit" style={{ width: `${(b.hit ?? 0) * 100}%` }} />
            </div>
            <span className="small calib-nums">
              said <b>{pct(b.avg_p ?? 0)}</b> · came in <b>{pct(b.hit ?? 0)}</b>{" "}
              <span className="muted">({b.n}{b.n < card.min_sample ? ", too few" : ""})</span>
            </span>
          </div>
        ))}
      </div>
    </section>
  );
}

function Extras({ card }: { card: Card }) {
  return (
    <div className="cards">
      <section className="card stack-10" aria-label="By confidence">
        <span className="label">Confidence labels</span>
        <span className="small muted">
          High, Medium and Low say how reliable chances like this were in the past. “High” should be the label where said and
          came in are closest.
        </span>
        {card.confidence.map((c) => (
          <span key={c.label}>
            <b>{c.label}</b>: said {pct(c.avg_p ?? 0)}, came in {pct(c.hit ?? 0)}{" "}
            <span className="muted small">({c.n} predictions)</span>
          </span>
        ))}
      </section>
      <section className="card stack-10" aria-label="Desk slips">
        <span className="label">The trader desk’s slips</span>
        <span className="small muted">
          Every single and parlay the desk built, the last version before kickoff. “Expected” is the average chance the desk gave
          them; safe parlays promise 60%+, risky ones 25%+.
        </span>
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
      {card.calibration_check && (
        <section className="card stack-10" aria-label="Calibration check">
          <span className="label">Is the calibration helping?</span>
          <span className="small muted">
            Nox’s chances are adjusted by a calibration learned from past seasons. The raw, unadjusted chance is logged next to
            each one; on the same matches, the lower accuracy score wins. Decided at {card.calibration_check.review_at} graded
            matches.
          </span>
          {card.calibration_check.markets.map((m) => (
            <span key={m.market}>
              <b>{m.market}</b>: adjusted {m.calibrated?.toFixed(3) ?? "–"}, raw {m.raw?.toFixed(3) ?? "–"}
            </span>
          ))}
          <span>
            <b>{card.calibration_check.status}</b>{" "}
            <span className="muted small">({card.calibration_check.matches} matches)</span>
          </span>
        </section>
      )}
      <section className="card stack-10" aria-label="Posted calls">
        <span className="label">Calls posted on Instagram</span>
        <span className="small muted">Only calls marked as posted on the Posts page, hits and misses.</span>
        <span>
          {card.posted.won} of {card.posted.settled} came in
          {card.posted.n > card.posted.settled && <span className="muted small"> · {card.posted.n - card.posted.settled} waiting</span>}
        </span>
      </section>
    </div>
  );
}

function Glossary({ min }: { min: number }) {
  return (
    <section className="card stack-10 sc-glossary" aria-label="What the numbers mean">
      <h2>What the numbers mean</h2>
      <p>
        <b>Accuracy score.</b> For every prediction, take the gap between the chance and what happened, square it, and average.
        Example: Nox says <b>70%</b> Arsenal win. Arsenal win: (1 − 0.70)² = <b>0.09</b>. Arsenal don’t win: (0 − 0.70)² ={" "}
        <b>0.49</b>. Confident and right scores low; confident and wrong is punished hard. Lower is better. Always saying 50% on
        a yes/no question scores 0.25.
      </p>
      <p>
        <b>League guess.</b> The simplest possible forecaster: always give the league’s usual rate over the past 3 years (say,
        home teams win 45% in that league). If Nox can’t beat that, it knows nothing about the teams.
      </p>
      <p>
        <b>Bookmakers’ final prices.</b> The odds just before kickoff, turned into chances with the bookmaker’s margin removed.
        They are free for anyone to read, so this is the bar for predictions worth paying for. Only match result and over/under
        2.5 have prices in our data.
      </p>
      <p>
        <b>Counted once.</b> “Over 2.5” and “under 2.5” are the same question, so only one side is counted; double chance mirrors
        the match result and is left out.
      </p>
      <p>
        <b>Too few to judge.</b> Below {min} graded predictions, luck swamps skill. Expect a few weekends before anything here is
        worth reading.
      </p>
    </section>
  );
}

function Body({ card }: { card: Card }) {
  return (
    <>
      <HowItWorks card={card} />
      {!card.graded_matches ? (
        <section className="card empty-card row-14 align-center">
          <Nox mood="unsure" size={110} ball={false} />
          <div className="stack-6">
            <b>Nothing graded yet.</b>
            <span className="muted">
              {card.since
                ? `The log started on ${card.since}. Grades appear here once those matches are played and the daily run has fetched the results.`
                : "Nothing logged yet. The daily run (or opening the desk) logs every prediction before kickoff."}
            </span>
          </div>
        </section>
      ) : (
        <>
          <Recent card={card} />
          <section className="stack-10" aria-label="Accuracy by market">
            <h2>How accurate is each kind of prediction?</h2>
            <p className="muted sc-explain">
              One card per market. Each compares Nox with two simple rivals: guessing the league’s usual rate, and the
              bookmakers’ final prices. The badge sums it up.
            </p>
            <div className="sc-markets">
              {[...(card.overall ? [card.overall] : []), ...card.markets].map((r) => (
                <MarketCard key={r.family} r={r} min={card.min_sample} />
              ))}
            </div>
          </section>
          <Calibration card={card} />
          <Extras card={card} />
        </>
      )}
      <Glossary min={card.min_sample} />
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
          How good are Nox’s predictions on matches it had never seen? Every prediction is saved before kickoff and checked
          after the final whistle. Nothing is picked out or deleted.
        </p>
        {card.data && (
          <div className="row-8 wrap-row">
            <span className="chip chip-static">Logging since {card.data.since ?? "–"}</span>
            <span className={`chip chip-static ${card.data.log_ok ? "t-ok" : "t-away"}`}>
              {card.data.log_ok ? "✓ Log untouched since it was written" : "⚠ Log was edited"}
            </span>
          </div>
        )}
        {card.data && !card.data.log_ok && (
          <p className="warning">
            The log no longer matches its fingerprints: {card.data.log_problems.map((p) => `${p.file} ${p.taken_at}: ${p.what}`).join("; ")}
          </p>
        )}
      </section>
      {card.error ? <Problem error={card.error} /> : card.loading || !card.data ? <Loading what="the scorecard" /> : <Body card={card.data} />}
    </Page>
  );
}
