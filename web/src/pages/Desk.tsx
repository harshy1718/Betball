// The private trader desk: Nox builds slips, you build your own, the trader reviews them.
// Every number comes from the desk API (src/football_edge/trader); this page only shows them.
import { Fragment, useEffect, useMemo, useRef, useState } from "react";
import { useSearchParams } from "react-router-dom";
import Nox from "../components/Nox";
import { Loading, Page, PrivateTabs, Problem } from "../components/ui";
import { handicapTable } from "../format";
import { desk, pct, useApi, type Board, type DeskLeg, type Mood, type Review, type Slip } from "../api";

const riskClass = (r: string) => `risk risk-${r.toLowerCase().replace(" ", "-")}`;

function Conf({ l }: { l: DeskLeg }) {
  return (
    <span className={`conf conf-${l.confidence.toLowerCase()}`} title={l.skill ? "" : "The model is no better than a plain average on this kind of bet"}>
      {l.confidence}
      {!l.skill && " · no skill"}
    </span>
  );
}

function SlipCard({ slip, bare = false }: { slip: Slip; bare?: boolean }) {
  const keys = slip.legs.map((l) => l.key);
  const [priceText, setPriceText] = useState("");
  const [priced, setPriced] = useState<Slip | null>(null);
  const [review, setReview] = useState<Review | null>(null);
  const [busy, setBusy] = useState<"" | "price" | "review" | "save">("");
  const [note, setNote] = useState("");
  const price = parseFloat(priceText);
  const s = priced ?? slip;
  useEffect(() => {
    setPriced(null);
    setReview(null);
    setNote("");
  }, [keys.join(",")]); // eslint-disable-line react-hooks/exhaustive-deps

  async function run<T>(kind: "price" | "review" | "save", f: () => Promise<T>, done: (x: T) => void) {
    setBusy(kind);
    setNote("");
    try {
      done(await f());
    } catch (e) {
      setNote((e as Error).message);
    } finally {
      setBusy("");
    }
  }
  const okPrice = Number.isFinite(price) && price > 1;
  const mood: Mood = s.price && s.price.verdict.startsWith("below") ? "shock" : s.risk === "Low" || s.risk === "Medium" ? "sure" : "unsure";
  return (
    <article className={`card slip${bare ? " bare" : ""}`}>
      {slip.title && !bare && (
        <div className="stack-4">
          <span className="kicker small">{slip.title.toUpperCase()}</span>
          {slip.why && <span className="muted small">{slip.why}</span>}
        </div>
      )}
      {!(bare && s.kind === "single") && (
      <ul className="slip-legs">
        {s.legs.map((l) => (
          <li key={l.key} className={l.weak ? "weak" : ""}>
            <div className="stack-4">
              <b>{l.label}</b>
              {handicapTable(l) && <span className="hcp-table small">On a 3-way handicap table: {handicapTable(l)}</span>}
              <span className="muted small">
                {l.home} v {l.away}
                {l.market === "cards" && " · sites count cards differently"}
              </span>
            </div>
            <div className="leg-right">
              <span className="leg-p">{pct(l.p)}</span>
              <Conf l={l} />
            </div>
          </li>
        ))}
      </ul>
      )}
      <div className="slip-head">
        <div className="stack-4">
          <span className="muted small">Chance the {s.kind === "single" ? "bet" : "slip"} wins</span>
          <span className="slip-p">{pct(s.p)}</span>
          {s.fair_odds && <span className="muted small">Our fair odds {s.fair_odds.toFixed(2)}</span>}
        </div>
        <span className={riskClass(s.risk)}>{s.risk} risk</span>
      </div>
      <p className="desk-view">{s.desk_view}</p>
      {s.leg_costs.length > 0 && (
        <ul className="costs">
          {s.leg_costs.map((c) => (
            <li key={c.key}>
              Without <b>{c.label}</b>: {pct(c.without)}
            </li>
          ))}
        </ul>
      )}
      <span className="small">
        Stake:{" "}
        <b>
          {s.stake.toFixed(0)} {s.currency}
        </b>{" "}
        <span className="muted">({s.stake_note})</span>
      </span>
      <form
        className="price-row"
        onSubmit={(e) => {
          e.preventDefault();
          if (okPrice) void run("price", () => desk.check(keys, price), setPriced);
        }}
      >
        <label className="small muted" htmlFor={`price-${keys.join("-")}`}>
          Your site’s odds
        </label>
        <input
          id={`price-${keys.join("-")}`}
          inputMode="decimal"
          placeholder="e.g. 2.40"
          value={priceText}
          onChange={(e) => setPriceText(e.target.value)}
        />
        <button type="submit" className="chip" disabled={!okPrice || busy !== ""}>
          {busy === "price" ? "Checking…" : "Check value"}
        </button>
      </form>
      {s.price && (
        <p className={`value ${s.price.verdict.startsWith("above") ? "good" : "bad"}`}>
          At {s.price.odds.toFixed(2)}: {s.price.verdict}, about {s.price.back_per_100.toFixed(0)} back per 100 staked on average.{" "}
          <span className="muted small">{s.price.note}</span>
        </p>
      )}
      <div className="row-8">
        <button type="button" className="btn btn-gold btn-sm" disabled={busy !== ""} onClick={() => void run("review", () => desk.review(keys, okPrice ? price : null), setReview)}>
          {busy === "review" ? "The trader is thinking…" : "Ask the trader"}
        </button>
        <button
          type="button"
          className="btn btn-outline btn-sm"
          disabled={busy !== ""}
          onClick={() => void run("save", () => desk.save(keys, okPrice ? price : null), (r) => setNote(`Saved as ${r.slip_id}. It settles after the match.`))}
        >
          Save slip
        </button>
      </div>
      {note && <p className="small muted">{note}</p>}
      {review && (
        <div className="msg msg-nox">
          <Nox mood={mood} size={44} ball={false} />
          <div className="stack-6 msg-body">
            <div className="bubble bubble-chat tail-left">
              <p>{review.text}</p>
            </div>
            <span className="msg-meta">
              {review.checked ? <span className="ok">✓ Every number checked against the desk</span> : <span className="warn">⚠ Couldn’t check: {review.unverified.join(", ")}</span>}
              {` · ${Math.round(review.seconds)}s`}
            </span>
          </div>
        </div>
      )}
    </article>
  );
}

function SingleRow({ s, showMatch = false, why = false }: { s: Slip; showMatch?: boolean; why?: boolean }) {
  const [open, setOpen] = useState(false);
  const l = s.legs[0];
  const golden = s.title === "Golden single";
  return (
    <div className={`single${open ? " open" : ""}`}>
      <button type="button" className="single-row" aria-expanded={open} onClick={() => setOpen(!open)}>
        <span className="single-label">
          {golden && <span className="t-gold" aria-label="Golden call">★ </span>}
          <b>{l.label}</b>
          {handicapTable(l) && <span className="hcp-table small"> · table {handicapTable(l)}</span>}
          {showMatch && (
            <span className="muted small">
              {" "}
              · {l.home} v {l.away}
            </span>
          )}
          {why && s.why && <span className="single-why muted small">{s.why}</span>}
        </span>
        <span className="leg-p">{pct(l.p)}</span>
        <Conf l={l} />
        <span className={riskClass(s.risk)}>{s.risk}</span>
        <span className="small muted single-stake">
          {s.stake.toFixed(0)} {s.currency}
        </span>
        <span aria-hidden="true" className="muted">{open ? "▴" : "▾"}</span>
      </button>
      {open && <SlipCard slip={s} bare />}
    </div>
  );
}

function MatchBoard({ m, focus = false }: { m: Board["matches"][number]; focus?: boolean }) {
  const { match } = m;
  const golden = m.singles.filter((x) => x.title === "Golden single");
  const good = m.singles.filter((x) => x.title !== "Golden single");
  return (
    <article className="match-board">
      <header className="mb-head">
        <span className="kicker small">
          {match.day_ist.toUpperCase()} · {match.time_ist} IST · {match.league_name.toUpperCase()}
        </span>
        <h3 className="mb-title">
          {match.home} v {match.away}
        </h3>
      </header>
      <div className="mb-body">
        {(golden.length > 0 || focus) && (
          <section className="stack-8 mb-singles mb-golden" aria-label="Golden bets">
            <span className="label">
              <span className="t-gold">★</span> Golden bets <span className="muted small">· unproven, still being tracked</span>
            </span>
            {golden.length ? (
              golden.map((s) => <SingleRow key={s.legs[0].key} s={s} why />)
            ) : (
              <span className="muted small">No golden call in this match.</span>
            )}
          </section>
        )}
        <section className="stack-8 mb-singles" aria-label="Good singles">
          <span className="label">Good singles</span>
          {good.length ? good.map((s) => <SingleRow key={s.legs[0].key} s={s} />) : <span className="muted small">No strong single in this match.</span>}
        </section>
        <div className="mb-parlays">
          {(["safe", "risky"] as const).map((kind) => (
            <section key={kind} className={`stack-8 mb-parlay ${kind}`} aria-label={`${kind} parlay`}>
              <span className="label">{kind === "safe" ? "Safe parlay" : "Risky parlay"}</span>
              <span className="muted small">
                {kind === "safe"
                  ? "Highest payout that still comes in 60%+ of the time"
                  : "Bigger payout, comes in 25%+ of the time, High-confidence legs only"}
              </span>
              {m.parlays[kind] ? (
                <SlipCard slip={m.parlays[kind]!} bare />
              ) : (
                <span className="muted small">
                  {kind === "safe" ? "No mix of strong legs comes in 60%+ while every leg pulls its weight." : "Not enough High-confidence legs in this match for a risky parlay."}
                </span>
              )}
            </section>
          ))}
        </div>
      </div>
    </article>
  );
}

function Saved({ tick }: { tick: number }) {
  const h = useApi(desk.history, `history-${tick}`);
  if (h.error) return <p className="warning">{h.error}</p>;
  if (!h.data) return null;
  const { slips, record } = h.data;
  return (
    <div className="stack-14">
      {record.length > 0 && (
        <div className="cards">
          {record.map((r) => (
            <div key={r.risk} className="card">
              <span className={riskClass(r.risk)}>{r.risk} risk</span>
              <span>
                {r.won} of {r.settled} came in ({pct(r.won / r.settled)}); expected {pct(r.expected)}
              </span>
              {r.profit != null && <span className="muted small">Profit on priced slips: {r.profit.toFixed(0)}</span>}
            </div>
          ))}
        </div>
      )}
      {slips.length ? (
        <div className="table-wrap">
          <table className="table">
            <thead>
              <tr>
                <th>Slip</th>
                <th>Legs</th>
                <th>Chance</th>
                <th>Risk</th>
                <th>Odds</th>
                <th>Stake</th>
                <th>Result</th>
              </tr>
            </thead>
            <tbody>
              {slips.map((s) => {
                const won = String(s.won).toLowerCase() === "true";
                return (
                  <tr key={s.slip_id}>
                    <td>
                      {s.slip_id}
                      <br />
                      <span className="muted small">{s.saved_on.slice(0, 10)}</span>
                    </td>
                    <td>{s.labels}</td>
                    <td>{pct(Number(s.prob))}</td>
                    <td>
                      <span className={riskClass(s.risk)}>{s.risk}</span>
                    </td>
                    <td>{s.price ? Number(s.price).toFixed(2) : "–"}</td>
                    <td>{Number(s.stake).toFixed(0)}</td>
                    <td>{s.result ? <b className={won ? "t-ok" : "t-away"}>{won ? "Won" : "Lost"} ({s.result})</b> : <span className="muted">waiting</span>}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      ) : (
        <p className="muted">No saved slips yet. Save one and it settles itself once the results are in.</p>
      )}
    </div>
  );
}

type MatchEntry = Board["matches"][number];

// Pick one match, or all of them. The choice lives in the address (?match=id), so a refresh keeps it.
// A themed list instead of the browser's <select>, whose open list Windows draws in its own colours.
// Keyboard: Enter/Space/arrows open it, arrows move, Enter picks, Escape closes.
function MatchPicker({ matches, value, onChange }: { matches: MatchEntry[]; value: string; onChange: (id: string) => void }) {
  const items = useMemo(() => [{ id: "", day: "", m: null as MatchEntry | null }, ...matches.map((m) => ({ id: m.match.id, day: m.match.day_ist, m }))], [matches]);
  const [open, setOpen] = useState(false);
  const [active, setActive] = useState(0);
  const box = useRef<HTMLDivElement>(null);
  const list = useRef<HTMLUListElement>(null);
  const current = Math.max(0, items.findIndex((x) => x.id === value));
  const golden = (m: MatchEntry) => m.singles.some((s) => s.title === "Golden single");

  useEffect(() => {
    if (!open) return;
    const away = (e: MouseEvent) => {
      if (box.current && !box.current.contains(e.target as Node)) setOpen(false);
    };
    document.addEventListener("mousedown", away);
    return () => document.removeEventListener("mousedown", away);
  }, [open]);
  useEffect(() => {
    if (open) list.current?.querySelector(`[data-i="${active}"]`)?.scrollIntoView({ block: "nearest" });
  }, [open, active]);

  const show = () => {
    setActive(current);
    setOpen(true);
  };
  const choose = (i: number) => {
    onChange(items[i].id);
    setOpen(false);
  };
  const onKey = (e: React.KeyboardEvent) => {
    if (!open) {
      if (["Enter", " ", "ArrowDown", "ArrowUp"].includes(e.key)) {
        e.preventDefault();
        show();
      }
      return;
    }
    if (e.key === "ArrowDown") setActive((a) => Math.min(items.length - 1, a + 1));
    else if (e.key === "ArrowUp") setActive((a) => Math.max(0, a - 1));
    else if (e.key === "Home") setActive(0);
    else if (e.key === "End") setActive(items.length - 1);
    else if (e.key === "Enter" || e.key === " ") choose(active);
    else if (e.key === "Escape" || e.key === "Tab") {
      setOpen(false);
      return;
    } else return;
    e.preventDefault();
  };

  const label = (x: (typeof items)[number]) =>
    x.m ? (
      <>
        <span className="mp-time">{x.m.match.time_ist}</span>
        <span className="mp-teams">
          {x.m.match.home} v {x.m.match.away}
        </span>
        <span className="mp-league">{x.m.match.league_name}</span>
        {golden(x.m) && (
          <span className="t-gold" aria-label="has a golden call">
            ★
          </span>
        )}
      </>
    ) : (
      <span className="mp-teams">All matches ({matches.length})</span>
    );

  return (
    <div className="match-picker" ref={box}>
      <span className="mp-label" id="desk-match-label">
        Show
      </span>
      <div className="mp">
        <button
          type="button"
          className={`mp-button${open ? " open" : ""}`}
          aria-haspopup="listbox"
          aria-expanded={open}
          aria-labelledby="desk-match-label desk-match-value"
          aria-activedescendant={open ? `mp-opt-${active}` : undefined}
          aria-controls="desk-match-list"
          onClick={() => (open ? setOpen(false) : show())}
          onKeyDown={onKey}
        >
          <span className="mp-value" id="desk-match-value">
            {label(items[current])}
          </span>
          <span className="mp-caret" aria-hidden="true">
            ▾
          </span>
        </button>
        {open && (
          <ul className="mp-list" role="listbox" id="desk-match-list" aria-labelledby="desk-match-label" ref={list}>
            {items.map((x, i) => (
              <Fragment key={x.id || "all"}>
                {x.day && x.day !== items[i - 1]?.day && (
                  <li className="mp-day" role="presentation">
                    {x.day}
                  </li>
                )}
                <li
                  id={`mp-opt-${i}`}
                  data-i={i}
                  role="option"
                  aria-selected={i === current}
                  className={`mp-opt${i === active ? " active" : ""}${i === current ? " chosen" : ""}`}
                  onMouseEnter={() => setActive(i)}
                  onMouseDown={(e) => e.preventDefault()}
                  onClick={() => choose(i)}
                >
                  {label(x)}
                  {i === current && (
                    <span className="mp-tick" aria-hidden="true">
                      ✓
                    </span>
                  )}
                </li>
              </Fragment>
            ))}
          </ul>
        )}
      </div>
    </div>
  );
}

export default function Desk() {
  const board = useApi<Board>(desk.board, "desk-board");
  const [tick, setTick] = useState(0);
  const [params, setParams] = useSearchParams();
  const all = board.data?.matches ?? [];
  const wanted = params.get("match") ?? "";
  const chosen = all.find((m) => m.match.id === wanted);
  const shown = chosen ? [chosen] : all;
  const pick = (id: string) => setParams(id ? { match: id } : {}, { replace: true });
  return (
    <Page>
      <PrivateTabs />
      <section className="hero desk-hero" aria-label="Trader desk">
        <div className="hero-nox">
          <Nox size={170} animate />
        </div>
        <div className="hero-copy stack-10">
          <h1 className="h1">Trader desk</h1>
          <p className="muted">
            Nox generates everything here from the model: the good singles in every match and two parlays per match. The safe
            parlay is the highest-paying mix that still comes in 60%+ of the time; the risky one pays more and comes in 25%+ of
            the time, built only from High-confidence legs. Every leg in both must pull its weight. Our model’s own view: apart from the golden bars, which are still being tracked, no edge
            over the market is proven, and a parlay pays the site’s margin on every leg.
          </p>
        </div>
      </section>
      {board.error ? (
        <Problem error={board.error} />
      ) : board.loading || !board.data ? (
        <Loading what="the board" />
      ) : (
        <>
          {all.length > 0 && (
            <div className="desk-bar stack-6">
              <MatchPicker matches={all} value={chosen ? wanted : ""} onChange={pick} />
              {wanted && !chosen && <span className="muted small">That match is no longer on the board, so here are all of them.</span>}
            </div>
          )}
          {!chosen && (
            <>
              <section aria-label="Top of the board" className="stack-14">
                <div className="section-head">
                  <h2>Top of the board</h2>
                  <span className="legend">Next 3 days · corners and odd/even left out (no skill) · correct scores only as singles</span>
                </div>
                <div className="cards slips">
                  {board.data.top.map((s) => (
                    <SlipCard key={s.title} slip={s} />
                  ))}
                </div>
              </section>
              {board.data.golden.length > 0 && (
                <section aria-label="Golden singles" className="stack-10">
                  <div className="section-head">
                    <h2>
                      <span className="t-gold">★</span> Golden singles
                    </h2>
                    <span className="legend">Every golden call in the next 3 days · unproven, still being tracked</span>
                  </div>
                  <div className="stack-8">
                    {board.data.golden.map((s) => (
                      <SingleRow key={s.legs[0].key} s={s} showMatch />
                    ))}
                  </div>
                </section>
              )}
            </>
          )}
          <section aria-label="Match by match" className="stack-14">
            {!chosen && <h2>Match by match</h2>}
            {shown.length ? (
              shown.map((m) => <MatchBoard key={m.match.id} m={m} focus={Boolean(chosen)} />)
            ) : (
              <p className="muted">No matches in the next 3 days.</p>
            )}
            {chosen && (
              <button type="button" className="link-btn" onClick={() => pick("")}>
                Show all matches
              </button>
            )}
          </section>
        </>
      )}
      <section aria-label="Saved slips" className="stack-14">
        <div className="section-head">
          <h2>Saved slips</h2>
          <button type="button" className="link-btn" onClick={() => setTick((t) => t + 1)}>
            Refresh results
          </button>
        </div>
        <Saved tick={tick} />
      </section>
    </Page>
  );
}
