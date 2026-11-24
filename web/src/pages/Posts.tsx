// Instagram posts: strong calls only (a fixed rule from the track record), each as a
// 1080x1350 image plus a caption, and a weekly recap of every posted call, misses included.
// Predictions only: no betting words, prices or club badges anywhere on these images.
import { useRef, useState, type ReactNode } from "react";
import { toPng } from "html-to-image";
import Nox from "../components/Nox";
import { Loading, Page, PrivateTabs, Problem } from "../components/ui";
import { pct, tracking, useApi, type PostCandidate, type PostsPage, type Recap } from "../api";

const W = 1080;
const H = 1350;
const PREVIEW = 0.32;

async function download(node: HTMLElement | null, name: string) {
  if (!node) return;
  await document.fonts.ready;
  const url = await toPng(node, { width: W, height: H, pixelRatio: 1, cacheBust: true });
  const a = document.createElement("a");
  a.href = url;
  a.download = name;
  a.click();
}

/** The image is drawn at full Instagram size and shown scaled down. */
function Frame({ children, frameRef }: { children: ReactNode; frameRef: React.RefObject<HTMLDivElement> }) {
  return (
    <div className="post-preview" style={{ width: W * PREVIEW, height: H * PREVIEW }}>
      <div style={{ transform: `scale(${PREVIEW})`, transformOrigin: "top left" }}>
        <div ref={frameRef} className="ig">
          {children}
        </div>
      </div>
    </div>
  );
}

function Brand() {
  return (
    <div className="ig-brand">
      <Nox size={86} ball={false} />
      <div>
        <div className="ig-name">nox</div>
        <div className="ig-tag">football predictions, all night</div>
      </div>
    </div>
  );
}

function PostImage({ c }: { c: PostCandidate }) {
  const m = c.match;
  const main = c.calls[0];
  const segs = [
    { k: "home", label: m.home, p: m.probs.home },
    { k: "draw", label: "Draw", p: m.probs.draw },
    { k: "away", label: m.away, p: m.probs.away },
  ];
  return (
    <>
      <Brand />
      <div className="ig-kicker">
        {m.league_name.toUpperCase()} · {m.day_ist.toUpperCase()} · {m.time_ist} IST
      </div>
      <div className="ig-title">
        {m.home}
        <span className="ig-v">v</span>
        {m.away}
      </div>
      <div className="ig-call">
        <Nox mood="sure" size={300} ball={false} />
        <div className="ig-bubble">
          <div className="ig-bubble-head">Nox’s call</div>
          <div className="ig-bubble-call">{main.call}</div>
          <div className="ig-bubble-p">{pct(main.p)}</div>
          {c.calls[1] && (
            <div className="ig-bubble-also">
              Also: {c.calls[1].call} · {pct(c.calls[1].p)}
            </div>
          )}
        </div>
      </div>
      <div className="ig-bar">
        {segs.map((s) => (
          <div key={s.k} className={`ig-seg ig-${s.k}`} style={{ flexGrow: Math.max(s.p, 0.06) }}>
            <span>{pct(s.p)}</span>
          </div>
        ))}
      </div>
      <div className="ig-bar-labels">
        {segs.map((s) => (
          <span key={s.k}>{s.label}</span>
        ))}
      </div>
      <div className="ig-facts">
        <div>
          <span>Most likely score</span>
          <b>{m.top_score.score}</b>
        </div>
        <div>
          <span>Expected goals</span>
          <b>
            {m.exp_goals.home.toFixed(1)} – {m.exp_goals.away.toFixed(1)}
          </b>
        </div>
        <div>
          <span>Over 2.5 goals</span>
          <b>{pct(m.over25)}</b>
        </div>
      </div>
      <div className="ig-foot">Every call is logged before kickoff and graded after the match, hits and misses.</div>
    </>
  );
}

function RecapImage({ r }: { r: Recap }) {
  // Every posted call is shown (misses included): rows get compact past 6 so 10 still fit.
  const shown = r.calls.slice(0, 10);
  const compact = r.calls.length > 6;
  return (
    <>
      <Brand />
      <div className="ig-kicker">
        WEEKLY RECAP · {r.from?.toUpperCase()} – {r.to?.toUpperCase()}
      </div>
      <div className="ig-recap-big">
        {r.won} <span>of</span> {r.settled}
      </div>
      <div className="ig-recap-sub">posted calls came in</div>
      <div className={`ig-recap-list${compact ? " compact" : ""}`}>
        {shown.map((c, i) => (
          <div key={i} className={`ig-recap-row ${c.won == null ? "wait" : c.won ? "hit" : "miss"}`}>
            <span className="ig-recap-mark">{c.won == null ? "…" : c.won ? "✓" : "✗"}</span>
            <span className="ig-recap-what">
              <b>{c.call}</b>
              <small>
                {c.match} · {pct(c.p)}
              </small>
            </span>
            <span className="ig-recap-score">{c.score ?? "–"}</span>
          </div>
        ))}
        {r.calls.length > shown.length && <div className="ig-recap-more">+ {r.calls.length - shown.length} more</div>}
      </div>
      <div className="ig-foot">Every call we posted this week, hits and misses. Nothing deleted.</div>
    </>
  );
}

function PostCard({ c, onPosted }: { c: PostCandidate; onPosted: () => void }) {
  const ref = useRef<HTMLDivElement>(null);
  const [note, setNote] = useState("");
  const [arming, setArming] = useState("");
  const [busy, setBusy] = useState(false);
  const m = c.match;
  async function mark(kind: string) {
    if (arming !== kind) {
      setArming(kind);
      return;
    }
    setBusy(true);
    try {
      await tracking.mark(m.id, kind);
      setNote("Logged as posted. It will be in the weekly recap, hit or miss.");
      onPosted();
    } catch (e) {
      setNote((e as Error).message);
    } finally {
      setBusy(false);
      setArming("");
    }
  }
  return (
    <article className="card post-card">
      <Frame frameRef={ref}>
        <PostImage c={c} />
      </Frame>
      <div className="stack-10 post-side">
        <span className="kicker small">
          {m.day_ist.toUpperCase()} · {m.time_ist} IST · {m.league_name.toUpperCase()}
          {c.big && <span className="chip chip-static chip-mini">big club</span>}
        </span>
        <h3 className="mb-title">
          {m.home} v {m.away}
        </h3>
        {c.calls.map((x) => (
          <div key={x.kind} className="row-8 align-center wrap-row">
            <b>
              {x.call} · {pct(x.p)}
            </b>
            {x.posted ? (
              <span className="chip chip-static t-ok">✓ Posted</span>
            ) : (
              <button type="button" className={`btn btn-sm ${arming === x.kind ? "btn-gold" : "btn-outline"}`} disabled={busy} onClick={() => void mark(x.kind)}>
                {arming === x.kind ? "Confirm: posted (can’t be undone)" : "Mark as posted"}
              </button>
            )}
          </div>
        ))}
        <textarea className="caption" readOnly value={c.caption} rows={10} aria-label="Caption" />
        <div className="row-8 wrap-row">
          <button type="button" className="btn btn-gold btn-sm" onClick={() => void download(ref.current, `nox_${m.id}.png`)}>
            Download image
          </button>
          <button
            type="button"
            className="btn btn-outline btn-sm"
            onClick={() => void navigator.clipboard.writeText(c.caption).then(() => setNote("Caption copied."))}
          >
            Copy caption
          </button>
        </div>
        {note && <span className="small muted">{note}</span>}
      </div>
    </article>
  );
}

function RecapCard() {
  const ref = useRef<HTMLDivElement>(null);
  const r = useApi<Recap>(() => tracking.recap(7), "recap");
  if (r.error) return <Problem error={r.error} />;
  if (!r.data) return <Loading what="the recap" />;
  if (!r.data.calls.length)
    return <p className="muted">No posted calls in the last 7 days yet. Mark calls as posted and they show up here once played.</p>;
  return (
    <article className="card post-card">
      <Frame frameRef={ref}>
        <RecapImage r={r.data} />
      </Frame>
      <div className="stack-10 post-side">
        <b>
          {r.data.won} of {r.data.settled} came in
          {r.data.pending > 0 && <span className="muted small"> · {r.data.pending} still to play</span>}
        </b>
        <span className="muted small">Post it once every call of the week has been played, so the count is final.</span>
        <button type="button" className="btn btn-gold btn-sm" onClick={() => void download(ref.current, `nox_recap_${r.data?.to}.png`)}>
          Download recap image
        </button>
      </div>
    </article>
  );
}

export default function Posts() {
  const [tick, setTick] = useState(0);
  const page = useApi<PostsPage>(tracking.posts, `posts-${tick}`);
  return (
    <Page>
      <PrivateTabs />
      <section className="stack-10" aria-label="Posts">
        <h1 className="h1">Posts</h1>
        <p className="muted desk-intro">
          Strong calls only, chosen by a fixed rule from the track record, never by feel. Pick the ones to post, download the
          image, copy the caption, then mark it as posted: from then on it counts in the weekly recap, hit or miss.
        </p>
        {page.data && (
          <div className="card stack-6 rule-card">
            <span className="label">What counts as a strong call</span>
            {page.data.rule.map((r) => (
              <span key={r.market} className="small">
                {r.text}
              </span>
            ))}
          </div>
        )}
      </section>
      {page.error ? (
        <Problem error={page.error} />
      ) : page.loading || !page.data ? (
        <Loading what="the strong calls" />
      ) : page.data.posts.length ? (
        <section className="stack-18" aria-label="Strong calls">
          {page.data.posts.map((c) => (
            <PostCard key={c.match.id} c={c} onPosted={() => setTick((t) => t + 1)} />
          ))}
        </section>
      ) : (
        <p className="muted">No strong calls in the next few days. That happens: the rule does not bend to fill a week.</p>
      )}
      <section className="stack-14" aria-label="Weekly recap">
        <h2>Weekly recap</h2>
        <RecapCard key={tick} />
      </section>
    </Page>
  );
}
