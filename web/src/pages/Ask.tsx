import { useEffect, useRef, useState } from "react";
import { useSearchParams } from "react-router-dom";
import Nox from "../components/Nox";
import { Page } from "../components/ui";
import { api, askNox, useApi, type AskEvent, type AskStatus, type MatchSummary, type Mood } from "../api";

interface Msg {
  role: "you" | "nox";
  text: string;
  tools?: string[];
  checked?: boolean;
  unverified?: string[];
  refused?: boolean;
  seconds?: number;
  error?: boolean;
}

const TOOL_NAMES: Record<string, string> = {
  predict_match: "match prediction",
  upcoming_matches: "fixtures",
  team_form: "team form",
  head_to_head: "past meetings",
};

function progress(name: string, a: Record<string, unknown>): string {
  const s = (k: string) => String(a[k] ?? "").trim();
  switch (name) {
    case "predict_match":
      return `Looking at ${s("home")} v ${s("away")}…`;
    case "upcoming_matches":
      return "Checking the fixtures…";
    case "team_form":
      return `Reading ${s("team")}’s last results…`;
    case "head_to_head":
      return `Digging up ${s("team_a")} v ${s("team_b")} meetings…`;
    default:
      return "Checking the model…";
  }
}

// Browser storage can be missing or blocked; the chat still works, it just won't survive a reload.
const store = {
  get(key: string): string | null {
    try {
      return sessionStorage.getItem(key);
    } catch {
      return null;
    }
  },
  set(key: string, value: string) {
    try {
      sessionStorage.setItem(key, value);
    } catch {
      /* ignore */
    }
  },
};
const newSession = () => `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 10)}`;

function suggestions(next: MatchSummary[] | undefined): string[] {
  const out: string[] = [];
  const [a, b] = next ?? [];
  if (a) out.push(`Who wins ${a.home} v ${a.away}?`);
  if (b) out.push(`How many goals in ${b.home} v ${b.away}?`);
  if (a) out.push(`How has ${a.away} been playing lately?`);
  out.push("What matches are on in the next two days?");
  return out;
}

function Reply({ m }: { m: Msg }) {
  return (
    <div className="msg msg-nox">
      <Nox mood={m.error ? "shock" : m.refused ? "unsure" : "sure"} size={44} ball={false} />
      <div className="stack-6 msg-body">
        <div className={`bubble bubble-chat tail-left${m.error ? " bubble-error" : ""}`}>
          {m.text.split("\n").map((line, i) => (
            <p key={i}>{line}</p>
          ))}
        </div>
        {!m.error && (
          <span className="msg-meta">
            {m.checked ? (
              <span className="ok">✓ Every number checked against the model</span>
            ) : (
              <span className="warn">⚠ Couldn’t check: {m.unverified?.join(", ")}</span>
            )}
            {m.tools?.length ? ` · Looked at: ${[...new Set(m.tools.map((t) => TOOL_NAMES[t] ?? t))].join(", ")}` : ""}
            {m.seconds != null ? ` · ${Math.round(m.seconds)}s` : ""}
          </span>
        )}
      </div>
    </div>
  );
}

export default function Ask() {
  const status = useApi<AskStatus>(api.askStatus, "ask-status");
  const next = useApi<MatchSummary[]>(() => api.next(2), "next2");
  const [params, setParams] = useSearchParams();
  const [session, setSession] = useState(() => store.get("nox-session") ?? newSession());
  const [msgs, setMsgs] = useState<Msg[]>(() => {
    try {
      return JSON.parse(store.get("nox-chat") ?? "[]") as Msg[];
    } catch {
      return [];
    }
  });
  const [text, setText] = useState("");
  const [working, setWorking] = useState<string | null>(null);
  const [elapsed, setElapsed] = useState(0);
  const end = useRef<HTMLDivElement>(null);
  const asked = useRef(false);

  useEffect(() => store.set("nox-session", session), [session]);
  useEffect(() => {
    store.set("nox-chat", JSON.stringify(msgs.slice(-40)));
    end.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [msgs, working]);
  useEffect(() => {
    if (working === null) return;
    const t0 = Date.now();
    setElapsed(0);
    const t = setInterval(() => setElapsed(Math.round((Date.now() - t0) / 1000)), 1000);
    return () => clearInterval(t);
  }, [working === null]); // eslint-disable-line react-hooks/exhaustive-deps

  async function send(q: string) {
    const question = q.trim();
    if (!question || working !== null) return;
    setText("");
    setMsgs((m) => [...m, { role: "you", text: question }]);
    setWorking("Nox is thinking…");
    try {
      await askNox(session, question, (e: AskEvent) => {
        if (e.type === "tool") setWorking(progress(e.name, e.args));
        else if (e.type === "answer")
          setMsgs((m) => [...m, { role: "nox", text: e.answer, tools: e.tools, checked: e.checked, unverified: e.unverified, refused: e.refused, seconds: e.seconds }]);
        else setMsgs((m) => [...m, { role: "nox", text: e.message, error: true }]);
      });
    } catch (err) {
      setMsgs((m) => [...m, { role: "nox", text: `I couldn’t reach the server (${(err as Error).message}). Is football-edge serve running?`, error: true }]);
    } finally {
      setWorking(null);
    }
  }

  // A question passed in the address (from a match page) is asked once, straight away.
  useEffect(() => {
    const q = params.get("q");
    if (q && !asked.current) {
      asked.current = true;
      setParams({}, { replace: true });
      void send(q);
    }
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  const last = msgs[msgs.length - 1];
  const mood: Mood = working !== null ? "unsure" : last?.error ? "shock" : "sure";
  const down = status.data && !status.data.ready;
  return (
    <Page>
      <section aria-label="Ask Nox" className="ask">
        <aside className="ask-side">
          <Nox mood={mood} size={230} animate={working !== null || !msgs.length} />
          <div className="stack-6 ask-side-text">
            <h1 className="h1">Ask Nox</h1>
            <p className="muted">
              Ask about any Premier League, Serie A, La Liga, Bundesliga or Ligue 1 match: who wins, goals, cards, form or past
              meetings. Nox checks the model before he answers.
            </p>
            {status.data?.model && (
              <p className="muted small">
                Runs on {status.data.model}, a small AI model on this computer. Answers take up to a minute.
              </p>
            )}
          </div>
        </aside>
        <div className="chat">
          {down && <p className="warning">{status.data?.problem}</p>}
          <div className="thread" aria-live="polite">
            {!msgs.length && (
              <div className="msg msg-nox">
                <Nox size={44} ball={false} />
                <div className="bubble bubble-chat tail-left">
                  <p>Hoo! I’ve read every match coming up. What do you want to know?</p>
                </div>
              </div>
            )}
            {msgs.map((m, i) =>
              m.role === "you" ? (
                <div key={i} className="msg msg-you">
                  <div className="you-bubble">{m.text}</div>
                </div>
              ) : (
                <Reply key={i} m={m} />
              ),
            )}
            {working !== null && (
              <div className="msg msg-nox" role="status">
                <Nox mood="unsure" size={44} ball={false} />
                <div className="bubble bubble-chat tail-left working">
                  <span className="dots" aria-hidden="true">
                    <i />
                    <i />
                    <i />
                  </span>
                  {working} <span className="muted-dark">{elapsed}s</span>
                </div>
              </div>
            )}
            <div ref={end} />
          </div>
          {!msgs.length && (
            <div className="row-8" aria-label="Try asking">
              {suggestions(next.data).map((q) => (
                <button key={q} type="button" className="chip" onClick={() => void send(q)} disabled={working !== null}>
                  {q}
                </button>
              ))}
            </div>
          )}
          <form
            className="composer"
            onSubmit={(e) => {
              e.preventDefault();
              void send(text);
            }}
          >
            <label htmlFor="ask-input" className="sr-only">
              Your question for Nox
            </label>
            <textarea
              id="ask-input"
              rows={2}
              maxLength={300}
              value={text}
              placeholder="e.g. Will both teams score in Liverpool v Manchester United?"
              onChange={(e) => setText(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  void send(text);
                }
              }}
            />
            <div className="composer-row">
              {msgs.length > 0 && (
                <button
                  type="button"
                  className="link-btn"
                  disabled={working !== null}
                  onClick={() => {
                    setMsgs([]);
                    setSession(newSession());
                  }}
                >
                  New chat
                </button>
              )}
              <button type="submit" className="btn btn-gold" disabled={!text.trim() || working !== null}>
                {working !== null ? "Nox is answering…" : "Ask"}
              </button>
            </div>
          </form>
          <p className="muted small">Nox only makes predictions. They’re chances, not certainties.</p>
        </div>
      </section>
    </Page>
  );
}
