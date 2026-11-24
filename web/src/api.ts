// Types and fetchers for the Python API (src/football_edge/web). Every number on the site
// comes from these responses; the React side only formats them.
import { useEffect, useState } from "react";

export type Mood = "sure" | "unsure" | "shock";

export interface Golden {
  kind: "win" | "ou" | "cs";
  title: string;
  call: string;
  p: number;
  rule: string;
}

export interface MatchSummary {
  id: string;
  league: string;
  league_name: string;
  kickoff_utc: string;
  day_ist: string;
  time_ist: string;
  home: string;
  away: string;
  probs: { home: number; draw: number; away: number };
  pick: { side: "home" | "draw" | "away"; team: string; text: string; p: number; confidence: string; why: string };
  mood: Mood;
  line: string;
  exp_goals: { home: number; away: number };
  top_score: { score: string; p: number };
  over25: number;
  btts: number;
  golden: Golden[];
  big: number;
}

export interface MatchDetail extends MatchSummary {
  goals_call: { text: string; p: number; confidence: string; why: string };
  top_scores: { score: string; p: number }[];
  grid: number[][];
  corners: { expected: number | null; over_9_5: number | null };
  cards: { expected: number | null; over_3_5: number | null; over_4_5: number | null; red: number | null };
  form: Record<"home" | "away", { results: string[]; scored: number; conceded: number }>;
  meetings: { date: string; home: string; away: string; score: string }[];
}

export interface Meta {
  trained_up_to: string;
  matches: number;
  days: number;
  warning: string | null;
}

export interface GoldenPage {
  calls: MatchSummary[];
  bars: { kind: string; title: string; text: string }[];
  tracker: { title: string; logged: number; settled: number; came_true: number }[];
}

export interface AskStatus {
  ready: boolean;
  model: string;
  problem: string | null;
}

export type AskEvent =
  | { type: "tool"; name: string; args: Record<string, unknown> }
  | { type: "answer"; answer: string; tools: string[]; checked: boolean; unverified: string[]; refused: boolean; seconds: number }
  | { type: "error"; message: string; busy?: boolean };

/** Ask Nox a question. The API streams one JSON object per line; each is passed to onEvent. */
export async function askNox(session: string, question: string, onEvent: (e: AskEvent) => void): Promise<void> {
  const r = await fetch("/api/ask", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session, question }),
  });
  if (!r.ok || !r.body) {
    onEvent({ type: "error", message: `The server said ${r.status}. Is football-edge serve running?` });
    return;
  }
  const reader = r.body.getReader();
  const dec = new TextDecoder();
  let buf = "";
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buf += dec.decode(value, { stream: true });
    let i;
    while ((i = buf.indexOf("\n")) >= 0) {
      const line = buf.slice(0, i).trim();
      buf = buf.slice(i + 1);
      if (line) onEvent(JSON.parse(line) as AskEvent);
    }
  }
}

// -- the private trader desk --------------------------------------------------------------
export interface DeskLeg {
  key: string;
  match_id: string;
  home: string;
  away: string;
  kickoff_utc: string;
  market: string;
  sel: string;
  line: number | null;
  label: string;
  p: number;
  confidence: string;
  skill: boolean;
  weak: boolean;
}

export interface Slip {
  title?: string;
  why?: string;
  kind: string;
  legs: DeskLeg[];
  p: number;
  fair_odds: number | null;
  risk: "Low" | "Medium" | "High" | "Very high";
  weak_legs: string[];
  leg_costs: { key: string; label: string; with: number; without: number }[];
  stake: number;
  currency: string;
  stake_note: string;
  price: { odds: number; back_per_100: number; verdict: string; note: string } | null;
  desk_view: string;
}

export interface Review {
  text: string;
  checked: boolean;
  unverified: string[];
  seconds: number;
  desk_view: string;
}

export interface SavedSlip {
  saved_on: string;
  slip_id: string;
  kind: string;
  labels: string;
  prob: number;
  risk: string;
  price: number | null;
  stake: number;
  result: string | null;
  won: boolean | string | null;
  profit: number | null;
}

export interface DeskHistory {
  slips: SavedSlip[];
  record: { risk: string; settled: number; won: number; expected: number; profit: number | null }[];
}

async function post<T>(path: string, body: unknown): Promise<T> {
  const r = await fetch(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  if (!r.ok) {
    let detail = `${r.status}`;
    try {
      detail = (await r.json()).detail ?? detail;
    } catch {
      /* not JSON */
    }
    throw new Error(detail);
  }
  return (await r.json()) as T;
}

export interface Board {
  top: Slip[];
  golden: Slip[];
  matches: { match: MatchSummary; singles: Slip[]; parlays: { safe: Slip | null; risky: Slip | null } }[];
}

export const desk = {
  board: () => get<Board>("/api/desk/board"),
  check: (keys: string[], price?: number | null) => post<Slip>("/api/desk/check", { keys, price: price ?? null }),
  review: (keys: string[], price?: number | null) => post<Review>("/api/desk/review", { keys, price: price ?? null }),
  save: (keys: string[], price?: number | null, stake?: number | null) =>
    post<SavedSlip>("/api/desk/save", { keys, price: price ?? null, stake: stake ?? null }),
  history: () => get<DeskHistory>("/api/desk/history"),
};

// -- live tracking: scorecard and Instagram posts (private) ---------------------------------
export interface ScoreRow {
  family: string;
  name: string;
  n: number;
  matches: number;
  avg_p: number | null;
  hit: number | null;
  brier: number | null;
  base_brier: number | null;
  skill: number | null;
  market_n: number;
  market_brier: number | null;
  model_brier_same: number | null;
  verdict: string;
}

export interface GradedCall {
  call: string;
  p: number;
  won: boolean;
}

export interface RecentMatch {
  id: string;
  kickoff_utc: string;
  league: string;
  home: string;
  away: string;
  score: string;
  result?: GradedCall;
  goals?: GradedCall;
}

export interface Scorecard {
  since: string | null;
  logged_matches: number;
  graded_matches: number;
  upcoming_matches: number;
  awaiting_results: number;
  next_kickoff: string | null;
  recent: RecentMatch[];
  min_sample: number;
  overall: ScoreRow | null;
  markets: ScoreRow[];
  calibration: { band: string; lo: number; hi: number; n: number; avg_p: number | null; hit: number | null }[];
  confidence: { label: string; n: number; avg_p: number | null; hit: number | null; gap: number | null }[];
  slips: { title: string; n: number; expected: number | null; hit: number | null; won: number; floor: number | null }[];
  posted: { n: number; settled: number; won: number };
  log_ok: boolean;
  log_problems: { taken_at: string; file: string; what: string }[];
}

export interface PostCall {
  kind: "result" | "goals";
  call: string;
  p: number;
  confidence: string;
  posted: boolean;
}

export interface PostCandidate {
  match: MatchSummary;
  calls: PostCall[];
  big: boolean;
  caption: string;
}

export interface PostsPage {
  posts: PostCandidate[];
  rule: { market: string; text: string }[];
}

export interface Recap {
  from: string | null;
  to: string | null;
  calls: { match: string; call: string; p: number; score: string | null; won: boolean | null; day: string }[];
  settled: number;
  won: number;
  pending: number;
}

export const tracking = {
  scorecard: () => get<Scorecard>("/api/desk/scorecard"),
  posts: () => get<PostsPage>("/api/posts"),
  mark: (match_id: string, kind: string) => post<{ already: boolean }>("/api/posts/mark", { match_id, kind }),
  recap: (days = 7) => get<Recap>(`/api/posts/recap?days=${days}`),
};

export const pct = (p: number) => `${Math.round(p * 100)}%`;

async function get<T>(path: string): Promise<T> {
  const r = await fetch(path);
  if (!r.ok) {
    let detail = `${r.status}`;
    try {
      detail = (await r.json()).detail ?? detail;
    } catch {
      /* not JSON */
    }
    throw new Error(detail);
  }
  return (await r.json()) as T;
}

export const api = {
  meta: () => get<Meta>("/api/meta"),
  next: (n = 5) => get<MatchSummary[]>(`/api/next?n=${n}`),
  matches: () => get<MatchSummary[]>("/api/matches"),
  match: (id: string) => get<MatchDetail>(`/api/matches/${encodeURIComponent(id)}`),
  golden: () => get<GoldenPage>("/api/golden"),
  askStatus: () => get<AskStatus>("/api/ask/status"),
};

export type Loadable<T> = { data?: T; error?: string; loading: boolean };

/** Fetch once per `key`; re-fetches when the key changes (e.g. a new match id). */
export function useApi<T>(fn: () => Promise<T>, key: string): Loadable<T> {
  const [state, setState] = useState<Loadable<T>>({ loading: true });
  useEffect(() => {
    let live = true;
    setState({ loading: true });
    fn()
      .then((data) => live && setState({ data, loading: false }))
      .catch((e: Error) => live && setState({ error: e.message, loading: false }));
    return () => {
      live = false;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key]);
  return state;
}
