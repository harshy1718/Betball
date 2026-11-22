import { pct, type MatchSummary } from "./api";

/** "Liverpool 53%" or "Draw 30%". */
export const pickText = (m: MatchSummary) => `${m.pick.team} ${pct(m.pick.p)}`;

/** "in the past, ..." -> "In the past, ... ." */
export const sentence = (s: string) => (s ? s.charAt(0).toUpperCase() + s.slice(1) + "." : "");

/** Kick-off within the next 24 hours. */
export const soon = (iso: string) => new Date(iso).getTime() - Date.now() < 24 * 3600 * 1000;

/** Kick-off time has passed (the API keeps matches for 2 hours after kick-off). */
export const started = (m: MatchSummary) => new Date(m.kickoff_utc).getTime() <= Date.now();
