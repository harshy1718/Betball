import { pct, type MatchSummary } from "./api";

/** "Liverpool 53%" or "Draw 30%". */
export const pickText = (m: MatchSummary) => `${m.pick.team} ${pct(m.pick.p)}`;

/** "in the past, ..." -> "In the past, ... ." */
export const sentence = (s: string) => (s ? s.charAt(0).toUpperCase() + s.slice(1) + "." : "");

/** Kick-off within the next 24 hours. */
export const soon = (iso: string) => new Date(iso).getTime() - Date.now() < 24 * 3600 * 1000;

/** Kick-off time has passed (the API keeps matches for 2 hours after kick-off). */
export const started = (m: MatchSummary) => new Date(m.kickoff_utc).getTime() <= Date.now();

/** A handicap leg as a three-way handicap table writes it ("2:0" = home starts 2 goals up),
 *  so "Getafe +1.5" can be found on a site's table: "2:0 · Getafe". */
export function handicapTable(l: { market: string; sel: string; home: string; away: string }): string | null {
  const m = /^(home|away)([+-])(\d+(?:\.5)?)$/.exec(l.sel);
  if (l.market !== "hcp" || !m) return null;
  const [, side, sign, k] = m;
  const goals = sign === "+" ? Number(k) + 0.5 : Number(k) - 0.5; // +1.5 -> side starts 2 up; -1.5 -> other side 1 up
  const headStartHome = (side === "home") === (sign === "+");
  const score = headStartHome ? `${goals}:0` : `0:${goals}`;
  return `${score} · ${side === "home" ? l.home : l.away}`;
}
