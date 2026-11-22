// Nox the owl, drawn in SVG. One drawing, three moods:
//   sure   – one wing waves (a confident call)
//   unsure – one eyebrow up, a wing scratches his head, a "?" (a close call)
//   shock  – wide eyes, open beak, both wings up, "!!" (an upset)
import type { Mood } from "../api";

const BODY = "#5566C2";
const DARK = "#4253A9";
const MASK = "#8E9CE8";
const BELLY = "#FFF1DC";
const AMBER = "#FFB72B";
const BEAK = "#FF8A1F";
const BLUSH = "#FF9EB5";
const PUPIL = "#1A1A2E";

const EYE = {
  sure: { iris: 27, pupil: 17, dx: 0, dy: 2 },
  unsure: { iris: 27, pupil: 16, dx: -9, dy: -9 },
  shock: { iris: 31, pupil: 9, dx: 0, dy: 0 },
};

function Eye({ cx, mood }: { cx: number; mood: Mood }) {
  const e = EYE[mood];
  const x = cx + e.dx;
  const y = 136 + e.dy;
  return (
    <g className="nox-eye">
      <circle cx={cx} cy={136} r={43} fill="#FFFFFF" />
      <circle cx={x} cy={y} r={e.iris} fill={AMBER} />
      <circle cx={x} cy={y} r={e.pupil} fill={PUPIL} />
      <circle cx={x - 9} cy={y - 10} r={8} fill="#FFFFFF" />
      <circle cx={x + 8} cy={y + 8} r={3.5} fill="#FFFFFF" />
    </g>
  );
}

function Wing({ cx, cy, rot }: { cx: number; cy: number; rot: number }) {
  return <ellipse cx={cx} cy={cy} rx={19} ry={34} transform={`rotate(${rot} ${cx} ${cy})`} fill={DARK} />;
}

function Ball() {
  const pts = [0, 1, 2, 3, 4].map((k) => {
    const a = ((-90 + 72 * k) * Math.PI) / 180;
    return [Math.cos(a), Math.sin(a)];
  });
  return (
    <g transform="translate(250 262)">
      <circle r={24} fill="#FFFFFF" stroke={PUPIL} strokeWidth={4} />
      <polygon points={pts.map(([x, y]) => `${(9 * x).toFixed(1)},${(9 * y).toFixed(1)}`).join(" ")} fill={PUPIL} />
      <g stroke={PUPIL} strokeWidth={3}>
        {pts.map(([x, y], i) => (
          <line key={i} x1={9 * x} y1={9 * y} x2={22 * x} y2={22 * y} />
        ))}
      </g>
    </g>
  );
}

export default function Nox({
  mood = "sure",
  size = 300,
  ball = true,
  animate = false,
  label,
}: {
  mood?: Mood;
  size?: number;
  ball?: boolean;
  animate?: boolean;
  label?: string;
}) {
  const raised =
    mood === "sure" ? (
      <g className={animate ? "nox-wave" : undefined}>
        <Wing cx={262} cy={150} rot={-40} />
      </g>
    ) : mood === "unsure" ? (
      <Wing cx={236} cy={176} rot={-160} />
    ) : (
      <Wing cx={252} cy={150} rot={-40} />
    );
  return (
    <svg
      viewBox="0 0 300 300"
      width={size}
      height={size}
      role="img"
      aria-label={label ?? `Nox the owl, ${mood === "sure" ? "confident" : mood === "unsure" ? "unsure" : "surprised"}`}
      className={animate ? "nox nox-float" : "nox"}
      style={{ display: "block", flex: "none" }}
    >
      <ellipse cx={80} cy={66} rx={20} ry={34} transform="rotate(-28 80 66)" fill={BODY} />
      <ellipse cx={220} cy={66} rx={20} ry={34} transform="rotate(28 220 66)" fill={BODY} />
      {mood === "shock" ? <Wing cx={48} cy={150} rot={40} /> : <Wing cx={42} cy={214} rot={18} />}
      <ellipse cx={150} cy={166} rx={114} ry={116} fill={BODY} />
      <ellipse cx={150} cy={240} rx={60} ry={42} fill={BELLY} />
      {[
        [136, 228],
        [164, 228],
        [150, 246],
      ].map(([x, y]) => (
        <path key={`${x}${y}`} d={`M${x - 9} ${y} q9 9 18 0`} fill="none" stroke="#EBD6B8" strokeWidth={3.5} strokeLinecap="round" />
      ))}
      <circle cx={108} cy={136} r={56} fill={MASK} />
      <circle cx={192} cy={136} r={56} fill={MASK} />
      <Eye cx={108} mood={mood} />
      <Eye cx={192} mood={mood} />
      <ellipse cx={74} cy={188} rx={16} ry={9} fill={BLUSH} opacity={0.8} />
      <ellipse cx={226} cy={188} rx={16} ry={9} fill={BLUSH} opacity={0.8} />
      {mood === "shock" ? (
        <path d="M141 172 L159 172 L150 182 Z M143 188 L157 188 L150 197 Z" fill={BEAK} stroke={BEAK} strokeWidth={4} strokeLinejoin="round" />
      ) : (
        <path d="M141 172 L159 172 L150 185 Z" fill={BEAK} stroke={BEAK} strokeWidth={4} strokeLinejoin="round" />
      )}
      {raised}
      {mood === "unsure" && (
        <>
          <path d="M80 80 Q100 68 124 78 M176 90 L220 90" stroke={PUPIL} strokeWidth={6} strokeLinecap="round" fill="none" />
          <text x={262} y={66} fontFamily="Fredoka, sans-serif" fontWeight={700} fontSize={44} fill="#FFC94A">
            ?
          </text>
        </>
      )}
      {mood === "shock" && (
        <text x={250} y={70} fontFamily="Fredoka, sans-serif" fontWeight={700} fontSize={46} fill="#FFC94A">
          !!
        </text>
      )}
      <g fill={BEAK}>
        <ellipse cx={114} cy={282} rx={10} ry={7} />
        <ellipse cx={130} cy={285} rx={10} ry={7} />
        <ellipse cx={170} cy={285} rx={10} ry={7} />
        <ellipse cx={186} cy={282} rx={10} ry={7} />
      </g>
      {ball && <Ball />}
    </svg>
  );
}
