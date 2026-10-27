import { AnimatePresence, motion } from "motion/react";

import type { MoveProb } from "../api";

type Props = {
  moves: MoveProb[] | undefined;
  accent: string;
  onPick?: (uci: string) => void;
  count?: number;
  size?: "lg" | "sm";
};

/** Liste éditoriale des coups préférés : rang, coup, filet de probabilité, pourcentage. */
export function MoveBars({ moves, accent, onPick, count = 5, size = "lg" }: Props) {
  const rowH = size === "lg" ? "py-2.5" : "py-1.5";
  if (!moves)
    return (
      <div className="space-y-3">
        {Array.from({ length: count }, (_, i) => (
          <div key={i} className="h-6 animate-pulse bg-ink/5" />
        ))}
      </div>
    );
  if (!moves.length) return <p className="font-serif text-lg italic text-muted">Partie terminée.</p>;
  return (
    <ol>
      <AnimatePresence initial={false} mode="popLayout">
        {moves.slice(0, count).map((m, i) => (
          <motion.li
            key={m.uci}
            layout
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.25 }}
            className="border-t border-rule first:border-t-0"
          >
            <button
              onClick={() => onPick?.(m.uci)}
              disabled={!onPick}
              className={`group grid w-full grid-cols-[1.6rem_4.2rem_1fr_3.2rem] items-center gap-3 text-left ${rowH}`}
            >
              <span className="font-mono text-[10.5px] text-muted">{String(i + 1).padStart(2, "0")}</span>
              <span
                className={`font-serif leading-none transition-transform group-enabled:group-hover:translate-x-0.5 ${size === "lg" ? "text-[26px]" : "text-xl"}`}
                style={{ color: i === 0 ? accent : undefined }}
              >
                {m.san}
              </span>
              <span className="relative h-px bg-rule">
                <motion.span
                  className="absolute left-0 top-1/2 h-[3px] -translate-y-1/2"
                  style={{ background: accent, opacity: i === 0 ? 1 : 0.55 }}
                  initial={false}
                  animate={{ width: `${Math.max(1.5, m.prob * 100)}%` }}
                  transition={{ type: "spring", stiffness: 140, damping: 24 }}
                />
              </span>
              <span className="text-right font-mono text-xs tabular-nums">
                {(m.prob * 100).toFixed(m.prob < 0.1 ? 1 : 0)}%
              </span>
            </button>
          </motion.li>
        ))}
      </AnimatePresence>
    </ol>
  );
}
