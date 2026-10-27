import { AnimatePresence, motion } from "motion/react";

import type { MoveProb } from "../api";

type Props = {
  moves: MoveProb[] | undefined;
  accent: string;
  onPick?: (uci: string) => void;
  count?: number;
};

const OPACITY = [1, 0.75, 0.6, 0.5, 0.42];

export function MoveBars({ moves, accent, onPick, count = 5 }: Props) {
  if (!moves) return <div className="h-[212px] animate-pulse rounded-xl bg-white/[0.03]" />;
  if (!moves.length) return <p className="py-6 text-center text-sm text-slate-400">Partie terminée.</p>;
  const top = moves.slice(0, count);
  return (
    <ul className="space-y-1.5">
      <AnimatePresence initial={false} mode="popLayout">
        {top.map((m, i) => (
          <motion.li
            key={m.uci}
            layout
            initial={{ opacity: 0, y: 6 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0 }}
            transition={{ type: "spring", stiffness: 500, damping: 40 }}
          >
            <button
              onClick={() => onPick?.(m.uci)}
              disabled={!onPick}
              className="group relative flex w-full items-center gap-3 overflow-hidden rounded-xl bg-white/[0.03] px-3 py-2.5 text-left enabled:hover:bg-white/[0.07]"
            >
              <motion.div
                className="absolute inset-y-0 left-0 rounded-xl"
                style={{ background: accent, opacity: 0.18 * OPACITY[i] + 0.04 }}
                initial={false}
                animate={{ width: `${Math.max(2, m.prob * 100)}%` }}
                transition={{ type: "spring", stiffness: 160, damping: 26 }}
              />
              <span className="relative w-4 text-xs tabular-nums text-slate-500">{i + 1}</span>
              <span className="relative font-display text-[17px] font-semibold text-white">{m.san}</span>
              <span className="relative ml-auto text-sm font-medium tabular-nums" style={{ color: accent, opacity: OPACITY[i] }}>
                {(m.prob * 100).toFixed(m.prob < 0.1 ? 1 : 0)} %
              </span>
            </button>
          </motion.li>
        ))}
      </AnimatePresence>
    </ul>
  );
}
