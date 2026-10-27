import type { Color } from "@lichess-org/chessground/types";
import { motion } from "motion/react";

import type { Evaluation } from "../api";

/** Chances de gain des Blancs (0 à 100), avec la même courbe que Lichess. */
export function whitePercent(e: Evaluation): number {
  if (e.mate === 0) return e.winner === "white" ? 100 : 0;
  if (e.mate != null) return e.mate > 0 ? 100 : 0;
  const cp = e.cp ?? 0;
  return 50 + 50 * (2 / (1 + Math.exp(-0.00368208 * cp)) - 1);
}

export function evalLabel(e: Evaluation): string {
  if (e.mate === 0) return e.winner === "white" ? "1-0" : "0-1";
  if (e.mate != null) return `M${Math.abs(e.mate)}`;
  const pawns = (e.cp ?? 0) / 100;
  return `${pawns > 0 ? "+" : ""}${pawns.toFixed(1)}`;
}

/** Barre verticale : la part blanche monte avec l'avantage des Blancs, du côté des Blancs à l'écran. */
export function EvalBar({ evaluation, orientation }: { evaluation?: Evaluation; orientation: Color }) {
  const pct = evaluation ? whitePercent(evaluation) : 50;
  const whiteAhead = pct >= 50;
  const whiteAtBottom = orientation === "white";
  // Le chiffre s'affiche au bout du camp qui mène
  const labelAtBottom = whiteAhead === whiteAtBottom;
  return (
    <div className="relative w-3.5 shrink-0 self-stretch overflow-hidden border border-ink bg-ink sm:w-[22px]" title="Évaluation Stockfish">
      <motion.div
        className={`absolute inset-x-0 bg-paper ${whiteAtBottom ? "bottom-0" : "top-0"}`}
        initial={false}
        animate={{ height: `${pct}%` }}
        transition={{ type: "spring", stiffness: 90, damping: 20 }}
      />
      <div className="absolute inset-x-0 top-1/2 h-px bg-[var(--accent)] opacity-60" />
      {evaluation && (
        <span
          className={`absolute inset-x-0 hidden text-center font-mono text-[8.5px] font-semibold leading-none tracking-tight sm:block ${
            labelAtBottom ? "bottom-1.5" : "top-1.5"
          } ${whiteAhead ? "text-ink" : "text-paper"}`}
        >
          {evalLabel(evaluation)}
        </span>
      )}
    </div>
  );
}
