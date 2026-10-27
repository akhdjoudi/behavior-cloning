import type { Color, Key } from "@lichess-org/chessground/types";
import { motion } from "motion/react";

import type { MoveProb } from "../api";
import { metaOf } from "../players";
import { arrowsFor, Board } from "./Board";
import { displayOrder, Monogram } from "./Players";

type Props = {
  players: string[];
  preds: Map<number, MoveProb[]>;
  fen: string;
  orientation: Color;
  turnColor: Color;
  lastMove?: [Key, Key];
};

/** Coups préférés regroupés : quels joueurs sont d'accord, lesquels divergent. */
export function Consensus({ players, preds }: { players: string[]; preds: Map<number, MoveProb[]> }) {
  const groups = new Map<string, number[]>();
  for (const id of displayOrder(players)) {
    const best = preds.get(id)?.[0];
    if (id !== 0 && best) groups.set(best.san, [...(groups.get(best.san) ?? []), id]);
  }
  const sorted = [...groups.entries()].sort((a, b) => b[1].length - a[1].length);
  const total = sorted.reduce((n, [, ids]) => n + ids.length, 0);
  if (!sorted.length) return <div className="h-10 animate-pulse bg-ink/5" />;
  return (
    <ul>
      {sorted.map(([san, ids]) => (
        <motion.li layout key={san} className="flex items-center gap-4 border-t border-rule py-3 first:border-t-0">
          <span className="w-16 font-serif text-3xl leading-none">{san}</span>
          <span className="flex flex-wrap gap-1">
            {ids.map((id) => (
              <Monogram key={id} name={players[id]} size={28} active />
            ))}
          </span>
          <span className="ml-auto font-mono text-[11px] text-muted">
            {ids.length}/{total}
          </span>
        </motion.li>
      ))}
    </ul>
  );
}

export function Duel({ players, preds, fen, orientation, turnColor, lastMove }: Props) {
  return (
    <div className="grid grid-cols-2 gap-x-5 gap-y-7 sm:grid-cols-3">
      {displayOrder(players).map((id) => {
        const m = metaOf(players[id]);
        const moves = preds.get(id);
        const best = moves?.[0];
        return (
          <figure key={id}>
            <div className="mini-frame" style={{ boxShadow: `4px 4px 0 ${m.accent}` }}>
              <Board
                fen={fen}
                orientation={orientation}
                turnColor={turnColor}
                lastMove={lastMove}
                viewOnly
                coordinates={false}
                accent={m.accent}
                shapes={arrowsFor(moves, 3, false)}
              />
            </div>
            <figcaption className="mt-3 flex items-baseline justify-between gap-2">
              <span className="truncate font-serif text-xl leading-none">{m.last}</span>
              {best && (
                <span className="font-serif text-xl italic leading-none" style={{ color: m.accent }}>
                  {best.san}
                </span>
              )}
            </figcaption>
            <div className="mt-1.5 flex justify-between font-mono text-[10px] tabular-nums text-muted">
              {moves?.slice(0, 3).map((mv) => (
                <span key={mv.uci}>
                  {mv.san} {Math.round(mv.prob * 100)}%
                </span>
              ))}
            </div>
          </figure>
        );
      })}
    </div>
  );
}
