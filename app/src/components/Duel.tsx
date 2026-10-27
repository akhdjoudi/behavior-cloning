import type { Color, Key } from "@lichess-org/chessground/types";
import { motion } from "motion/react";

import type { MoveProb } from "../api";
import { metaOf } from "../players";
import { arrowsFor, Board } from "./Board";
import { Avatar } from "./PlayerList";

type Props = {
  players: string[];
  preds: Map<number, MoveProb[]>;
  fen: string;
  orientation: Color;
  turnColor: Color;
  lastMove?: [Key, Key];
};

/** Coups préférés regroupés : quels joueurs sont d'accord, lesquels divergent. */
function consensus(players: string[], preds: Map<number, MoveProb[]>) {
  const groups = new Map<string, number[]>();
  for (const [id, moves] of preds) {
    if (id === 0 || !moves.length) continue;
    const san = moves[0].san;
    groups.set(san, [...(groups.get(san) ?? []), id]);
  }
  return [...groups.entries()].sort((a, b) => b[1].length - a[1].length).map(([san, ids]) => ({ san, names: ids.map((i) => players[i]) }));
}

export function ConsensusChips({ players, preds }: { players: string[]; preds: Map<number, MoveProb[]> }) {
  const groups = consensus(players, preds);
  if (!groups.length) return null;
  return (
    <div className="flex flex-wrap gap-2">
      {groups.map((g) => (
        <motion.div layout key={g.san} className="flex items-center gap-2 rounded-full border border-white/10 bg-white/[0.04] py-1 pl-1 pr-3">
          <div className="flex -space-x-2">
            {g.names.map((n) => (
              <Avatar key={n} name={n} size={24} />
            ))}
          </div>
          <span className="font-display text-sm font-semibold text-white">{g.san}</span>
        </motion.div>
      ))}
    </div>
  );
}

export function Duel({ players, preds, fen, orientation, turnColor, lastMove }: Props) {
  const ids = [...players.keys()].filter((i) => i !== 0).concat([0]);
  return (
    <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-2 xl:grid-cols-3">
      {ids.map((id) => {
        const m = metaOf(players[id]);
        const moves = preds.get(id);
        const best = moves?.[0];
        return (
          <div key={id} className="rounded-2xl border border-white/5 bg-white/[0.03] p-2.5">
            <div className="mb-2 flex items-center gap-2">
              <Avatar name={players[id]} size={26} />
              <span className="truncate text-[13px] font-medium text-white">{m.short}</span>
              {best && (
                <span className="ml-auto font-display text-sm font-semibold" style={{ color: m.accent }}>
                  {best.san}
                </span>
              )}
            </div>
            <div className="pointer-events-none">
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
            <div className="mt-2 flex justify-between text-[11px] tabular-nums text-slate-400">
              {moves?.slice(0, 3).map((mv) => (
                <span key={mv.uci}>
                  {mv.san} <span className="text-slate-500">{Math.round(mv.prob * 100)}%</span>
                </span>
              ))}
            </div>
          </div>
        );
      })}
    </div>
  );
}
