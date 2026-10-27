import type { Key } from "@lichess-org/chessground/types";
import { Chess } from "chess.js";
import { AnimatePresence, motion } from "motion/react";
import { useEffect, useState } from "react";

import { predict, type MoveProb } from "../api";
import { fmt, metaOf } from "../players";
import { Board } from "./Board";
import { MoveBars } from "./MoveBars";

const START = new Chess().fen();
const AFTER_E4 = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1";
const AFTER_D4 = "rnbqkbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR b KQkq - 0 1";
const TYPICAL_PLIES = 8;

type Repertoire = { white?: MoveProb[]; vsE4?: MoveProb[]; vsD4?: MoveProb[] };
type Line = { sans: string[]; fen: string; last?: [Key, Key] };

/** Répertoire selon le modèle : premier coup avec les Blancs, réponses à 1.e4 et à 1.d4. */
function useRepertoire(player: number, ready: boolean): Repertoire {
  const [rep, setRep] = useState<Repertoire>({});
  useEffect(() => {
    if (!ready) return;
    let cancelled = false;
    setRep({});
    Promise.all([predict(START, [player]), predict(AFTER_E4, [player]), predict(AFTER_D4, [player])]).then(([a, b, c]) => {
      if (!cancelled) setRep({ white: a.get(player), vsE4: b.get(player), vsD4: c.get(player) });
    });
    return () => {
      cancelled = true;
    };
  }, [player, ready]);
  return rep;
}

/** Partie type : le modèle joue les deux camps avec le style du joueur, en prenant toujours son coup préféré. */
function useTypicalLine(player: number, ready: boolean): Line | undefined {
  const [line, setLine] = useState<Line>();
  useEffect(() => {
    if (!ready) return;
    let cancelled = false;
    setLine(undefined);
    (async () => {
      const game = new Chess();
      for (let i = 0; i < TYPICAL_PLIES && !game.isGameOver(); i++) {
        const best = (await predict(game.fen(), [player])).get(player)?.[0];
        if (cancelled || !best) return;
        game.move({ from: best.uci.slice(0, 2), to: best.uci.slice(2, 4), promotion: best.uci[4] });
      }
      const h = game.history({ verbose: true });
      const last = h[h.length - 1];
      if (!cancelled) setLine({ sans: h.map((m) => m.san), fen: game.fen(), last: last ? [last.from, last.to] : undefined });
    })().catch(() => {});
    return () => {
      cancelled = true;
    };
  }, [player, ready]);
  return line;
}

function formatLine(sans: string[]) {
  return sans.map((s, i) => (i % 2 === 0 ? `${i / 2 + 1}. ${s}` : s)).join(" ");
}

type Props = {
  name: string;
  player: number;
  index: number;
  ready: boolean;
  onPlay: () => void;
  onAnalyse: () => void;
};

export function Profile({ name, player, index, ready, onPlay, onAnalyse }: Props) {
  const m = metaOf(name);
  const rep = useRepertoire(player, ready);
  const line = useTypicalLine(player, ready);

  return (
    <AnimatePresence mode="wait">
      <motion.article
        key={player}
        initial={{ opacity: 0, y: 14 }}
        animate={{ opacity: 1, y: 0 }}
        exit={{ opacity: 0, y: -8 }}
        transition={{ duration: 0.35, ease: [0.22, 1, 0.36, 1] }}
        className="relative"
      >
        {/* Initiales géantes en filigrane */}
        <div
          aria-hidden
          className="pointer-events-none absolute -top-6 right-0 select-none font-serif text-[9rem] leading-none sm:text-[15rem] lg:-top-12 lg:text-[19rem]"
          style={{ color: m.accent, opacity: 0.1 }}
        >
          {m.initials}
        </div>

        <p className="label">
          Nº {String(index).padStart(2, "0")} · {m.born} · {m.years}
        </p>
        <h2 className="mt-3 font-serif text-[64px] leading-[0.88] tracking-[-0.01em] sm:text-[96px] lg:text-[112px]">
          {m.first}
          <br />
          <span style={{ color: m.accent }}>{m.last}</span>
        </h2>
        <p className="mt-4 font-serif text-2xl text-muted">« {m.nickname} »</p>
        <p className="mt-6 max-w-2xl text-[17px] leading-relaxed">{m.style}</p>

        <ul className="mt-6 flex flex-wrap gap-x-5 gap-y-2">
          {m.traits.map((t) => (
            <li key={t} className="flex items-center gap-2 font-mono text-[11px] uppercase tracking-[0.14em]">
              <span className="size-1.5" style={{ background: m.accent }} />
              {t}
            </li>
          ))}
        </ul>

        <dl className="mt-10 grid grid-cols-2 border-y border-ink sm:grid-cols-4">
          {[
            ["Parties", fmt(m.games)],
            ["Positions apprises", fmt(m.positions)],
            ["Titre", m.title],
            ["Période", m.years],
          ].map(([k, v], i) => (
            <div key={k} className={`px-0 py-4 sm:px-4 ${i % 2 ? "pl-4" : ""} ${i > 0 ? "sm:border-l sm:border-rule" : "sm:pl-0"} ${i >= 2 ? "border-t border-rule sm:border-t-0" : ""}`}>
              <dt className="label">{k}</dt>
              <dd className="mt-1.5 font-serif text-[22px] leading-tight">{v}</dd>
            </div>
          ))}
        </dl>

        <section className="mt-12">
          <div className="flex items-baseline justify-between gap-4 border-b border-ink pb-2">
            <h3 className="font-serif text-3xl">Répertoire</h3>
            <span className="label">selon le modèle</span>
          </div>
          <div className="grid gap-8 pt-5 md:grid-cols-3">
            {[
              ["Avec les Blancs", rep.white],
              ["Contre 1. e4", rep.vsE4],
              ["Contre 1. d4", rep.vsD4],
            ].map(([title, moves]) => (
              <div key={title as string}>
                <p className="label mb-2">{title as string}</p>
                <MoveBars moves={moves as MoveProb[] | undefined} accent={m.accent} count={4} size="sm" />
              </div>
            ))}
          </div>
        </section>

        <section className="mt-12 grid items-start gap-8 md:grid-cols-[260px_1fr]">
          <div className="mini-frame" style={{ boxShadow: `5px 5px 0 ${m.accent}` }}>
            <Board fen={line?.fen ?? START} orientation="white" turnColor="white" lastMove={line?.last} viewOnly coordinates={false} accent={m.accent} />
          </div>
          <div>
            <div className="flex items-baseline justify-between gap-4 border-b border-ink pb-2">
              <h3 className="font-serif text-3xl">Partie type</h3>
              <span className="label">{TYPICAL_PLIES / 2} premiers coups</span>
            </div>
            <p className="mt-4 min-h-[2.5em] font-serif text-[28px] leading-snug">
              {line ? formatLine(line.sans) : <span className="text-muted">Le modèle réfléchit…</span>}
            </p>
            <p className="mt-3 max-w-lg text-sm leading-relaxed text-muted">
              Le modèle joue les deux camps comme {m.short}, en choisissant à chaque fois son coup le plus probable.
            </p>
            <div className="mt-8 flex flex-wrap gap-3">
              <button
                onClick={onPlay}
                className="group flex items-center gap-3 border border-ink bg-ink px-5 py-3 font-mono text-xs uppercase tracking-[0.16em] text-paper transition-colors hover:bg-[var(--accent)] hover:border-[var(--accent)]"
              >
                Jouer contre {m.short}
                <span className="transition-transform group-hover:translate-x-1">→</span>
              </button>
              <button
                onClick={onAnalyse}
                className="border border-ink px-5 py-3 font-mono text-xs uppercase tracking-[0.16em] transition-colors hover:bg-ink hover:text-paper"
              >
                Analyser
              </button>
            </div>
          </div>
        </section>

        <section className="mt-12 border-t border-rule pt-5">
          <div className="flex flex-wrap items-baseline justify-between gap-2">
            <h3 className="font-serif text-2xl">Signature de style</h3>
            <span className="label">bientôt · benchmark Stockfish</span>
          </div>
          <p className="mt-2 max-w-xl text-sm text-muted">
            Agressivité, goût du sacrifice, prudence, longueur des parties : le radar de style arrivera avec les métriques de
            style_bench.py.
          </p>
        </section>
      </motion.article>
    </AnimatePresence>
  );
}
