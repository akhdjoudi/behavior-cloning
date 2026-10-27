import type { DrawShape } from "@lichess-org/chessground/draw";
import type { Color, Key } from "@lichess-org/chessground/types";
import { Chess, type Square } from "chess.js";
import { AnimatePresence, motion } from "motion/react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { getInfo, predict, sampleMove, type Info, type MoveProb } from "./api";
import { arrowsFor, Board } from "./components/Board";
import { ConsensusChips, Duel } from "./components/Duel";
import { MoveBars } from "./components/MoveBars";
import { PlayerList } from "./components/PlayerList";
import { metaOf } from "./players";

type Mode = "analyse" | "jouer" | "duel";

const MODES: { id: Mode; label: string; hint: string }[] = [
  { id: "analyse", label: "Analyse", hint: "Joue les deux camps et regarde ce que chaque légende jouerait." },
  { id: "jouer", label: "Jouer", hint: "Affronte un style : le bot répond comme le joueur choisi." },
  { id: "duel", label: "Duel", hint: "La même position vue par tous les joueurs, côte à côte." },
];

const BOT_DELAY_MS = 450;

export default function App() {
  const [info, setInfo] = useState<Info | null>(null);
  const [apiError, setApiError] = useState(false);
  const game = useRef(new Chess());
  const [fen, setFen] = useState(game.current.fen());
  const [player, setPlayer] = useState(1);
  const [mode, setMode] = useState<Mode>("analyse");
  const [orientation, setOrientation] = useState<Color>("white");
  const [userColor, setUserColor] = useState<Color>("white");
  const [vary, setVary] = useState(false);
  const [showArrows, setShowArrows] = useState(true);
  const [preds, setPreds] = useState<{ fen: string; byPlayer: Map<number, MoveProb[]> }>({ fen: "", byPlayer: new Map() });

  useEffect(() => {
    getInfo()
      .then(setInfo)
      .catch(() => setApiError(true));
  }, []);

  const chess = game.current;
  const players = info?.players ?? [];
  const meta = metaOf(players[player] ?? "Tal");
  const turn: Color = chess.turn() === "w" ? "white" : "black";
  const gameOver = chess.isGameOver();

  const dests = useMemo(() => {
    const d = new Map<Key, Key[]>();
    for (const m of chess.moves({ verbose: true })) d.set(m.from, [...(d.get(m.from) ?? []), m.to]);
    return d;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [fen]);

  const history = useMemo(() => chess.history({ verbose: true }), [fen]); // eslint-disable-line react-hooks/exhaustive-deps
  const last = history[history.length - 1];
  const lastMove: [Key, Key] | undefined = last ? [last.from, last.to] : undefined;

  // Prédictions de la position courante : le joueur choisi, ou tout le monde en mode duel
  useEffect(() => {
    if (!info) return;
    const wanted = mode === "duel" ? [...info.players.keys()] : [player];
    let cancelled = false;
    predict(fen, wanted)
      .then((byPlayer) => !cancelled && setPreds({ fen, byPlayer }))
      .catch(() => setApiError(true));
    return () => {
      cancelled = true;
    };
  }, [fen, player, mode, info]);

  const current = preds.fen === fen ? preds.byPlayer : undefined;

  const play = useCallback((from: string, to: string, promotion?: string) => {
    const g = game.current;
    const candidates = g.moves({ verbose: true }).filter((m) => m.from === from && m.to === to);
    if (!candidates.length) return;
    const move = candidates.find((m) => m.promotion === (promotion ?? "q")) ?? candidates[0];
    g.move({ from: move.from as Square, to: move.to as Square, promotion: move.promotion });
    setFen(g.fen());
  }, []);

  const playUci = useCallback((uci: string) => play(uci.slice(0, 2), uci.slice(2, 4), uci[4]), [play]);

  // Le bot joue quand c'est son tour, avec le style du joueur choisi
  const botTurn = mode === "jouer" && turn !== userColor && !gameOver;
  useEffect(() => {
    const moves = current?.get(player);
    if (!botTurn || !moves?.length) return;
    const t = setTimeout(() => playUci(sampleMove(moves, vary ? 0.8 : 0).uci), BOT_DELAY_MS);
    return () => clearTimeout(t);
  }, [botTurn, current, player, vary, playUci]);

  const reset = () => {
    chess.reset();
    setFen(chess.fen());
  };
  const undo = () => {
    chess.undo();
    // En mode jeu, on revient au dernier coup du joueur humain
    if (mode === "jouer" && (chess.turn() === "w" ? "white" : "black") !== userColor) chess.undo();
    setFen(chess.fen());
  };
  const chooseColor = (c: Color) => {
    setUserColor(c);
    setOrientation(c);
    reset();
  };
  const switchMode = (m: Mode) => {
    setMode(m);
    if (m === "jouer") {
      setShowArrows(false);
      setOrientation(userColor);
    } else setShowArrows(true);
  };

  const shapes: DrawShape[] = useMemo(() => {
    if (!current || !showArrows) return [];
    if (mode !== "duel") return arrowsFor(current.get(player));
    // Duel : la flèche préférée de chaque joueur, à sa couleur
    return [...current.entries()]
      .filter(([id, moves]) => id !== 0 && moves.length)
      .map(([id, moves]) => ({
        orig: moves[0].uci.slice(0, 2) as Key,
        dest: moves[0].uci.slice(2, 4) as Key,
        brush: `pl-${players[id]}`,
        modifiers: { lineWidth: 9 },
      }));
  }, [current, showArrows, mode, player, players]);

  const status = chess.isCheckmate()
    ? `Échec et mat · ${turn === "white" ? "les Noirs" : "les Blancs"} gagnent`
    : chess.isStalemate()
      ? "Pat · partie nulle"
      : chess.isDraw()
        ? "Partie nulle"
        : botTurn
          ? `${meta.short} réfléchit…`
          : `Trait aux ${turn === "white" ? "Blancs" : "Noirs"}${chess.inCheck() ? " · échec" : ""}`;

  const movable: Color | "both" | undefined = gameOver ? undefined : mode === "jouer" ? (botTurn ? undefined : userColor) : "both";

  return (
    <div className="relative min-h-screen overflow-x-hidden bg-[#0b0d12] text-slate-200">
      <div
        className="pointer-events-none fixed inset-0 transition-[background] duration-700"
        style={{ background: `radial-gradient(900px 500px at 50% -10%, ${meta.accent}22, transparent 70%)` }}
      />

      <header className="relative mx-auto flex max-w-[1440px] flex-wrap items-center justify-between gap-4 px-4 py-5 lg:px-8">
        <div className="flex items-center gap-3">
          <div className="grid size-10 place-items-center rounded-xl bg-white/[0.06] text-2xl ring-1 ring-white/10">♞</div>
          <div>
            <h1 className="font-display text-xl font-semibold tracking-tight text-white">Chess Style Cloning</h1>
            <p className="text-xs text-slate-400">Joue contre le style des légendes des échecs</p>
          </div>
        </div>

        <div className="flex items-center gap-1 rounded-full border border-white/10 bg-white/[0.04] p-1">
          {MODES.map((m) => (
            <button
              key={m.id}
              onClick={() => switchMode(m.id)}
              className={`relative rounded-full px-4 py-1.5 text-sm font-medium transition-colors ${mode === m.id ? "text-[#0b0d12]" : "text-slate-300 hover:text-white"}`}
            >
              {mode === m.id && (
                <motion.span
                  layoutId="mode-pill"
                  className="absolute inset-0 rounded-full bg-white"
                  transition={{ type: "spring", stiffness: 500, damping: 38 }}
                />
              )}
              <span className="relative">{m.label}</span>
            </button>
          ))}
        </div>

        {info && (
          <div
            className={`rounded-full border px-3 py-1 text-xs ${info.trained ? "border-white/10 text-slate-400" : "border-amber-400/30 bg-amber-400/10 text-amber-200"}`}
          >
            {info.trained ? `Modèle · ${info.model}` : "Poids aléatoires · modèle non entraîné"}
          </div>
        )}
      </header>

      {apiError && (
        <div className="relative mx-auto mb-4 max-w-[1440px] px-4 lg:px-8">
          <div className="rounded-xl border border-red-400/30 bg-red-400/10 px-4 py-3 text-sm text-red-200">
            API injoignable. Lance <code className="rounded bg-black/30 px-1.5 py-0.5">python -m chessstyle.server</code> puis recharge
            la page.
          </div>
        </div>
      )}

      <main
        className={`relative mx-auto grid max-w-[1440px] grid-cols-[minmax(0,1fr)] gap-6 px-4 pb-12 lg:px-8 ${
          mode === "duel" ? "lg:grid-cols-[minmax(0,1fr)_minmax(0,1.15fr)]" : "lg:grid-cols-[280px_minmax(0,1fr)_340px]"
        }`}
      >
        {mode !== "duel" && (
          <aside className="order-2 min-w-0 lg:order-1">
            <PlayerList players={players} selected={player} onSelect={setPlayer} />
          </aside>
        )}

        <section className="order-1 min-w-0 lg:order-2">
          <div className="mx-auto w-full max-w-[min(680px,calc(100vh-150px))]">
            <div className="mb-3 flex items-center justify-between gap-3">
              <AnimatePresence mode="wait">
                <motion.p
                  key={status}
                  initial={{ opacity: 0, y: -4 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: 4 }}
                  className="text-sm text-slate-300"
                >
                  {status}
                </motion.p>
              </AnimatePresence>
              <p className="hidden text-xs text-slate-500 sm:block">{MODES.find((m) => m.id === mode)?.hint}</p>
            </div>
            <div className="board-frame rounded-[14px] p-[6px]" style={{ boxShadow: `0 30px 80px -40px ${meta.accent}aa` }}>
              <Board
                fen={fen}
                orientation={orientation}
                turnColor={turn}
                dests={dests}
                movableColor={movable}
                lastMove={lastMove}
                check={chess.inCheck()}
                shapes={shapes}
                onMove={(from, to) => play(from, to)}
                accent={meta.accent}
              />
            </div>
            <div className="mt-4 flex flex-wrap items-center gap-2">
              <Control onClick={undo} disabled={!history.length}>
                ↶ Annuler
              </Control>
              <Control onClick={reset}>⟲ Nouvelle partie</Control>
              <Control onClick={() => setOrientation(orientation === "white" ? "black" : "white")}>⇅ Retourner</Control>
              <Toggle on={showArrows} onClick={() => setShowArrows(!showArrows)}>
                Flèches
              </Toggle>
            </div>
          </div>
        </section>

        <aside className="order-3 min-w-0 space-y-5">
          {mode === "duel" ? (
            <>
              <Panel title="Qui joue quoi ?">
                {current ? <ConsensusChips players={players} preds={current} /> : <div className="h-8 animate-pulse rounded-full bg-white/[0.03]" />}
              </Panel>
              <Duel players={players} preds={current ?? new Map()} fen={fen} orientation={orientation} turnColor={turn} lastMove={lastMove} />
            </>
          ) : (
            <>
              {mode === "jouer" && (
                <Panel title="Ta partie">
                  <div className="flex gap-2">
                    <Toggle on={userColor === "white"} onClick={() => chooseColor("white")}>
                      Blancs
                    </Toggle>
                    <Toggle on={userColor === "black"} onClick={() => chooseColor("black")}>
                      Noirs
                    </Toggle>
                    <Toggle on={vary} onClick={() => setVary(!vary)} title="Le bot tire ses coups selon les probabilités au lieu de jouer toujours le plus probable">
                      Varier les coups
                    </Toggle>
                  </div>
                </Panel>
              )}
              <Panel title={mode === "jouer" ? `${meta.short} à ta place` : `Ce que jouerait ${meta.short}`} accent={meta.accent}>
                <MoveBars
                  moves={current?.get(player)}
                  accent={meta.accent}
                  onPick={mode === "analyse" && !gameOver ? playUci : undefined}
                />
              </Panel>
              <Panel title="Coups joués">
                <MoveList sans={history.map((m) => m.san)} />
              </Panel>
            </>
          )}
        </aside>
      </main>
    </div>
  );
}

function Panel({ title, accent, children }: { title: string; accent?: string; children: React.ReactNode }) {
  return (
    <section className="rounded-2xl border border-white/[0.07] bg-white/[0.025] p-4 backdrop-blur">
      <h2 className="mb-3 flex items-center gap-2 text-xs font-medium uppercase tracking-[0.14em] text-slate-400">
        {accent && <span className="size-1.5 rounded-full" style={{ background: accent }} />}
        {title}
      </h2>
      {children}
    </section>
  );
}

function Control(props: React.ButtonHTMLAttributes<HTMLButtonElement>) {
  return (
    <button
      {...props}
      className="rounded-full border border-white/10 bg-white/[0.04] px-3.5 py-1.5 text-sm text-slate-200 transition-colors hover:bg-white/[0.09] disabled:opacity-40"
    />
  );
}

function Toggle({ on, ...props }: { on: boolean } & React.ButtonHTMLAttributes<HTMLButtonElement>) {
  return (
    <button
      {...props}
      className={`rounded-full border px-3.5 py-1.5 text-sm transition-colors ${
        on ? "border-white/80 bg-white text-[#0b0d12]" : "border-white/10 bg-white/[0.04] text-slate-300 hover:bg-white/[0.09]"
      }`}
    />
  );
}

function MoveList({ sans }: { sans: string[] }) {
  const end = useRef<HTMLDivElement>(null);
  useEffect(() => end.current?.scrollIntoView({ block: "nearest" }), [sans.length]);
  if (!sans.length) return <p className="text-sm text-slate-500">Aucun coup pour l'instant.</p>;
  const rows = [];
  for (let i = 0; i < sans.length; i += 2) rows.push([sans[i], sans[i + 1]]);
  return (
    <div className="max-h-56 overflow-y-auto pr-1">
      <div className="grid grid-cols-[2.2rem_1fr_1fr] gap-x-2 gap-y-1 text-sm">
        {rows.map(([w, b], i) => (
          <div key={i} className="contents">
            <span className="tabular-nums text-slate-500">{i + 1}.</span>
            <span className="text-slate-200">{w}</span>
            <span className="text-slate-200">{b ?? ""}</span>
          </div>
        ))}
      </div>
      <div ref={end} />
    </div>
  );
}
