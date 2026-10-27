import type { DrawShape } from "@lichess-org/chessground/draw";
import type { Color, Key } from "@lichess-org/chessground/types";
import { Chess, type Square } from "chess.js";
import { AnimatePresence, motion } from "motion/react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { getInfo, predict, sampleMove, type Info, type MoveProb } from "./api";
import { arrowsFor, Board } from "./components/Board";
import { Consensus, Duel } from "./components/Duel";
import { MoveBars } from "./components/MoveBars";
import { displayOrder, PlayerIndex, PlayerSwitcher } from "./components/Players";
import { Profile } from "./components/Profile";
import { metaOf } from "./players";

type View = "joueurs" | "jouer" | "analyse" | "duel";

const VIEWS: { id: View; label: string }[] = [
  { id: "joueurs", label: "Joueurs" },
  { id: "jouer", label: "Jouer" },
  { id: "analyse", label: "Analyse" },
  { id: "duel", label: "Duel" },
];

const BOT_DELAY_MS = 500;

export default function App() {
  const [info, setInfo] = useState<Info | null>(null);
  const [apiError, setApiError] = useState(false);
  const game = useRef(new Chess());
  const [fen, setFen] = useState(game.current.fen());
  const [player, setPlayer] = useState(1);
  const [view, setView] = useState<View>("joueurs");
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
  const onBoard = view !== "joueurs";

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
    if (!info || !onBoard) return;
    const wanted = view === "duel" ? [...info.players.keys()] : [player];
    let cancelled = false;
    predict(fen, wanted)
      .then((byPlayer) => !cancelled && setPreds({ fen, byPlayer }))
      .catch(() => setApiError(true));
    return () => {
      cancelled = true;
    };
  }, [fen, player, view, info, onBoard]);

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
  const botTurn = view === "jouer" && turn !== userColor && !gameOver;
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
    if (view === "jouer" && (chess.turn() === "w" ? "white" : "black") !== userColor) chess.undo();
    setFen(chess.fen());
  };
  const chooseColor = (c: Color) => {
    setUserColor(c);
    setOrientation(c);
    reset();
  };
  const goTo = (v: View) => {
    setView(v);
    if (v === "jouer") {
      setShowArrows(false);
      setOrientation(userColor);
    } else setShowArrows(true);
    window.scrollTo({ top: 0, behavior: "smooth" });
  };

  const shapes: DrawShape[] = useMemo(() => {
    if (!current || !showArrows) return [];
    if (view !== "duel") return arrowsFor(current.get(player));
    // Duel : la flèche préférée de chaque joueur, à sa couleur
    return [...current.entries()]
      .filter(([id, moves]) => id !== 0 && moves.length)
      .map(([id, moves]) => ({
        orig: moves[0].uci.slice(0, 2) as Key,
        dest: moves[0].uci.slice(2, 4) as Key,
        brush: `pl-${players[id]}`,
        modifiers: { lineWidth: 9 },
      }));
  }, [current, showArrows, view, player, players]);

  const status = chess.isCheckmate()
    ? `Échec et mat, ${turn === "white" ? "les Noirs" : "les Blancs"} gagnent`
    : chess.isStalemate()
      ? "Pat, partie nulle"
      : chess.isDraw()
        ? "Partie nulle"
        : botTurn
          ? `${meta.short} réfléchit…`
          : `Trait aux ${turn === "white" ? "Blancs" : "Noirs"}${chess.inCheck() ? ", échec" : ""}`;

  const movable: Color | "both" | undefined = gameOver ? undefined : view === "jouer" ? (botTurn ? undefined : userColor) : "both";
  const accent = view === "duel" ? "#17140f" : meta.accent;

  return (
    <div className="relative min-h-screen" style={{ ["--accent" as string]: accent }}>
      <header className="sticky top-0 z-40 border-b border-ink bg-paper/90 backdrop-blur-sm">
        <div className="mx-auto flex h-16 max-w-[1440px] items-center justify-between gap-4 px-5 lg:px-10">
          <button onClick={() => goTo("joueurs")} className="flex items-baseline gap-3 text-left">
            <span className="font-serif text-[26px] leading-none">
              Style <em>Cloning</em>
            </span>
            <span className="label hidden md:inline">Échecs · imitation de style</span>
          </button>
          <nav className="flex gap-4 sm:gap-8">
            {VIEWS.map((v) => (
              <button
                key={v.id}
                onClick={() => goTo(v.id)}
                className={`relative py-1 font-mono text-[11px] uppercase tracking-[0.16em] transition-colors ${view === v.id ? "text-ink" : "text-muted hover:text-ink"}`}
              >
                {v.label}
                {view === v.id && (
                  <motion.span layoutId="nav-underline" className="absolute -bottom-px left-0 right-0 h-[2px] bg-[var(--accent)]" />
                )}
              </button>
            ))}
          </nav>
          {info && <span className="label hidden lg:inline">{info.trained ? `Modèle · ${info.model}` : "Poids aléatoires"}</span>}
        </div>
      </header>

      {apiError && (
        <div className="border-b border-ink bg-[#cf3a1f] px-5 py-2.5 text-center font-mono text-xs text-paper">
          API injoignable · lance python -m chessstyle.server puis recharge la page
        </div>
      )}

      <main className="relative mx-auto max-w-[1440px] px-5 pb-20 lg:px-10">
        <AnimatePresence mode="wait">
          {view === "joueurs" ? (
            <motion.div
              key="joueurs"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="grid grid-cols-[minmax(0,1fr)] gap-10 pt-8 lg:grid-cols-[300px_minmax(0,1fr)] lg:gap-16 lg:pt-14"
            >
              <aside className="min-w-0 lg:sticky lg:top-28 lg:self-start">
                <p className="label mb-4 hidden lg:block">Index des joueurs</p>
                <PlayerIndex players={players} selected={player} onSelect={setPlayer} />
                <p className="mt-6 hidden max-w-[260px] text-sm leading-relaxed text-muted lg:block">
                  Un seul réseau de neurones, entraîné sur leurs parties, qui imite le style de chacun.
                </p>
              </aside>
              {info && (
                <Profile
                  name={players[player]}
                  player={player}
                  index={displayOrder(players).indexOf(player) + 1}
                  ready={!!info}
                  onPlay={() => {
                    reset();
                    goTo("jouer");
                  }}
                  onAnalyse={() => goTo("analyse")}
                />
              )}
            </motion.div>
          ) : (
            <motion.div
              key="board"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className={`grid grid-cols-[minmax(0,1fr)] gap-10 pt-8 lg:gap-14 lg:pt-12 ${
                view === "duel" ? "lg:grid-cols-[minmax(0,0.9fr)_minmax(0,1.1fr)]" : "lg:grid-cols-[minmax(0,1fr)_380px]"
              }`}
            >
              <section className="min-w-0">
                <div className="mx-auto w-full max-w-[min(720px,calc(100vh-190px))]">
                  <div className="mb-4 flex items-baseline justify-between gap-4">
                    <AnimatePresence mode="wait">
                      <motion.p
                        key={status}
                        initial={{ opacity: 0, y: -3 }}
                        animate={{ opacity: 1, y: 0 }}
                        exit={{ opacity: 0, y: 3 }}
                        className="font-serif text-2xl italic"
                      >
                        {status}
                      </motion.p>
                    </AnimatePresence>
                    <span className="label">Coup {Math.floor(history.length / 2) + 1}</span>
                  </div>
                  <div className="board-frame">
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
                  <div className="mt-7 flex flex-wrap items-center gap-x-6 gap-y-3">
                    <TextButton onClick={undo} disabled={!history.length}>
                      ← Annuler
                    </TextButton>
                    <TextButton onClick={reset}>Nouvelle partie</TextButton>
                    <TextButton onClick={() => setOrientation(orientation === "white" ? "black" : "white")}>Retourner</TextButton>
                    <Check on={showArrows} onClick={() => setShowArrows(!showArrows)}>
                      Flèches
                    </Check>
                  </div>
                </div>
              </section>

              <aside className="min-w-0 space-y-10">
                {view === "duel" ? (
                  <>
                    <Block title="Qui joue quoi ?" note="coup préféré de chaque légende">
                      {current ? <Consensus players={players} preds={current} /> : <div className="h-10 animate-pulse bg-ink/5" />}
                    </Block>
                    <Duel players={players} preds={current ?? new Map()} fen={fen} orientation={orientation} turnColor={turn} lastMove={lastMove} />
                  </>
                ) : (
                  <>
                    <PlayerSwitcher players={players} selected={player} onSelect={setPlayer} />
                    {view === "jouer" && (
                      <Block title="Ta partie">
                        <div className="flex flex-wrap gap-x-6 gap-y-3">
                          <Check on={userColor === "white"} onClick={() => chooseColor("white")}>
                            Blancs
                          </Check>
                          <Check on={userColor === "black"} onClick={() => chooseColor("black")}>
                            Noirs
                          </Check>
                          <Check on={vary} onClick={() => setVary(!vary)}>
                            Varier les coups
                          </Check>
                        </div>
                      </Block>
                    )}
                    <Block title={view === "jouer" ? `${meta.short} à ta place` : `Ce que jouerait ${meta.short}`} note="probabilité du modèle">
                      <MoveBars moves={current?.get(player)} accent={meta.accent} onPick={view === "analyse" && !gameOver ? playUci : undefined} />
                    </Block>
                    <Block title="Partie">
                      <MoveList sans={history.map((m) => m.san)} />
                    </Block>
                  </>
                )}
              </aside>
            </motion.div>
          )}
        </AnimatePresence>
      </main>

      <footer className="border-t border-ink">
        <div className="mx-auto flex max-w-[1440px] flex-wrap justify-between gap-2 px-5 py-4 lg:px-10">
          <span className="label">Données · PGN Mentor, Lichess Elite 2024</span>
          <span className="label">behavior-cloning · contrat d'encodage v{info?.encoding_version ?? 1}</span>
        </div>
      </footer>
    </div>
  );
}

function Block({ title, note, children }: { title: string; note?: string; children: React.ReactNode }) {
  return (
    <section>
      <div className="mb-3 flex items-baseline justify-between gap-4 border-b border-ink pb-2">
        <h2 className="font-serif text-[26px] leading-none">{title}</h2>
        {note && <span className="label">{note}</span>}
      </div>
      {children}
    </section>
  );
}

function TextButton(props: React.ButtonHTMLAttributes<HTMLButtonElement>) {
  return (
    <button
      {...props}
      className="font-mono text-[11px] uppercase tracking-[0.16em] underline-offset-4 hover:underline disabled:opacity-35 disabled:hover:no-underline"
    />
  );
}

function Check({ on, children, ...props }: { on: boolean } & React.ButtonHTMLAttributes<HTMLButtonElement>) {
  return (
    <button {...props} className="flex items-center gap-2 font-mono text-[11px] uppercase tracking-[0.16em]">
      <span
        className="grid size-3.5 place-items-center border border-ink transition-colors"
        style={{ background: on ? "var(--accent)" : "transparent", borderColor: on ? "var(--accent)" : undefined }}
      />
      {children}
    </button>
  );
}

function MoveList({ sans }: { sans: string[] }) {
  const end = useRef<HTMLDivElement>(null);
  useEffect(() => end.current?.scrollIntoView({ block: "nearest" }), [sans.length]);
  if (!sans.length) return <p className="font-serif text-lg italic text-muted">Aucun coup pour l'instant.</p>;
  const rows = [];
  for (let i = 0; i < sans.length; i += 2) rows.push([sans[i], sans[i + 1]]);
  return (
    <div className="max-h-60 overflow-y-auto pr-1">
      <div className="grid grid-cols-[2.4rem_1fr_1fr] gap-y-1">
        {rows.map(([w, b], i) => (
          <div key={i} className="contents">
            <span className="font-mono text-[11px] leading-7 text-muted">{i + 1}.</span>
            <span className="font-serif text-xl leading-7">{w}</span>
            <span className="font-serif text-xl leading-7">{b ?? ""}</span>
          </div>
        ))}
      </div>
      <div ref={end} />
    </div>
  );
}
