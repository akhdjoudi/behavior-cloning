import { Chessground } from "@lichess-org/chessground";
import type { Api } from "@lichess-org/chessground/api";
import type { DrawBrushes, DrawShape } from "@lichess-org/chessground/draw";
import type { Color, Key } from "@lichess-org/chessground/types";
import { useEffect, useRef } from "react";

import type { MoveProb } from "../api";
import { PLAYER_META } from "../players";

type Props = {
  fen: string;
  orientation: Color;
  turnColor: Color;
  dests?: Map<Key, Key[]>;
  movableColor?: Color | "both";
  lastMove?: [Key, Key];
  check?: boolean;
  shapes?: DrawShape[];
  onMove?: (from: Key, to: Key) => void;
  viewOnly?: boolean;
  coordinates?: boolean;
  accent: string;
};

// Une brosse par rang : le coup préféré est le plus opaque
const OPACITY = [0.92, 0.7, 0.55, 0.45, 0.38];

function brushes(accent: string): DrawBrushes {
  const base = (key: string, color: string) => ({ key, color, opacity: 0.8, lineWidth: 10 });
  const ranked = Object.fromEntries(
    OPACITY.map((opacity, i) => [`p${i}`, { key: `p${i}`, color: accent, opacity, lineWidth: 10 }]),
  );
  // Une brosse par joueur, pour le mode duel (une flèche par joueur sur le même échiquier)
  const players = Object.fromEntries(
    Object.entries(PLAYER_META).map(([name, m]) => [`pl-${name}`, { key: `pl-${name}`, color: m.accent, opacity: 0.85, lineWidth: 10 }]),
  );
  return {
    green: base("g", "#15781B"),
    red: base("r", "#882020"),
    blue: base("b", "#003088"),
    yellow: base("y", "#e68f00"),
    ...ranked,
    ...players,
  };
}

/** Flèches des meilleurs coups : épaisseur selon la probabilité, opacité selon le rang. */
export function arrowsFor(moves: MoveProb[] | undefined, count = 5, labels = true): DrawShape[] {
  if (!moves?.length) return [];
  const top = moves[0].prob;
  return moves.slice(0, count).map((m, i) => ({
    orig: m.uci.slice(0, 2) as Key,
    dest: m.uci.slice(2, 4) as Key,
    brush: `p${i}`,
    modifiers: { lineWidth: Math.round(4 + 12 * Math.sqrt(m.prob / top)) },
    label: labels && m.prob >= 0.04 ? { text: `${Math.round(m.prob * 100)}`, fill: "#0b0d12" } : undefined,
  }));
}

export function Board(props: Props) {
  const el = useRef<HTMLDivElement>(null);
  const api = useRef<Api | null>(null);
  const onMove = useRef(props.onMove);
  onMove.current = props.onMove;

  useEffect(() => {
    api.current = Chessground(el.current!, {
      animation: { enabled: true, duration: 220 },
      coordinates: props.coordinates ?? true,
      viewOnly: props.viewOnly,
      highlight: { lastMove: true, check: true },
      premovable: { enabled: false },
      drawable: { enabled: false, visible: true, brushes: brushes(props.accent) },
      movable: { free: false, showDests: true, events: { after: (from, to) => onMove.current?.(from, to) } },
    });
    return () => api.current?.destroy();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    api.current?.set({
      fen: props.fen,
      orientation: props.orientation,
      turnColor: props.turnColor,
      lastMove: props.lastMove,
      check: props.check,
      movable: { color: props.movableColor, dests: props.dests ?? new Map() },
    });
  }, [props.fen, props.orientation, props.turnColor, props.lastMove, props.check, props.movableColor, props.dests]);

  useEffect(() => {
    api.current?.set({ drawable: { brushes: brushes(props.accent) } });
    api.current?.setAutoShapes(props.shapes ?? []);
  }, [props.shapes, props.accent]);

  return <div ref={el} className="board aspect-square w-full" />;
}
