import { motion } from "motion/react";

import { metaOf } from "../players";

/** Monogramme carré : initiales en italique, rempli à la couleur du joueur quand il est actif. */
export function Monogram({ name, size = 36, active = false }: { name: string; size?: number; active?: boolean }) {
  const m = metaOf(name);
  return (
    <span
      className="grid shrink-0 place-items-center border font-serif italic leading-none transition-colors"
      style={{
        width: size,
        height: size,
        fontSize: size * 0.48,
        borderColor: active ? m.accent : "var(--color-ink)",
        background: active ? m.accent : "transparent",
        color: active ? "var(--color-paper)" : "var(--color-ink)",
      }}
    >
      {m.initials}
    </span>
  );
}

/** Ordre d'affichage : les légendes d'abord, la référence Lichess en dernier. */
export function displayOrder(players: string[]): number[] {
  return [...players.keys()].filter((i) => i !== 0).concat(players.length ? [0] : []);
}

/** Index numéroté de la page Joueurs. */
export function PlayerIndex({ players, selected, onSelect }: { players: string[]; selected: number; onSelect: (id: number) => void }) {
  return (
    <ol className="no-scrollbar flex overflow-x-auto border-b border-rule lg:block lg:overflow-visible lg:border-b-0">
      {displayOrder(players).map((id, i) => {
        const m = metaOf(players[id]);
        const active = id === selected;
        return (
          <li key={id} className="shrink-0 lg:border-t lg:border-rule">
            <button
              onClick={() => onSelect(id)}
              className="group relative grid w-full grid-cols-[2rem_auto] items-baseline gap-x-3 px-3 py-3 text-left lg:grid-cols-[2.2rem_1fr_auto] lg:px-0 lg:py-4"
            >
              {active && (
                <motion.span
                  layoutId="index-mark"
                  className="absolute bottom-0 left-0 h-[3px] w-full lg:top-0 lg:bottom-auto lg:h-full lg:w-[3px] lg:-translate-x-4"
                  style={{ background: m.accent }}
                  transition={{ type: "spring", stiffness: 420, damping: 38 }}
                />
              )}
              <span className="font-mono text-[11px] text-muted">{String(i + 1).padStart(2, "0")}</span>
              <span
                className="whitespace-nowrap font-serif text-2xl leading-none transition-transform duration-300 group-hover:translate-x-1 lg:text-[34px]"
                style={{ color: active ? m.accent : undefined }}
              >
                {m.last}
              </span>
              <span className="hidden font-mono text-[10.5px] text-muted lg:block">{m.years}</span>
            </button>
          </li>
        );
      })}
    </ol>
  );
}

/** Sélecteur compact des pages de jeu : une rangée de monogrammes. */
export function PlayerSwitcher({ players, selected, onSelect }: { players: string[]; selected: number; onSelect: (id: number) => void }) {
  const m = metaOf(players[selected] ?? "");
  return (
    <div>
      <div className="flex flex-wrap gap-1.5">
        {displayOrder(players).map((id) => (
          <button key={id} onClick={() => onSelect(id)} title={metaOf(players[id]).short} className="transition-transform hover:-translate-y-0.5">
            <Monogram name={players[id]} active={id === selected} />
          </button>
        ))}
      </div>
      <div className="mt-4">
        <div className="font-serif text-4xl leading-none">
          {m.first} <em style={{ color: m.accent }}>{m.last}</em>
        </div>
        <div className="mt-1.5 font-serif text-lg italic text-muted">{m.nickname}</div>
      </div>
    </div>
  );
}
