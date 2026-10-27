import { AnimatePresence, motion } from "motion/react";

import { metaOf } from "../players";

type Props = {
  players: string[];
  selected: number;
  onSelect: (id: number) => void;
};

export function Avatar({ name, size = 40 }: { name: string; size?: number }) {
  const m = metaOf(name);
  return (
    <div
      className="grid shrink-0 place-items-center rounded-full font-display font-semibold text-[#0b0d12]"
      style={{
        width: size,
        height: size,
        fontSize: size * 0.36,
        background: `radial-gradient(circle at 30% 25%, #fff8, transparent 55%), ${m.accent}`,
        boxShadow: `0 0 0 1px #ffffff1a, 0 6px 20px -6px ${m.accent}`,
      }}
    >
      {m.initials}
    </div>
  );
}

export function PlayerList({ players, selected, onSelect }: Props) {
  // Les joueurs iconiques d'abord, la référence Lichess en dernier
  const order = [...players.keys()].filter((i) => i !== 0).concat(players.length ? [0] : []);
  return (
    <nav className="no-scrollbar flex gap-2 overflow-x-auto pb-1 lg:flex-col lg:overflow-visible lg:pb-0">
      {order.map((id) => {
        const m = metaOf(players[id]);
        const active = id === selected;
        return (
          <motion.button
            key={id}
            layout
            onClick={() => onSelect(id)}
            className={`relative min-w-[200px] rounded-2xl border p-3 text-left transition-colors lg:min-w-0 ${
              active ? "border-white/20 bg-white/[0.07]" : "border-white/5 bg-white/[0.02] hover:bg-white/[0.05]"
            }`}
          >
            {active && (
              <motion.div
                layoutId="player-glow"
                className="pointer-events-none absolute inset-0 rounded-2xl"
                style={{ boxShadow: `inset 0 0 0 1px ${m.accent}66, 0 10px 40px -18px ${m.accent}` }}
                transition={{ type: "spring", stiffness: 400, damping: 34 }}
              />
            )}
            <div className="flex items-center gap-3">
              <Avatar name={players[id]} />
              <div className="min-w-0">
                <div className="truncate font-display text-[15px] font-semibold text-white">{m.name}</div>
                <div className="truncate text-xs text-slate-400">{id === 0 ? m.years : m.title}</div>
              </div>
            </div>
            <AnimatePresence initial={false}>
              {active && m.style && (
                <motion.p
                  initial={{ opacity: 0, height: 0 }}
                  animate={{ opacity: 1, height: "auto" }}
                  exit={{ opacity: 0, height: 0 }}
                  className="overflow-hidden text-[13px] leading-relaxed text-slate-300"
                >
                  <span className="block pt-2.5">{m.style}</span>
                </motion.p>
              )}
            </AnimatePresence>
          </motion.button>
        );
      })}
    </nav>
  );
}
