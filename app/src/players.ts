// Présentation des joueurs, indexée par le nom que renvoie /api/info (players.json du modèle)

export type PlayerMeta = {
  name: string;
  short: string;
  initials: string;
  years: string;
  title: string;
  style: string;
  accent: string;
};

export const PLAYER_META: Record<string, PlayerMeta> = {
  generique: {
    name: "Joueur Lichess 2200+",
    short: "Référence",
    initials: "Li",
    years: "Lichess Elite 2024",
    title: "Le style moyen des forts joueurs en ligne",
    style: "Le point de comparaison : ce que jouerait un bon joueur humain, sans personnalité particulière.",
    accent: "#94a3b8",
  },
  Tal: {
    name: "Mikhail Tal",
    short: "Tal",
    initials: "MT",
    years: "1936 – 1992",
    title: "Champion du monde 1960",
    style: "Le magicien de Riga : sacrifices, complications et attaque à tout prix.",
    accent: "#f97362",
  },
  Petrosian: {
    name: "Tigran Petrosian",
    short: "Petrosian",
    initials: "TP",
    years: "1929 – 1984",
    title: "Champion du monde 1963 – 1969",
    style: "Le tigre de fer : prophylaxie, sacrifices de qualité, la sécurité avant tout.",
    accent: "#2dd4bf",
  },
  Capablanca: {
    name: "José Raúl Capablanca",
    short: "Capablanca",
    initials: "JC",
    years: "1888 – 1942",
    title: "Champion du monde 1921 – 1927",
    style: "La machine d'échecs : simplicité, jeu positionnel naturel, finales limpides.",
    accent: "#fbbf24",
  },
  Fischer: {
    name: "Bobby Fischer",
    short: "Fischer",
    initials: "BF",
    years: "1943 – 2008",
    title: "Champion du monde 1972 – 1975",
    style: "Précision et énergie : ouvertures affûtées, jeu clair et combatif.",
    accent: "#4ade80",
  },
  Kasparov: {
    name: "Garry Kasparov",
    short: "Kasparov",
    initials: "GK",
    years: "né en 1963",
    title: "Champion du monde 1985 – 2000",
    style: "Dynamisme et préparation : initiative, attaque, ouvertures profondes.",
    accent: "#c084fc",
  },
  Carlsen: {
    name: "Magnus Carlsen",
    short: "Carlsen",
    initials: "MC",
    years: "né en 1990",
    title: "Champion du monde 2013 – 2023",
    style: "L'universel : pression durable, finales, exploite la moindre imprécision.",
    accent: "#60a5fa",
  },
};

export function metaOf(name: string): PlayerMeta {
  return (
    PLAYER_META[name] ?? {
      name,
      short: name,
      initials: name.slice(0, 2).toUpperCase(),
      years: "",
      title: "",
      style: "",
      accent: "#94a3b8",
    }
  );
}
