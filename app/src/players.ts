// Fiches des joueurs, indexées par le nom que renvoie /api/info (players.json du modèle).
// Parties et positions : statistiques d'ingestion de docs/DECISIONS.md.

export type PlayerMeta = {
  first: string;
  last: string;
  short: string;
  initials: string;
  years: string;
  born: string;
  title: string;
  nickname: string;
  style: string;
  traits: string[];
  games: number;
  positions: number;
  accent: string;
};

export const PLAYER_META: Record<string, PlayerMeta> = {
  Tal: {
    first: "Mikhail",
    last: "Tal",
    short: "Tal",
    initials: "MT",
    years: "1936 – 1992",
    born: "Riga",
    title: "Champion du monde 1960",
    nickname: "Le magicien de Riga",
    style: "Il sacrifiait d'abord et calculait ensuite. Ses parties sont des incendies : pièces en prise, rois exposés, complications que l'adversaire ne parvient plus à démêler.",
    traits: ["Sacrifices", "Attaque", "Complications"],
    games: 2371,
    positions: 84262,
    accent: "#cf3a1f",
  },
  Petrosian: {
    first: "Tigran",
    last: "Petrosian",
    short: "Petrosian",
    initials: "TP",
    years: "1929 – 1984",
    born: "Tbilissi",
    title: "Champion du monde 1963 – 1969",
    nickname: "Le tigre de fer",
    style: "Le maître de la prophylaxie : il devinait le plan adverse avant qu'il n'existe. Difficile à battre, il sacrifiait volontiers la qualité pour verrouiller une position.",
    traits: ["Prophylaxie", "Défense", "Qualité"],
    games: 1895,
    positions: 66412,
    accent: "#1c6a70",
  },
  Capablanca: {
    first: "José Raúl",
    last: "Capablanca",
    short: "Capablanca",
    initials: "JC",
    years: "1888 – 1942",
    born: "La Havane",
    title: "Champion du monde 1921 – 1927",
    nickname: "La machine d'échecs",
    style: "Une clarté désarmante : il simplifiait, échangeait au bon moment et gagnait des finales que d'autres auraient annulées. Le jeu positionnel à l'état naturel.",
    traits: ["Simplicité", "Finales", "Position"],
    games: 597,
    positions: 23346,
    accent: "#b07b1c",
  },
  Fischer: {
    first: "Bobby",
    last: "Fischer",
    short: "Fischer",
    initials: "BF",
    years: "1943 – 2008",
    born: "Chicago",
    title: "Champion du monde 1972 – 1975",
    nickname: "Le prodige de Brooklyn",
    style: "Précision et énergie. Un répertoire étroit mais affûté, presque toujours 1.e4, et des parties d'une clarté telle qu'elles semblent écrites à l'avance.",
    traits: ["Précision", "1.e4", "Clarté"],
    games: 786,
    positions: 32159,
    accent: "#2d6a3c",
  },
  Kasparov: {
    first: "Garry",
    last: "Kasparov",
    short: "Kasparov",
    initials: "GK",
    years: "né en 1963",
    born: "Bakou",
    title: "Champion du monde 1985 – 2000",
    nickname: "L'ogre de Bakou",
    style: "Dynamisme et préparation monstre. Il prenait l'initiative dès l'ouverture et ne la rendait plus, transformant la moindre activité en attaque.",
    traits: ["Initiative", "Dynamisme", "Préparation"],
    games: 1963,
    positions: 74781,
    accent: "#7a2c62",
  },
  Carlsen: {
    first: "Magnus",
    last: "Carlsen",
    short: "Carlsen",
    initials: "MC",
    years: "né en 1990",
    born: "Tønsberg",
    title: "Champion du monde 2013 – 2023",
    nickname: "Le Mozart des échecs",
    style: "L'universel. Il joue des positions égales pendant des heures, maintient une pression sans faille et convertit la plus petite imprécision en finale gagnée.",
    traits: ["Universalité", "Pression", "Finales"],
    games: 6488,
    positions: 300640,
    accent: "#2544a3",
  },
  generique: {
    first: "Joueur",
    last: "Lichess",
    short: "Lichess",
    initials: "Li",
    years: "Elite 2024",
    born: "En ligne",
    title: "Référence 2200+ Elo",
    nickname: "Le joueur sans visage",
    style: "La moyenne de milliers de forts joueurs en ligne : ce que jouerait un bon humain sans personnalité particulière. Le point de comparaison de tous les styles.",
    traits: ["Référence", "Pratique", "Moyenne humaine"],
    games: 20157,
    positions: 2156452,
    accent: "#57524a",
  },
};

export function metaOf(name: string): PlayerMeta {
  return (
    PLAYER_META[name] ?? {
      first: name,
      last: "",
      short: name,
      initials: name.slice(0, 2).toUpperCase(),
      years: "",
      born: "",
      title: "",
      nickname: "",
      style: "",
      traits: [],
      games: 0,
      positions: 0,
      accent: "#57524a",
    }
  );
}

// Espace insécable classique : la police serif n'a pas le glyphe de l'espace fine (U+202F)
export const fmt = (n: number) => n.toLocaleString("fr-FR").replace(/ /g, " ");
