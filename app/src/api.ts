// Client de l'API Python (src/chessstyle/server.py)

export type MoveProb = { uci: string; san: string; prob: number };

export type Info = {
  players: string[];
  trained: boolean;
  model: string;
  encoding_version: number;
  engine: string | null; // Stockfish côté serveur, null s'il n'est pas installé
};

/** Évaluation Stockfish du point de vue des Blancs : centipions, ou mat en N (négatif si les Noirs matent). */
export type Evaluation = {
  cp: number | null;
  mate: number | null;
  winner?: "white" | "black";
  depth?: number;
  best?: string | null;
  best_san?: string | null;
};

export async function getInfo(): Promise<Info> {
  const res = await fetch("/api/info");
  if (!res.ok) throw new Error(`API ${res.status}`);
  return res.json();
}

/** Coups légaux triés par probabilité, pour chaque joueur demandé. */
export async function predict(fen: string, players: number[]): Promise<Map<number, MoveProb[]>> {
  const res = await fetch("/api/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ fen, players }),
  });
  if (!res.ok) throw new Error(`API ${res.status}`);
  const data: { players: { id: number; moves: MoveProb[] }[] } = await res.json();
  return new Map(data.players.map((p) => [p.id, p.moves]));
}

export async function evaluate(fen: string): Promise<Evaluation> {
  const res = await fetch("/api/eval", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ fen }),
  });
  if (!res.ok) throw new Error(`API ${res.status}`);
  return res.json();
}

/** Tire un coup : température 0 = le plus probable, 1 = la distribution du modèle telle quelle. */
export function sampleMove(moves: MoveProb[], temperature: number): MoveProb {
  if (temperature <= 0 || moves.length === 1) return moves[0];
  const weights = moves.map((m) => Math.pow(m.prob, 1 / temperature));
  let r = Math.random() * weights.reduce((a, b) => a + b, 0);
  for (let i = 0; i < moves.length; i++) {
    r -= weights[i];
    if (r <= 0) return moves[i];
  }
  return moves[0];
}
