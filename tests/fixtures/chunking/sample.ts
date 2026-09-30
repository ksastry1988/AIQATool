export interface Config {
  retries: number;
}

type Mode = "fast" | "safe";

enum Level {
  Low,
  High,
}

export const makeConfig = (mode: Mode): Config => ({ retries: mode === "fast" ? 1 : 3 });

export abstract class Store {
  abstract get(key: string): string;
}
