import { log } from "./log.js";

/** Handles a request. */
export function handle(req) {
  return log(req);
}

const retry = async (fn) => {
  return fn();
};

const MAX = 3;

class Client {
  send() {
    return MAX;
  }
}

app.get("/", (req) => handle(req));
