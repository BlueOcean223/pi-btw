/**
 * Per-session side-question state, held in memory only.
 *
 * Nothing here is written to the session file. `pi.appendEntry` would push the
 * exchanges into the session tree, and then a transcript of the main task would
 * carry questions the main task never saw.
 *
 * Everything is keyed by session id, and every write names the session that
 * asked rather than the one that is open. A session switch therefore neither
 * clears the map nor misfiles an answer that arrives after it: switching back
 * inside one process finds the list intact.
 */

import type { SideResult } from "./request.js";

/** One answered side question, as it will be replayed into the next request. */
export interface Exchange {
	question: string;
	/** The answer as it was shown, including any notice appended to it. */
	response: string;
	fallbackNotice?: string;
	createdAt: number;
}

/** Claude Code's `btwHistory` cap. Older exchanges fall off the front. */
export const MAX_HISTORY = 20;

const histories = new Map<string, Exchange[]>();

/** The stored exchanges for a session, oldest first. Returns a copy. */
export function getHistory(sessionId: string): Exchange[] {
	return [...(histories.get(sessionId) ?? [])];
}

export function appendExchange(sessionId: string, exchange: Exchange): void {
	const list = histories.get(sessionId) ?? [];
	list.push(exchange);
	histories.set(sessionId, list.slice(-MAX_HISTORY));
}

/**
 * The `x` key: drop the replay list, keeping at most the one exchange the
 * panel is showing.
 *
 * Keeping it matters beyond the redraw. The panel goes on displaying that
 * answer either way, so dropping it from storage as well would make the panel
 * disagree with itself: close it, reopen with an empty `/btw`, and the answer
 * still on screen a second ago is gone.
 */
export function clearHistory(sessionId: string, keep?: Exchange): void {
	if (keep) histories.set(sessionId, [keep]);
	else histories.delete(sessionId);
}

/** Test seam. Not called by the extension. */
export function resetAllHistory(): void {
	histories.clear();
	inflight.clear();
}

/**
 * A side question whose request is still open.
 *
 * `text` and `result` are the reason this is an object rather than a bare
 * promise: the panel can be closed with Ctrl+] and reopened with an empty
 * `/btw`, and the reopened panel has to show everything that streamed in while
 * it was gone, not just what arrives after it subscribes.
 */
export interface Inflight {
	sessionId: string;
	question: string;
	controller: AbortController;
	/** Answer text so far. */
	text: string;
	/** Set once the stream ends. Undefined while the request is open. */
	result?: SideResult;
	settled: Promise<SideResult>;
	/** Panels currently rendering this request. */
	listeners: Set<() => void>;
}

const inflight = new Map<string, Inflight>();

export function getInflight(sessionId: string): Inflight | undefined {
	return inflight.get(sessionId);
}

export function setInflight(request: Inflight): void {
	inflight.set(request.sessionId, request);
}

/** Forget `request`, unless the slot already holds a newer one. */
export function clearInflight(request: Inflight): void {
	if (inflight.get(request.sessionId) === request) inflight.delete(request.sessionId);
}

export function notifyInflight(request: Inflight): void {
	for (const listener of request.listeners) listener();
}
