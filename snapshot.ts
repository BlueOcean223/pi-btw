/**
 * The last request body the main loop actually sent.
 *
 * Anthropic's prompt cache keys on prefix bytes: system, tools, model,
 * thinking, and the message prefix. A side question that rebuilds those from
 * the current settings gets bytes that are close but not identical, and a
 * near-miss costs a full cache write on a context that can be hundreds of
 * thousands of tokens. Reusing the exact body the main loop sent and appending
 * to it is what makes the second side question cheap.
 *
 * What this captures is the request, not the response. The assistant message
 * that request produced, and any tool results recorded since, are not in it. A
 * side question therefore sees the conversation as of the main loop's last
 * outbound request; the next main request refreshes it.
 */

import type { Model } from "@earendil-works/pi-ai";

export interface Snapshot {
	sessionId: string;
	provider: string;
	modelId: string;
	api: string;
	payload: unknown;
}

let current: Snapshot | undefined;

/**
 * Bodies this extension produced.
 *
 * `before_provider_request` only fires for the main agent loop today, so a side
 * question cannot overwrite the snapshot with its own body. The design asks for
 * a re-entrancy guard in case pi later routes every outbound request through
 * the hook; a depth counter would be wrong here, because a side question runs
 * concurrently with the main loop by construction and the counter would blank
 * out legitimate main-loop snapshots for the whole time one is open. Matching
 * on the payload object itself has no such window.
 */
const ownPayloads = new WeakSet<object>();

export function markOwnPayload(payload: unknown): void {
	if (payload && typeof payload === "object") ownPayloads.add(payload);
}

function isOwnPayload(payload: unknown): boolean {
	return !!payload && typeof payload === "object" && ownPayloads.has(payload);
}

/**
 * Whether a body is pi's cache-warming probe rather than a real turn.
 *
 * The cache warmer re-sends the current prefix with the output capped at one
 * token, and it inherits the main loop's `onPayload`, so it reaches this hook
 * looking exactly like a main request. Its bytes are a fine prefix, but its
 * output cap is not: inheriting it would answer every side question with one
 * token. Skip it and keep the previous snapshot, which has the same prefix.
 */
export function isCacheWarmProbe(payload: unknown): boolean {
	if (!payload || typeof payload !== "object") return false;
	const body = payload as Record<string, unknown>;
	for (const field of ["max_tokens", "max_completion_tokens", "max_output_tokens"]) {
		if (body[field] === 1) return true;
	}
	const generation = body.generationConfig;
	if (generation && typeof generation === "object") {
		if ((generation as Record<string, unknown>).maxOutputTokens === 1) return true;
	}
	return false;
}

/**
 * Keep a body as the snapshot for its session.
 *
 * The copy is taken here, and taken after the guards, so the identity test
 * still sees the object the hook was handed. A body that will not clone throws
 * out to the caller, which drops it rather than let the main request fail.
 */
export function recordSnapshot(snapshot: Snapshot): void {
	if (isOwnPayload(snapshot.payload)) return;
	if (isCacheWarmProbe(snapshot.payload)) return;
	current = { ...snapshot, payload: structuredClone(snapshot.payload) };
}

/**
 * The snapshot for this session and model, or undefined when there is none to
 * reuse.
 *
 * The model has to match. Claude Code keeps the snapshot bytes and swaps in the
 * current model when the user switched models mid-session; here the model id
 * lives inside the captured body, so sending it under a different model would
 * either request the old model or hit a provider that never saw it. A miss
 * costs one uncached side question after a model switch, and the next main
 * request replaces the snapshot.
 */
export function getSnapshot(sessionId: string, model: Model<any>): Snapshot | undefined {
	if (!current) return undefined;
	if (current.sessionId !== sessionId) return undefined;
	if (current.provider !== model.provider || current.modelId !== model.id || current.api !== model.api) {
		return undefined;
	}
	return current;
}

/** Test seam. Not called by the extension. */
export function resetSnapshot(): void {
	current = undefined;
}
