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

import type { Model, ProviderHeaders } from "@earendil-works/pi-ai";

export interface Snapshot {
	sessionId: string;
	provider: string;
	modelId: string;
	api: string;
	payload: unknown;
	/**
	 * The headers the main loop's request wrapper settled on for this request:
	 * session routing such as OpenCode's, attribution, and whatever
	 * `before_provider_headers` handlers added. The side question does not go
	 * through that wrapper, so these are the only copy it can get.
	 */
	headers?: ProviderHeaders;
}

let current: Snapshot | undefined;

/**
 * Headers seen since the last captured body.
 *
 * pi resolves headers before the adapter builds the body, so for one request
 * `before_provider_headers` fires first and `before_provider_request` second.
 * Compaction and summaries fire only the first; the next main request
 * overwrites what they left here before its body arrives.
 */
let pendingHeaders: ProviderHeaders | undefined;

/**
 * Remember the header object a request is about to send.
 *
 * Handlers edit it in place and in order, so the reference is kept and copied
 * only when the body arrives, by which point every handler has run.
 */
export function notePendingHeaders(headers: ProviderHeaders): void {
	pendingHeaders = headers;
}

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
 * The largest output cap that can only be a warm-up.
 *
 * The cache warmer asks for one token. Adapters are free to raise that to
 * their own floor — OpenAI Responses rejects anything under 16 and clamps —
 * so matching on exactly 1 would let those through. No real turn caps its
 * output this low, so the range is safe to claim.
 */
const PROBE_OUTPUT_CAP = 16;

/**
 * Whether a body is pi's cache-warming probe rather than a real turn.
 *
 * The warmer re-sends the current prefix with the output capped, and it
 * inherits the main loop's `onPayload`, so it reaches this hook looking
 * exactly like a main request. Its bytes are a fine prefix, but its output cap
 * is not: inheriting it would answer every side question in a token or two.
 * Skip it and keep the previous snapshot, which has the same prefix.
 */
export function isCacheWarmProbe(payload: unknown): boolean {
	if (!payload || typeof payload !== "object") return false;
	const body = payload as Record<string, unknown>;
	const capped = (value: unknown) => typeof value === "number" && value > 0 && value <= PROBE_OUTPUT_CAP;

	// `maxTokens` is Mistral's spelling.
	for (const field of ["max_tokens", "max_completion_tokens", "max_output_tokens", "maxTokens"]) {
		if (capped(body[field])) return true;
	}
	// Where the cap nests: Gemini REST, Pi's Google adapter, Bedrock, pi-messages.
	const nested: [string, string][] = [
		["generationConfig", "maxOutputTokens"],
		["config", "maxOutputTokens"],
		["inferenceConfig", "maxTokens"],
		["options", "maxTokens"],
	];
	for (const [parent, field] of nested) {
		const value = body[parent];
		if (value && typeof value === "object" && capped((value as Record<string, unknown>)[field])) return true;
	}
	return false;
}

/**
 * The body without the live objects some adapters put into it.
 *
 * Pi's Google adapters hand the SDK its abort signal inside the body, as
 * `config.abortSignal`. Node's `structuredClone` turns a signal into an empty
 * object and Bun's refuses it, so a copied signal is never one the SDK can
 * use. It is dropped here; the side question puts its own in.
 */
function withoutLiveObjects(payload: unknown): unknown {
	if (!payload || typeof payload !== "object") return payload;
	const config = (payload as Record<string, unknown>).config;
	if (!config || typeof config !== "object" || !("abortSignal" in config)) return payload;
	const { abortSignal: _signal, ...rest } = config as Record<string, unknown>;
	return { ...payload, config: rest };
}

/**
 * Keep a body as the snapshot for its session.
 *
 * The copy is taken here, and taken after the guards, so the identity test
 * still sees the object the hook was handed. A body that will not clone throws
 * out to the caller, which drops it rather than let the main request fail.
 */
export function recordSnapshot(snapshot: Snapshot): void {
	const headers = pendingHeaders;
	pendingHeaders = undefined;
	if (isOwnPayload(snapshot.payload)) return;
	if (isCacheWarmProbe(snapshot.payload)) return;
	current = {
		...snapshot,
		payload: structuredClone(withoutLiveObjects(snapshot.payload)),
		...(headers ? { headers: { ...headers } } : {}),
	};
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
	pendingHeaders = undefined;
}
