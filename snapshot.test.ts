/**
 * Which outbound bodies are worth keeping as the reusable prefix.
 *
 * Two kinds have to be turned away. The cache warmer's probe looks exactly like
 * a main request but caps the output at one token, and inheriting that cap
 * would answer every side question with a single token. A body this extension
 * produced itself must not become the prefix for the next one.
 */

import { beforeEach, describe, expect, test } from "bun:test";
import type { Model } from "@earendil-works/pi-ai";
import {
	getSnapshot,
	isCacheWarmProbe,
	markOwnPayload,
	notePendingHeaders,
	recordSnapshot,
	resetSnapshot,
} from "./snapshot.js";

const model = { provider: "anthropic", id: "claude-sonnet-5", api: "anthropic-messages" } as Model<any>;

function capture(payload: unknown, overrides: Partial<{ sessionId: string; modelId: string }> = {}) {
	recordSnapshot({
		sessionId: overrides.sessionId ?? "s1",
		provider: "anthropic",
		modelId: overrides.modelId ?? "claude-sonnet-5",
		api: "anthropic-messages",
		payload,
	});
}

beforeEach(() => {
	resetSnapshot();
});

describe("isCacheWarmProbe", () => {
	test("recognises the one-token probe across the field names providers use", () => {
		expect(isCacheWarmProbe({ max_tokens: 1 })).toBe(true);
		expect(isCacheWarmProbe({ max_completion_tokens: 1 })).toBe(true);
		expect(isCacheWarmProbe({ max_output_tokens: 1 })).toBe(true);
		expect(isCacheWarmProbe({ generationConfig: { maxOutputTokens: 1 } })).toBe(true);
		expect(isCacheWarmProbe({ maxTokens: 1 })).toBe(true);
		expect(isCacheWarmProbe({ config: { maxOutputTokens: 1 } })).toBe(true);
		expect(isCacheWarmProbe({ inferenceConfig: { maxTokens: 1 } })).toBe(true);
		expect(isCacheWarmProbe({ options: { maxTokens: 1 } })).toBe(true);
	});

	/** OpenAI Responses rejects a cap under 16 and clamps the probe up to it. */
	test("recognises a probe an adapter raised to its own floor", () => {
		expect(isCacheWarmProbe({ max_output_tokens: 16 })).toBe(true);
	});

	test("leaves a real request alone", () => {
		expect(isCacheWarmProbe({ max_tokens: 32000 })).toBe(false);
		expect(isCacheWarmProbe({ max_tokens: 17 })).toBe(false);
		expect(isCacheWarmProbe({ max_tokens: 0 })).toBe(false);
		expect(isCacheWarmProbe({})).toBe(false);
		expect(isCacheWarmProbe(undefined)).toBe(false);
	});
});

describe("recordSnapshot", () => {
	test("keeps the most recent main request", () => {
		capture({ messages: ["first"], system: "p", max_tokens: 100 });
		capture({ messages: ["second"], system: "p", max_tokens: 100 });
		expect(getSnapshot("s1", model)?.payload).toMatchObject({ messages: ["second"] });
	});

	test("skips the cache warmer's probe and keeps the request before it", () => {
		capture({ messages: ["real"], system: "p", max_tokens: 100 });
		capture({ messages: ["real"], system: "p", max_tokens: 1 });
		expect(getSnapshot("s1", model)?.payload).toMatchObject({ max_tokens: 100 });
	});

	test("skips a body this extension produced", () => {
		capture({ messages: ["real"], system: "p", max_tokens: 100 });
		const own = { messages: ["side question"], system: "p", max_tokens: 100 };
		markOwnPayload(own);
		capture(own);
		expect(getSnapshot("s1", model)?.payload).toMatchObject({ messages: ["real"] });
	});

	test("stores a copy, so the provider mutating its body afterwards cannot reach it", () => {
		const payload = { messages: ["first"], system: "p", max_tokens: 100 };
		capture(payload);
		payload.messages.push("added later");
		expect((getSnapshot("s1", model)?.payload as { messages: string[] }).messages).toEqual(["first"]);
	});

	/** Node clones a signal into `{}` and Bun refuses it; either way it is no signal the SDK can use. */
	test("drops the abort signal Pi's Google adapters put in the body, and leaves the main body alone", () => {
		const signal = new AbortController().signal;
		const payload = { model: "gemini", contents: [], config: { systemInstruction: "p", abortSignal: signal } };
		capture(payload);
		expect(getSnapshot("s1", model)?.payload).toEqual({ model: "gemini", contents: [], config: { systemInstruction: "p" } });
		expect(payload.config.abortSignal).toBe(signal);
	});
});

describe("captured headers", () => {
	test("the headers resolved for a request are kept with its body, as they stand when it arrives", () => {
		const headers: Record<string, string | null> = { "x-opencode-session": "s1" };
		notePendingHeaders(headers);
		// A later `before_provider_headers` handler, editing in place.
		headers["x-route"] = "blue";
		capture({ messages: [], system: "p", max_tokens: 100 });
		expect(getSnapshot("s1", model)?.headers).toEqual({ "x-opencode-session": "s1", "x-route": "blue" });
	});

	test("a body is not paired with headers from a request before it", () => {
		notePendingHeaders({ "x-route": "warm" });
		capture({ messages: [], system: "p", max_tokens: 1 });
		capture({ messages: [], system: "p", max_tokens: 100 });
		expect(getSnapshot("s1", model)?.headers).toBeUndefined();
	});
});

describe("getSnapshot", () => {
	test("returns nothing before the first main request", () => {
		expect(getSnapshot("s1", model)).toBeUndefined();
	});

	test("does not hand another session's prefix over", () => {
		capture({ messages: [], system: "p", max_tokens: 100 }, { sessionId: "s2" });
		expect(getSnapshot("s1", model)).toBeUndefined();
	});

	test("does not reuse a prefix captured under a different model", () => {
		capture({ messages: [], system: "p", max_tokens: 100 }, { modelId: "claude-opus-5" });
		expect(getSnapshot("s1", model)).toBeUndefined();
	});
});
