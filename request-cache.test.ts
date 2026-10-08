/** Exercise runSideQuestion through Pi's real Responses adapter, capturing HTTP locally. */
import { afterEach, describe, expect, test } from "bun:test";
import type { Context, Model, SimpleStreamOptions } from "@earendil-works/pi-ai";
import { streamSimple } from "@earendil-works/pi-ai/compat";
import type { ExtensionAPI, ExtensionCommandContext } from "@earendil-works/pi-coding-agent";
import { wrapQuestion } from "./prompt.js";
import { appendResponsesSideTurns, isResponsesBody, runSideQuestion, type ResponsesBody } from "./request.js";
import { recordSnapshot, resetSnapshot } from "./snapshot.js";

const sessionId = "btw-cache-regression";
const model: Model<"openai-responses"> = {
	id: "gpt-6-astra",
	name: "Responses test model",
	api: "openai-responses",
	provider: "test-responses",
	baseUrl: "http://127.0.0.1:1/v1",
	reasoning: true,
	input: ["text"],
	contextWindow: 272000,
	maxTokens: 128000,
	cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
};
const context: Context = {
	messages: [
		{ role: "system", content: "Answer questions about this project.", timestamp: 0 },
		{ role: "user", content: "Explain the parser.", timestamp: 0 },
	],
};

interface CapturedRequest {
	body: Record<string, any>;
	headers: Headers;
}

function harness() {
	const requests: CapturedRequest[] = [];
	let transcriptReads = 0;
	const captureStream = (model: Model<any>, context: Context, options?: SimpleStreamOptions) =>
		streamSimple(model, context, {
			...options,
			apiKey: "offline-test-key",
			maxRetries: 0,
			fetch: Object.assign(
				async (_url: string | URL | Request, init?: RequestInit) => {
					requests.push({ body: JSON.parse(String(init?.body)), headers: new Headers(init?.headers) });
					// Stop at the HTTP boundary. No provider call or simulated cache hit.
					throw new Error("offline test: request captured");
				},
				{ preconnect() {} },
			),
		});
	const ctx = {
		model,
		thinkingLevel: "high",
		modelRegistry: { streamSimple: captureStream },
		sessionManager: {
			buildSessionProjection: () => {
				transcriptReads++;
				return { messages: structuredClone(context.messages) };
			},
		},
	} as unknown as ExtensionCommandContext;
	return {
		requests,
		transcriptReads: () => transcriptReads,
		async main() {
			await captureStream(model, context, { sessionId, reasoning: "high", maxTokens: 2048 }).result();
			const request = requests.at(-1)!;
			recordSnapshot({ sessionId, provider: model.provider, modelId: model.id, api: model.api, payload: request.body });
			return request;
		},
		async side(history: { question: string; response: string; createdAt: number }[] = []) {
			const before = requests.length;
			await runSideQuestion({
				ctx,
				pi: {} as ExtensionAPI,
				sessionId,
				question: "What does that mean?",
				history,
				signal: new AbortController().signal,
				onText() {},
			});
			expect(requests).toHaveLength(before + 1);
			return requests.at(-1)!;
		},
	};
}

afterEach(resetSnapshot);

describe("Responses snapshot bodies", () => {
	test("recognizes array input, including requests without tools", () => {
		expect(isResponsesBody({ model: "test", input: [] })).toBe(true);
		for (const body of [null, undefined, "{}", { input: [] }, { model: "test", input: "hello" }, { messages: [] }]) {
			expect(isResponsesBody(body)).toBe(false);
		}
	});

	test("keeps opaque reasoning, tools, instructions and cache settings intact", () => {
		const body: ResponsesBody = {
			model: "test",
			instructions: "Original instructions",
			input: [
				{ role: "developer", content: "Original prompt" },
				{ type: "reasoning", id: "rs_original", encrypted_content: "opaque", summary: [] },
				{ type: "function_call", id: "fc_original", call_id: "call_original", name: "read", arguments: "{}" },
				{ type: "function_call_output", call_id: "call_original", output: "file contents" },
			],
			tools: [{ type: "function", name: "read", parameters: { type: "object", properties: {} } }],
			reasoning: { effort: "high", summary: "auto" },
			include: ["reasoning.encrypted_content"],
			max_output_tokens: 2048,
			prompt_cache_key: sessionId,
			prompt_cache_retention: "24h",
			prompt_cache_options: { ttl: "30m" },
			store: false,
		};
		const original = structuredClone(body);
		const after = appendResponsesSideTurns(body, [], "Question");
		expect(after.input.slice(0, body.input.length)).toEqual(original.input);
		expect({ ...after, input: original.input }).toEqual(original);
		expect(after.input).toHaveLength(original.input.length + 1);
		expect(body).toEqual(original);
	});
});

describe("Responses request cache reuse", () => {
	test("a side question preserves the captured input and every other body field", async () => {
		const h = harness();
		const main = await h.main();
		const side = await h.side();
		const { input: mainInput, ...mainOptions } = main.body;
		const { input: sideInput, ...sideOptions } = side.body;
		expect(sideOptions).toEqual(mainOptions);
		expect(sideInput.slice(0, mainInput.length)).toEqual(mainInput);
		expect(sideInput.slice(mainInput.length)).toEqual([
			{ role: "user", content: [{ type: "input_text", text: wrapQuestion("What does that mean?") }] },
		]);
		expect(h.transcriptReads()).toBe(0);
	});

	test("a captured body still needs session options to preserve HTTP routing headers", async () => {
		const h = harness();
		const main = await h.main();
		const side = await h.side();
		for (const header of ["session_id", "x-client-request-id"]) {
			expect(main.headers.get(header)).toBe(sessionId);
			expect(side.headers.get(header)).toBe(sessionId);
		}
	});

	test("the rebuild path also retains the session cache key and routing headers", async () => {
		resetSnapshot();
		const h = harness();
		const side = await h.side();
		expect(h.requests).toHaveLength(1);
		expect(h.transcriptReads()).toBe(1);
		expect(side.body.prompt_cache_key).toBe(sessionId);
		expect(side.headers.get("session_id")).toBe(sessionId);
		expect(side.headers.get("x-client-request-id")).toBe(sessionId);
	});

	test("repeated side questions append history without changing the snapshot", async () => {
		const h = harness();
		const main = await h.main();
		const original = structuredClone(main.body);
		await h.side();
		const side = await h.side([{ question: "Earlier question", response: "Earlier answer", createdAt: 0 }]);
		expect(side.body.input.slice(0, original.input.length)).toEqual(original.input);
		expect(side.body.input.slice(original.input.length)).toEqual([
			{ role: "user", content: [{ type: "input_text", text: "Earlier question" }] },
			{ role: "assistant", content: "Earlier answer" },
			{ role: "user", content: [{ type: "input_text", text: wrapQuestion("What does that mean?") }] },
		]);
		expect(main.body).toEqual(original);
		expect(h.transcriptReads()).toBe(0);
	});
});
