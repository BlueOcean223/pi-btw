/**
 * The snapshot path for each API Pi ships, through Pi's real adapters.
 *
 * Each test sends one main request, keeps its body the way the
 * `before_provider_request` hook does, then sends one side question and reads
 * what reached the wire. Requests stop at a local boundary: a `fetch` that
 * throws, or a local server for the adapters whose SDK will not take a custom
 * `fetch`. No provider is called.
 */
import { afterAll, afterEach, beforeAll, describe, expect, test } from "bun:test";
import type { Context, Model, ModelsSimpleStreamOptions, ProviderHeaders, SimpleStreamOptions } from "@earendil-works/pi-ai";
import { streamSimple } from "@earendil-works/pi-ai/compat";
import type { ExtensionAPI, ExtensionCommandContext } from "@earendil-works/pi-coding-agent";
import { wrapQuestion } from "./prompt.js";
import { runSideQuestion } from "./request.js";
import { notePendingHeaders, recordSnapshot, resetSnapshot } from "./snapshot.js";

const sessionId = "btw-apis";
const QUESTION = "What does that mean?";

interface Wire {
	body: Record<string, any>;
	headers: Headers;
}

/** A local endpoint for SDKs that bring their own HTTP client. Answers every request with a 400. */
let server: ReturnType<typeof Bun.serve>;
const served: Wire[] = [];
beforeAll(() => {
	server = Bun.serve({
		port: 0,
		async fetch(req) {
			served.push({ body: JSON.parse(await req.text()), headers: req.headers });
			return Response.json({ message: "offline test: request captured" }, { status: 400 });
		},
	});
});
afterAll(() => server.stop(true));
afterEach(() => {
	resetSnapshot();
	served.length = 0;
});

function model(api: string, overrides: Partial<Model<any>> = {}): Model<any> {
	return {
		id: "test-model",
		name: "test model",
		api,
		provider: `test-${api}`,
		baseUrl: "http://127.0.0.1:1/v1",
		reasoning: false,
		input: ["text"],
		contextWindow: 200000,
		maxTokens: 8192,
		cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
		...overrides,
	} as Model<any>;
}

const tool = {
	name: "read",
	description: "Read a file",
	parameters: { type: "object", properties: { path: { type: "string" } }, required: ["path"] },
};

function conversation(withTools: boolean): Context {
	return {
		messages: [
			{
				role: "system",
				content: "Answer questions about this project.",
				...(withTools ? { toolsAdded: [tool] } : {}),
				timestamp: 0,
			},
			{ role: "user", content: [{ type: "text", text: "Explain the parser." }], timestamp: 0 },
		] as Context["messages"],
	};
}

/**
 * Drive one main request and one side question for a model.
 *
 * `wire` selects where requests are read from: the harness's own throwing
 * `fetch`, or the local server.
 */
function harness(m: Model<any>, wire: "fetch" | "server", extra: Partial<SimpleStreamOptions> = {}) {
	const fetched: Wire[] = [];
	const sideOptions: SimpleStreamOptions[] = [];
	const payloads: unknown[] = [];
	let transcriptReads = 0;
	const context = conversation(true);
	const send = (target: Model<any>, ctx: Context, options?: SimpleStreamOptions) =>
		streamSimple(target, ctx, {
			...options,
			apiKey: "offline-test-key",
			maxRetries: 0,
			...extra,
			onPayload: async (payload, mm) => {
				const next = await options?.onPayload?.(payload, mm);
				payloads.push(next ?? payload);
				return next;
			},
			...(wire === "fetch"
				? {
						fetch: Object.assign(
							async (_url: string | URL | Request, init?: RequestInit) => {
								// Codex compresses its SSE body; nothing here reads that one.
								const body = typeof init?.body === "string" ? JSON.parse(init.body) : {};
								fetched.push({ body, headers: new Headers(init?.headers) });
								throw new Error("offline test: request captured");
							},
							{ preconnect() {} },
						),
					}
				: {}),
		});
	const requests = () => (wire === "fetch" ? fetched : served);
	const ctx = {
		model: m,
		thinkingLevel: "off",
		modelRegistry: {
			// The registry applies `transformHeaders` after resolving auth; the
			// adapter-level `streamSimple` used here does not, so do it as it would.
			streamSimple: (target: Model<any>, c: Context, options?: ModelsSimpleStreamOptions) => {
				sideOptions.push(options!);
				const { transformHeaders, ...rest } = options ?? {};
				const headers = transformHeaders ? (transformHeaders(rest.headers ?? {}) as ProviderHeaders) : rest.headers;
				return send(target, c, { ...rest, ...(headers ? { headers } : {}) });
			},
		},
		sessionManager: {
			buildSessionProjection: () => {
				transcriptReads++;
				return { messages: structuredClone(context.messages) };
			},
		},
	} as unknown as ExtensionCommandContext;
	return {
		sideOptions,
		payloads,
		transcriptReads: () => transcriptReads,
		async main(withTools = true) {
			const main = conversation(withTools);
			context.messages = main.messages;
			await send(m, main, {
				sessionId,
				onPayload: (payload, mm) => {
					recordSnapshot({ sessionId, provider: mm.provider, modelId: mm.id, api: mm.api, payload });
					return undefined;
				},
			}).result();
			return requests().at(-1)!;
		},
		async side(settings: Record<string, unknown> = {}) {
			const before = requests().length;
			await runSideQuestion({
				ctx,
				pi: { getThinkingLevel: () => "off", getSettings: () => settings } as unknown as ExtensionAPI,
				sessionId,
				question: QUESTION,
				history: [],
				signal: new AbortController().signal,
				onText() {},
			});
			expect(requests().length).toBe(before + 1);
			return requests().at(-1)!;
		},
	};
}

describe("Chat Completions", () => {
	test("a request without tools is reused, not rebuilt", async () => {
		const h = harness(model("openai-completions"), "fetch");
		const main = await h.main(false);
		expect(main.body.tools).toBeUndefined();
		const side = await h.side();
		expect(h.transcriptReads()).toBe(0);
		const { messages: mainMessages, ...mainOptions } = main.body;
		const { messages: sideMessages, ...sideOptions } = side.body;
		expect(sideOptions).toEqual(mainOptions);
		expect(sideMessages.slice(0, mainMessages.length)).toEqual(mainMessages);
		expect(sideMessages.slice(mainMessages.length)).toEqual([{ role: "user", content: wrapQuestion(QUESTION) }]);
	});
});

describe("Mistral", () => {
	test("the captured messages are reused", async () => {
		const h = harness(model("mistral-conversations"), "fetch");
		const main = await h.main();
		const side = await h.side();
		expect(h.transcriptReads()).toBe(0);
		expect(side.body.messages.slice(0, main.body.messages.length)).toEqual(main.body.messages);
		expect(side.body.messages.at(-1)).toEqual({ role: "user", content: wrapQuestion(QUESTION) });
	});
});

describe("Google", () => {
	const gemini = () => model("google-generative-ai", { baseUrl: `http://127.0.0.1:${server.port}` });

	test("the captured contents are reused", async () => {
		const h = harness(gemini(), "server");
		const main = await h.main();
		const side = await h.side();
		expect(h.transcriptReads()).toBe(0);
		const { contents: mainContents, ...mainRest } = main.body;
		const { contents: sideContents, ...sideRest } = side.body;
		expect(sideRest).toEqual(mainRest);
		expect(sideContents.slice(0, mainContents.length)).toEqual(mainContents);
		expect(sideContents.slice(mainContents.length)).toEqual([
			{ role: "user", parts: [{ text: wrapQuestion(QUESTION) }] },
		]);
	});

	test("the side question gets its own abort signal, not the main loop's", async () => {
		const h = harness(gemini(), "server");
		await h.main();
		await h.side();
		const sent = h.payloads.at(-1) as { config: { abortSignal: unknown } };
		expect(sent.config.abortSignal).toBe(h.sideOptions.at(-1)!.signal);
	});
});

describe("Bedrock", () => {
	const bedrock = () => model("bedrock-converse-stream", { baseUrl: `http://127.0.0.1:${server.port}` });
	const env = { AWS_BEDROCK_SKIP_AUTH: "1", AWS_BEDROCK_FORCE_HTTP1: "1", AWS_REGION: "us-east-1" };

	test("appended turns use Converse content blocks", async () => {
		const h = harness(bedrock(), "server", { env });
		const main = await h.main();
		expect(main.body.messages).toBeArray();
		const side = await h.side();
		expect(h.transcriptReads()).toBe(0);
		const { messages: mainMessages, ...mainRest } = main.body;
		const { messages: sideMessages, ...sideRest } = side.body;
		expect(sideRest).toEqual(mainRest);
		expect(sideMessages.slice(0, mainMessages.length)).toEqual(mainMessages);
		expect(sideMessages.slice(mainMessages.length)).toEqual([
			{ role: "user", content: [{ text: wrapQuestion(QUESTION) }] },
		]);
	});
});

describe("pi-messages", () => {
	test("the captured context and its options are reused", async () => {
		const h = harness(model("pi-messages"), "fetch");
		const main = await h.main();
		const side = await h.side();
		expect(h.transcriptReads()).toBe(0);
		expect(side.body.options).toEqual(main.body.options);
		expect(side.body.options.sessionId).toBe(sessionId);
		const mainMessages = main.body.context.messages;
		const sideMessages = side.body.context.messages;
		expect(sideMessages.slice(0, mainMessages.length)).toEqual(mainMessages);
		expect(sideMessages.slice(mainMessages.length)).toMatchObject([
			{ role: "user", content: [{ type: "text", text: wrapQuestion(QUESTION) }] },
		]);
	});
});

describe("request options from the main loop's wrapper", () => {
	test("headers the wrapper added to the captured request go out with the side question", async () => {
		const h = harness(model("openai-completions"), "fetch");
		notePendingHeaders({ "x-opencode-session": sessionId, "x-opencode-client": "pi", "X-Extension-Route": "blue" });
		await h.main();
		const side = await h.side();
		expect(side.headers.get("x-opencode-session")).toBe(sessionId);
		expect(side.headers.get("x-opencode-client")).toBe("pi");
		expect(side.headers.get("x-extension-route")).toBe("blue");
	});

	test("a configured transport and thinking budgets are passed on", async () => {
		const h = harness(model("openai-completions"), "fetch");
		await h.main();
		await h.side({ transport: "websocket", thinkingBudgets: { low: 1024 } });
		expect(h.sideOptions.at(-1)).toMatchObject({ transport: "websocket", thinkingBudgets: { low: 1024 } });
	});

	test("a Codex side question uses SSE whatever the setting", async () => {
		const b64 = (value: unknown) => Buffer.from(JSON.stringify(value)).toString("base64url");
		const token = `${b64({ alg: "none" })}.${b64({ "https://api.openai.com/auth": { chatgpt_account_id: "acct" } })}.sig`;
		const h = harness(model("openai-codex-responses"), "fetch", { apiKey: token });
		await h.side({ transport: "auto" });
		expect(h.sideOptions.at(-1)?.transport).toBe("sse");
	});
});
