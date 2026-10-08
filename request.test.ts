/**
 * Reading an answer out, and appending a question onto a captured body.
 *
 * The cache_control assertions are the ones worth having. If a breakpoint
 * migrates onto the appended question, every side question pays a full cache
 * write on the shared prefix, and the symptom — a large `cacheWrite` and a zero
 * `cacheRead` — is only visible in usage numbers nobody reads.
 */

import { describe, expect, test } from "bun:test";
import type { AssistantMessage, Message, Model, Usage } from "@earendil-works/pi-ai";
import { SessionManager } from "@earendil-works/pi-coding-agent";
import type { Exchange } from "./history.js";
import { CUT_OFF_TAIL, NO_TOOLS_NOTICE, OMIT_FABRICATED, SIDE_QUESTION_REMINDER, wrapQuestion } from "./prompt.js";
import {
	appendBedrockSideTurns,
	appendGoogleSideTurns,
	appendPiMessagesSideTurns,
	appendSideTurns,
	bodyEndsOnToolResult,
	bodyHasImages,
	endsWithSystemMessage,
	extendCapturedBody,
	extractResult,
	historyTurns,
	isMessagesBody,
	type MessagesBody,
	placeholderContext,
	replayHeaders,
	sessionTranscript,
	shouldRemember,
	sideTransport,
	trimTrailingIncompleteTurn,
} from "./request.js";

const NO_USAGE: Usage = {
	input: 0,
	output: 0,
	cacheRead: 0,
	cacheWrite: 0,
	totalTokens: 0,
	cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
};

function answer(overrides: Partial<AssistantMessage> = {}): AssistantMessage {
	return {
		role: "assistant",
		content: [],
		api: "anthropic-messages",
		provider: "anthropic",
		model: "claude-sonnet-5",
		usage: NO_USAGE,
		stopReason: "stop",
		timestamp: 0,
		...overrides,
	} as AssistantMessage;
}

function text(value: string) {
	return { type: "text" as const, text: value };
}

describe("extractResult", () => {
	test("plain text comes through trimmed and is worth remembering", () => {
		const result = extractResult(answer({ content: [text("  WBC is the white blood cell count.  ")] }));
		expect(result.response).toBe("WBC is the white blood cell count.");
		expect(result.synthetic).toBe(false);
		expect(shouldRemember(result)).toBe(true);
	});

	test("thinking blocks are dropped — they are not the answer", () => {
		const result = extractResult(
			answer({
				content: [{ type: "thinking", thinking: "weighing it up" }, text("Two.")],
			}),
		);
		expect(result.response).toBe("Two.");
	});

	test("a drawn tool call is shown with the notice, and still stored", () => {
		const drawn = 'Let me look:\n<invoke name="Read">';
		const result = extractResult(answer({ content: [text(drawn)] }));
		expect(result.response).toBe(`${drawn}\n\n${NO_TOOLS_NOTICE}`);
		expect(result.synthetic).toBe(false);
		expect(shouldRemember(result)).toBe(true);
	});

	test("a truncated answer is shown with its tail, and still stored", () => {
		const result = extractResult(answer({ content: [text("The three parts are")], stopReason: "length" }));
		expect(result.response).toBe(`The three parts are\n\n${CUT_OFF_TAIL}`);
		expect(shouldRemember(result)).toBe(true);
	});

	test("a real tool call is replaced, not shown, and not stored", () => {
		const result = extractResult(
			answer({
				content: [{ type: "toolCall", id: "t1", name: "read", arguments: {} }],
				stopReason: "toolUse",
			}),
		);
		expect(result.response).toContain("tried to call read");
		expect(result.synthetic).toBe(true);
		expect(shouldRemember(result)).toBe(false);
	});

	test("an API error is reported and not stored", () => {
		const result = extractResult(answer({ stopReason: "error", errorMessage: "overloaded" }));
		expect(result.response).toBe("(API error: overloaded)");
		expect(shouldRemember(result)).toBe(false);
	});

	test("an abort produces no answer and nothing to store", () => {
		const result = extractResult(answer({ content: [text("half an ans")], stopReason: "aborted" }));
		expect(result.aborted).toBe(true);
		expect(result.response).toBe("");
		expect(shouldRemember(result)).toBe(false);
	});

	test("an empty response says so rather than showing a blank panel", () => {
		const result = extractResult(answer({ content: [] }));
		expect(result.response).toBe("No response received");
		expect(shouldRemember(result)).toBe(false);
	});

	test("a rerouted request carries a fallback notice", () => {
		const result = extractResult(answer({ content: [text("hi")], responseModel: "claude-haiku-4-5-20251001" }));
		expect(result.fallbackNotice).toContain("claude-haiku-4-5-20251001");
	});

	test("usage is kept for verification", () => {
		const usage: Usage = { ...NO_USAGE, cacheRead: 12000, cacheWrite: 30 };
		const result = extractResult(answer({ content: [text("hi")], usage }));
		expect(result.usage).toEqual({ cacheRead: 12000, cacheWrite: 30 });
	});
});

describe("historyTurns", () => {
	const stored = (question: string, response: string): Exchange => ({ question, response, createdAt: 0 });

	test("rewrites each stored answer for replay", () => {
		const turns = historyTurns([stored("q1", "plain"), stored("q2", "<function_calls>")]);
		expect(turns).toEqual([
			{ question: "q1", answer: "plain" },
			{ question: "q2", answer: OMIT_FABRICATED },
		]);
	});

	test("drops an exchange with nothing on one side — an empty block is rejected by the API", () => {
		expect(historyTurns([stored("q", "   "), stored("  ", "a")])).toEqual([]);
	});
});

describe("isMessagesBody", () => {
	test("accepts a body whose conversation lives under `messages`", () => {
		expect(isMessagesBody({ model: "claude-sonnet-5", messages: [], system: "you are pi" })).toBe(true);
		expect(isMessagesBody({ model: "glm-5.3", messages: [], tools: [] })).toBe(true);
	});

	/** The system prompt is one of the messages, and `tools` is only written when there are some. */
	test("accepts a completions body without tools", () => {
		const body = { model: "glm-5.3", stream: true, messages: [{ role: "system", content: "you are pi" }] };
		expect(isMessagesBody(body)).toBe(true);
	});

	test("rejects a body that carries the conversation under another name", () => {
		expect(isMessagesBody({ model: "m", input: [], tools: [] })).toBe(false);
		expect(isMessagesBody({ model: "m", contents: [] })).toBe(false);
	});

	test("rejects a stray `messages` field with no model beside it", () => {
		expect(isMessagesBody({ messages: [] })).toBe(false);
	});

	test("rejects a Converse body, whose model id is `modelId`", () => {
		expect(isMessagesBody({ modelId: "anthropic.claude", messages: [], system: [{ text: "you are pi" }] })).toBe(false);
	});

	test("rejects non-objects", () => {
		expect(isMessagesBody(undefined)).toBe(false);
		expect(isMessagesBody("{}")).toBe(false);
	});
});

describe("extendCapturedBody", () => {
	const model = { api: "test", provider: "test", id: "test" } as Model<any>;
	const signal = new AbortController().signal;
	const extend = (api: string, payload: unknown) => extendCapturedBody(api, payload, [], "q", model, signal);

	/** Converse has `messages` and `system` too, but its entries are `{ role, content: [{ text }] }`. */
	test("a Converse body ending on user keeps that message and appends the question after its blocks", () => {
		const cachePoint = { cachePoint: { type: "default" } };
		const body = {
			modelId: "anthropic.claude",
			system: [{ text: "p" }],
			messages: [{ role: "user", content: [{ text: "hi" }, cachePoint] }],
		};
		const after = extend("bedrock-converse-stream", body) as { messages: unknown[] };
		expect(after.messages).toEqual([
			{ role: "user", content: [{ text: "hi" }, cachePoint, { text: wrapQuestion("q") }] },
		]);
		expect(body.messages[0]!.content).toHaveLength(2);
	});

	test("a body is read the way its own API writes it, whatever its field names", () => {
		const converse = { modelId: "anthropic.claude", system: [{ text: "p" }], messages: [] };
		expect(extend("anthropic-messages", converse)).toBeUndefined();
		expect(extend("openai-completions", { model: "m", input: [] })).toBeUndefined();
	});

	test("an API this extension does not know is rebuilt", () => {
		expect(extend("custom-extension-api", { model: "m", messages: [] })).toBeUndefined();
	});

	test("a Messages body ending on a flushed system message is rebuilt", () => {
		const body = { model: "m", system: "p", messages: [{ role: "user", content: "hi" }, { role: "system", content: "update" }] };
		expect(extend("anthropic-messages", body)).toBeUndefined();
	});

	test("the captured body is copied, not appended to in place", () => {
		const body = { model: "m", contents: [{ role: "user", parts: [{ text: "hi" }] }], config: {} };
		extend("google-generative-ai", body);
		expect(body.contents).toHaveLength(1);
		expect(body.config).toEqual({});
	});
});

describe("appendGoogleSideTurns", () => {
	test("replayed answers are `model` turns and the request carries the given signal", () => {
		const signal = new AbortController().signal;
		const body = { model: "gemini", contents: [{ role: "user", parts: [{ text: "hi" }] }], config: { systemInstruction: "p" } };
		const after = appendGoogleSideTurns(body, [{ question: "q1", answer: "a1" }], "q2", signal);
		expect(after.contents.slice(1)).toEqual([
			{ role: "user", parts: [{ text: "q1" }] },
			{ role: "model", parts: [{ text: "a1" }] },
			{ role: "user", parts: [{ text: wrapQuestion("q2") }] },
		]);
		expect(after.config).toEqual({ systemInstruction: "p", abortSignal: signal });
	});
});

describe("appendPiMessagesSideTurns", () => {
	test("appended turns are Pi messages, and the options stay as captured", () => {
		const model = { api: "pi-messages", provider: "pi", id: "m" } as Model<any>;
		const body = {
			model: "m",
			context: { messages: [{ role: "user", content: "hi", timestamp: 0 }] },
			options: { sessionId: "s1", maxTokens: 4096 },
		};
		const after = appendPiMessagesSideTurns(body, [{ question: "q1", answer: "a1" }], "q2", model);
		expect(after.options).toEqual(body.options);
		expect(after.context.messages.slice(1)).toMatchObject([
			{ role: "user", content: [{ type: "text", text: "q1" }] },
			{ role: "assistant", content: [{ type: "text", text: "a1" }], api: "pi-messages", provider: "pi", model: "m" },
			{ role: "user", content: [{ type: "text", text: wrapQuestion("q2") }] },
		]);
	});
});

describe("appendBedrockSideTurns", () => {
	test("history after a trailing user message still alternates", () => {
		const body = { modelId: "m", messages: [{ role: "user", content: [{ text: "hi" }] }] };
		const after = appendBedrockSideTurns(body, [{ question: "q1", answer: "a1" }], "q2");
		expect(after.messages).toEqual([
			{ role: "user", content: [{ text: "hi" }, { text: "q1" }] },
			{ role: "assistant", content: [{ text: "a1" }] },
			{ role: "user", content: [{ text: wrapQuestion("q2") }] },
		]);
		expect(body.messages).toEqual([{ role: "user", content: [{ text: "hi" }] }]);
	});

	test("a body already ending on assistant gets a new user message", () => {
		const body = {
			modelId: "m",
			messages: [
				{ role: "user", content: [{ text: "hi" }] },
				{ role: "assistant", content: [{ text: "ok" }] },
			],
		};
		const after = appendBedrockSideTurns(body, [], "q");
		expect(after.messages.slice(0, 2)).toEqual(body.messages);
		expect(after.messages[2]).toEqual({ role: "user", content: [{ text: wrapQuestion("q") }] });
	});
});

describe("placeholderContext", () => {
	test("carries an image only when asked to, so Copilot sends its vision header", () => {
		expect(placeholderContext("q").messages[0]).toMatchObject({ content: [{ type: "text", text: "q" }] });
		const content = (placeholderContext("q", { withImage: true }).messages[0] as { content: { type: string }[] }).content;
		expect(content.map((block) => block.type)).toEqual(["text", "image"]);
	});

	test("an agent-initiated placeholder does not end on a user message", () => {
		const messages = placeholderContext("q", { withImage: true, agentInitiated: true }).messages;
		expect(messages[0]).toMatchObject({ role: "user" });
		expect(messages.at(-1)?.role).toBe("toolResult");
	});

	test("a tool result is recognized in each wire shape Copilot sends", () => {
		expect(bodyEndsOnToolResult({ messages: [{ role: "user", content: [{ type: "tool_result", tool_use_id: "t" }] }] })).toBe(true);
		expect(bodyEndsOnToolResult({ messages: [{ role: "tool", tool_call_id: "t", content: "ok" }] })).toBe(true);
		expect(bodyEndsOnToolResult({ input: [{ type: "function_call_output", call_id: "t", output: "ok" }] })).toBe(true);
		expect(bodyEndsOnToolResult({ messages: [{ role: "user", content: [{ type: "text", text: "hi" }] }] })).toBe(false);
	});

	test("an image anywhere in a captured body is found", () => {
		expect(bodyHasImages({ messages: [{ role: "user", content: [{ type: "image", source: {} }] }] })).toBe(true);
		expect(bodyHasImages({ messages: [{ role: "user", content: [{ type: "image_url", image_url: {} }] }] })).toBe(true);
		expect(bodyHasImages({ input: [{ role: "user", content: [{ type: "input_image", image_url: "" }] }] })).toBe(true);
		expect(bodyHasImages({ messages: [{ role: "user", content: [{ type: "text", text: "image" }] }] })).toBe(false);
	});
});

describe("replayHeaders", () => {
	test("headers the main loop's wrapper added are replayed", () => {
		const captured = { "x-opencode-session": "s1", "X-Route": "blue" };
		expect(replayHeaders(captured, {})).toEqual(captured);
	});

	/** An OAuth token refreshed since the captured request must not go out stale. */
	test("credentials resolved now win, whatever the case of the name", () => {
		const captured = { Authorization: "Bearer old", "x-route": "blue" };
		expect(replayHeaders(captured, { authorization: "Bearer new" })).toEqual({
			"x-route": "blue",
			authorization: "Bearer new",
		});
	});
});

describe("sideTransport", () => {
	const api = (name: string) => ({ api: name }) as Model<any>;

	test("Codex always goes over SSE, leaving the main loop's WebSocket to it", () => {
		expect(sideTransport(api("openai-codex-responses"), "auto")).toBe("sse");
		expect(sideTransport(api("openai-codex-responses"), undefined)).toBe("sse");
	});

	test("anything else gets the configured transport", () => {
		expect(sideTransport(api("anthropic-messages"), "websocket")).toBe("websocket");
		expect(sideTransport(api("anthropic-messages"), undefined)).toBeUndefined();
	});
});

describe("endsWithSystemMessage", () => {
	/**
	 * Pi's Claude models that accept mid-conversation system messages let the
	 * adapter flush a prompt-section update at the end of the body. Appending a
	 * user turn after one is a shape the adapter itself never emits, so such a
	 * snapshot is not reused.
	 */
	test("spots the flushed prompt update at the end of a body", () => {
		const body = {
			system: "base",
			messages: [
				{ role: "user", content: [{ type: "text", text: "fix the parser" }] },
				{ role: "assistant", content: [{ type: "text", text: "ok" }] },
				{ role: "user", content: [{ type: "text", text: "now the lexer" }] },
				{ role: "system", content: [{ type: "text", text: 'Updated system prompt section "preamble"' }] },
			],
		};
		expect(endsWithSystemMessage(body)).toBe(true);
	});

	test("leaves an ordinary body alone", () => {
		expect(endsWithSystemMessage({ system: "base", messages: [{ role: "user", content: "hi" }] })).toBe(false);
		expect(endsWithSystemMessage({ system: "base", messages: [] })).toBe(false);
	});
});

describe("appendSideTurns", () => {
	/** A captured body with the breakpoint where the provider put it: on the last prefix block. */
	function captured(): MessagesBody {
		return {
			model: "claude-sonnet-5",
			max_tokens: 32000,
			thinking: { type: "enabled", budget_tokens: 10000 },
			system: [{ type: "text", text: "you are pi", cache_control: { type: "ephemeral" } }],
			tools: [{ name: "read", description: "read a file", input_schema: {} }],
			messages: [
				{ role: "user", content: [{ type: "text", text: "fix the parser" }] },
				{ role: "assistant", content: [{ type: "tool_use", id: "t1", name: "read", input: {} }] },
				{
					role: "user",
					content: [{ type: "tool_result", tool_use_id: "t1", content: "ok", cache_control: { type: "ephemeral" } }],
				},
			],
		};
	}

	function cacheControlPositions(body: MessagesBody): number[] {
		return body.messages.flatMap((message, index) => {
			const content = (message as { content?: unknown }).content;
			if (!Array.isArray(content)) return [];
			return content.some((block) => (block as Record<string, unknown>).cache_control) ? [index] : [];
		});
	}

	test("the breakpoint stays on the block the main request put it on", () => {
		const before = captured();
		const after = appendSideTurns(captured(), [], "what is WBC");
		expect(cacheControlPositions(before)).toEqual([2]);
		expect(cacheControlPositions(after)).toEqual([2]);
	});

	test("nothing appended carries a cache_control of its own", () => {
		const turns = [{ question: "earlier", answer: "earlier answer" }];
		const after = appendSideTurns(captured(), turns, "what is WBC");
		for (const message of after.messages.slice(3)) {
			expect(JSON.stringify(message)).not.toContain("cache_control");
		}
	});

	test("the earlier exchanges land after the prefix, then the wrapped question", () => {
		const turns = [
			{ question: "q1", answer: "a1" },
			{ question: "q2", answer: "a2" },
		];
		const after = appendSideTurns(captured(), turns, "q3");
		expect(after.messages.slice(3)).toEqual([
			{ role: "user", content: "q1" },
			{ role: "assistant", content: "a1" },
			{ role: "user", content: "q2" },
			{ role: "assistant", content: "a2" },
			{ role: "user", content: `${SIDE_QUESTION_REMINDER}\n\nq3` },
		]);
	});

	test("only the live question is wrapped in the reminder", () => {
		const after = appendSideTurns(captured(), [{ question: "q1", answer: "a1" }], "q2");
		const wrapped = after.messages.filter((m) => JSON.stringify(m).includes("system-reminder"));
		expect(wrapped).toHaveLength(1);
	});

	test("model, tools, thinking and the output cap are left alone", () => {
		const after = appendSideTurns(captured(), [], "q");
		const before = captured();
		expect(after.model).toEqual(before.model);
		expect(after.tools).toEqual(before.tools);
		expect(after.thinking).toEqual(before.thinking);
		expect(after.max_tokens).toEqual(before.max_tokens);
		expect(after.system).toEqual(before.system);
	});

	test("the captured body is not mutated, so the snapshot survives a second question", () => {
		const body = captured();
		appendSideTurns(body, [], "q");
		expect(body.messages).toHaveLength(3);
	});
});

describe("trimTrailingIncompleteTurn", () => {
	const now = 0;
	const user = (t: string): Message => ({ role: "user", content: [{ type: "text", text: t }], timestamp: now });
	const assistant = (content: AssistantMessage["content"], stopReason: AssistantMessage["stopReason"]): Message =>
		answer({ content, stopReason });
	const toolResult = (id: string): Message => ({
		role: "toolResult",
		toolCallId: id,
		toolName: "read",
		content: [{ type: "text", text: "ok" }],
		isError: false,
		timestamp: now,
	});
	const call = (id: string) => ({ type: "toolCall" as const, id, name: "read", arguments: {} });

	test("a settled conversation is left alone", () => {
		const messages = [user("hi"), assistant([{ type: "text", text: "hello" }], "stop")];
		expect(trimTrailingIncompleteTurn(messages)).toEqual(messages);
	});

	test("an assistant message still being generated is dropped", () => {
		const messages = [user("hi"), assistant([{ type: "text", text: "hel" }], "pending")];
		expect(trimTrailingIncompleteTurn(messages)).toEqual([user("hi")]);
	});

	test("a tool call whose result has come back is a legal prefix and stays", () => {
		const messages = [user("hi"), assistant([call("t1")], "toolUse"), toolResult("t1")];
		expect(trimTrailingIncompleteTurn(messages)).toEqual(messages);
	});

	test("a half-answered tool batch is dropped along with its partial results", () => {
		const messages = [user("hi"), assistant([call("t1"), call("t2")], "toolUse"), toolResult("t1")];
		expect(trimTrailingIncompleteTurn(messages)).toEqual([user("hi")]);
	});

	test("no placeholder result is invented for the missing one", () => {
		const messages = [user("hi"), assistant([call("t1"), call("t2")], "toolUse"), toolResult("t1")];
		expect(JSON.stringify(trimTrailingIncompleteTurn(messages))).not.toContain("Tool result missing");
	});

	test("an empty conversation trims to nothing without throwing", () => {
		expect(trimTrailingIncompleteTurn([])).toEqual([]);
	});
});

describe("sessionTranscript", () => {
	const user = (t: string): Message => ({ role: "user", content: [text(t)], timestamp: 0 });

	test("a message omitted by a context edit stays out, as it does for the main loop", () => {
		// Length recovery omits the truncated reply. Unlike an errored attempt,
		// no adapter filters a `length` reply on its own, so the edit is all that
		// keeps it out of the request.
		const session = SessionManager.inMemory("/tmp");
		session.appendMessage(user("summarize the repo"));
		const truncated = session.appendMessage(answer({ content: [text("The repo has thr")], stopReason: "length" }));
		session.appendContextEdit(truncated, null);

		expect(sessionTranscript(session)).toEqual([user("summarize the repo")]);
	});

	test("a message replaced by a context edit goes out with the replacement", () => {
		const session = SessionManager.inMemory("/tmp");
		session.appendMessage(user("read it"));
		session.appendMessage(
			answer({ content: [{ type: "toolCall", id: "t1", name: "read", arguments: {} }], stopReason: "toolUse" }),
		);
		const result = session.appendMessage({
			role: "toolResult",
			toolCallId: "t1",
			toolName: "read",
			content: [text("FULL FILE CONTENTS")],
			isError: false,
			timestamp: 0,
		});
		session.appendMessage(answer({ content: [text("done")] }));
		session.appendContextEdit(result, { content: "[pruned]" });

		const transcript = sessionTranscript(session);
		expect(transcript.find((m) => m.role === "toolResult")?.content).toEqual([text("[pruned]")]);
		expect(JSON.stringify(transcript)).not.toContain("FULL FILE CONTENTS");
	});

	test("a pi without the projection still gets its transcript from the entries", () => {
		const session = SessionManager.inMemory("/tmp");
		session.appendMessage(user("hi"));
		session.appendMessage(answer({ content: [text("hello")] }));
		const older = { buildContextEntries: () => session.buildContextEntries() } as unknown as SessionManager;

		expect(sessionTranscript(older)).toEqual(sessionTranscript(session));
	});
});
