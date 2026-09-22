/**
 * Reading an answer out, and appending a question onto a captured body.
 *
 * The cache_control assertions are the ones worth having. If a breakpoint
 * migrates onto the appended question, every side question pays a full cache
 * write on the shared prefix, and the symptom — a large `cacheWrite` and a zero
 * `cacheRead` — is only visible in usage numbers nobody reads.
 */

import { describe, expect, test } from "bun:test";
import type { AssistantMessage, Message, Usage } from "@earendil-works/pi-ai";
import type { Exchange } from "./history.js";
import { CUT_OFF_TAIL, NO_TOOLS_NOTICE, OMIT_FABRICATED, SIDE_QUESTION_REMINDER } from "./prompt.js";
import {
	appendSideTurns,
	endsWithSystemMessage,
	extractResult,
	historyTurns,
	isMessagesBody,
	type MessagesBody,
	shouldRemember,
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
		expect(isMessagesBody({ messages: [], system: "you are pi" })).toBe(true);
		expect(isMessagesBody({ messages: [], tools: [] })).toBe(true);
	});

	test("rejects a body that carries the conversation under another name", () => {
		expect(isMessagesBody({ input: [], tools: [] })).toBe(false);
		expect(isMessagesBody({ contents: [] })).toBe(false);
	});

	test("rejects a messages array with neither a prompt nor tools beside it", () => {
		expect(isMessagesBody({ messages: [] })).toBe(false);
	});

	test("rejects non-objects", () => {
		expect(isMessagesBody(undefined)).toBe(false);
		expect(isMessagesBody("{}")).toBe(false);
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
