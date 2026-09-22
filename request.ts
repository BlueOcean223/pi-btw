/**
 * Building one side-question request, and reading one answer back out.
 *
 * Two ways in. The first rewrites the snapshot of the main loop's last request
 * body: append the earlier exchanges and the new question to `messages`, change
 * nothing else, and hand the result to the provider through `onPayload`. That
 * is what keeps the prefix byte-identical and the cache warm. The second, used
 * when there is no usable snapshot, rebuilds the context from the session
 * transcript and accepts a cache miss.
 *
 * The tool schemas stay in the request in both paths. Dropping them would save
 * tokens and lose the cache: tools sit in the prefix, so a request without them
 * shares no prefix with the main loop. The side question cannot call them
 * anyway — `maxTurns` is effectively one here, because nothing feeds a tool
 * result back, and a `toolCall` that comes back is reported rather than run.
 */

import type { ExtensionAPI, ExtensionCommandContext } from "@earendil-works/pi-coding-agent";
import { convertToLlm, sessionEntryToContextMessages } from "@earendil-works/pi-coding-agent";
import type { AssistantMessage, Context, Message, Model, Tool } from "@earendil-works/pi-ai";
import type { Exchange } from "./history.js";
import {
	CUT_OFF_TAIL,
	FABRICATED_TOOL_CALL,
	NO_TOOLS_NOTICE,
	replayAssistant,
	wrapQuestion,
} from "./prompt.js";
import { getSnapshot, markOwnPayload } from "./snapshot.js";

export interface SideResult {
	/** The answer as it should be shown, including any appended notice. */
	response: string;
	/** True when the text is the extension's own explanation, not a model answer. Not stored. */
	synthetic: boolean;
	aborted: boolean;
	fallbackNotice?: string;
	/** Kept for verification — a warm second question reads, a cold one writes. Not shown. */
	usage?: { cacheRead: number; cacheWrite: number };
}

/** Whether a result is worth replaying into the next side question. */
export function shouldRemember(result: SideResult): boolean {
	return !result.aborted && !result.synthetic && result.response.trim().length > 0;
}

// ---------- Reading the answer ----------

/**
 * Turn the final assistant message into what the panel shows.
 *
 * A real `toolCall` and a tool call drawn in prose are two different failures
 * and get two different treatments. The protocol block is replaced outright:
 * there is no answer in it and nothing to show. The prose is shown as written,
 * because the answer may be in there, with a line saying none of it ran.
 */
export function extractResult(message: AssistantMessage): SideResult {
	const usage = message.usage
		? { cacheRead: message.usage.cacheRead, cacheWrite: message.usage.cacheWrite }
		: undefined;
	// The provider only sets `responseModel` when it answered with something
	// other than what was asked for.
	const fallbackNotice = message.responseModel
		? `Answered by ${message.responseModel} instead of ${message.model}.`
		: undefined;

	if (message.stopReason === "aborted") {
		return { response: "", synthetic: false, aborted: true, usage };
	}

	const text = message.content
		.filter((block): block is { type: "text"; text: string } => block.type === "text")
		.map((block) => block.text)
		.join("\n\n")
		.trim();

	if (text) {
		// The notice is stored with the answer, so the replay rewrite finds it
		// next time by the same test that produced it.
		if (FABRICATED_TOOL_CALL.test(text)) {
			return { response: `${text}\n\n${NO_TOOLS_NOTICE}`, synthetic: false, aborted: false, fallbackNotice, usage };
		}
		if (message.stopReason === "length") {
			return { response: `${text}\n\n${CUT_OFF_TAIL}`, synthetic: false, aborted: false, fallbackNotice, usage };
		}
		return { response: text, synthetic: false, aborted: false, fallbackNotice, usage };
	}

	const toolCall = message.content.find((block) => block.type === "toolCall");
	if (toolCall) {
		return {
			response: `(The model tried to call ${toolCall.name} instead of answering directly. Try rephrasing or ask in the main conversation.)`,
			synthetic: true,
			aborted: false,
			usage,
		};
	}

	if (message.stopReason === "error" || message.errorMessage) {
		return {
			response: `(API error: ${message.errorMessage ?? "unknown error"})`,
			synthetic: true,
			aborted: false,
			usage,
		};
	}

	return { response: "No response received", synthetic: true, aborted: false, usage };
}

// ---------- Building the request ----------

/** One replayed exchange, as a user/assistant pair. */
export interface ReplayTurn {
	question: string;
	answer: string;
}

/**
 * The earlier exchanges, rewritten for replay.
 *
 * They go after the shared prefix, never inside it, so the prefix bytes the
 * main loop sent stay exactly where they were. Earlier side questions can see
 * each other; none of them can see what the main loop did after this snapshot
 * was taken.
 */
export function historyTurns(history: readonly Exchange[]): ReplayTurn[] {
	const turns: ReplayTurn[] = [];
	for (const exchange of history) {
		const question = exchange.question.trim();
		const answer = replayAssistant(exchange).trim();
		// An empty block is rejected by the Anthropic API and carries nothing.
		if (!question || !answer) continue;
		turns.push({ question, answer });
	}
	return turns;
}

/** A provider body this extension knows how to append role/content messages to. */
export interface MessagesBody {
	messages: unknown[];
	[key: string]: unknown;
}

/**
 * Whether a captured body is one whose `messages` array takes plain
 * `{ role, content }` entries — Anthropic Messages and the OpenAI completions
 * shape both do. Bodies that carry the conversation under another name
 * (`input`, `contents`) fall through to the rebuild path.
 */
export function isMessagesBody(payload: unknown): payload is MessagesBody {
	if (!payload || typeof payload !== "object") return false;
	const body = payload as Record<string, unknown>;
	if (!Array.isArray(body.messages)) return false;
	return body.system !== undefined || Array.isArray(body.tools);
}

/**
 * Whether a captured body ends on a mid-conversation system message.
 *
 * Four of Pi's Claude models declare `supportsMidConvoSystemMessages`, and on
 * those the adapter holds a prompt-section update back until the next
 * assistant message — or, when none follows, flushes it at the very end. The
 * body then closes with a system message carrying the update, with the cache
 * breakpoint on it.
 *
 * Appending a user message there is a shape the adapter itself never emits: it
 * takes care to place a system message before an assistant turn or last, never
 * directly before a user turn. Rather than fabricate an assistant turn to make
 * room, such a body is left alone and the question goes through the rebuild
 * path, which lands the appended messages ahead of that flush and comes out
 * legal. The cost is one uncached side question, until the main loop's next
 * request replaces the snapshot.
 */
export function endsWithSystemMessage(body: MessagesBody): boolean {
	const last = body.messages[body.messages.length - 1];
	if (!last || typeof last !== "object") return false;
	return (last as { role?: unknown }).role === "system";
}

/**
 * Append the replayed exchanges and the wrapped question to a captured body.
 *
 * Nothing else is touched: not `model`, `system`, `tools`, `thinking`,
 * `max_tokens`, or `metadata`. Shrinking the output cap or the thinking budget
 * would change the prefix on some models and cost the whole cache.
 *
 * The appended messages carry no `cache_control`, and the breakpoint already
 * sitting on the shared prefix is left where it is. Moving it onto the
 * question would buy nothing and cost a cache write: the read still hits from
 * the prefix, and a one-shot suffix nobody will ever continue from has no
 * business becoming a cache entry of its own.
 */
export function appendSideTurns(body: MessagesBody, turns: readonly ReplayTurn[], question: string): MessagesBody {
	const messages = [...body.messages];
	for (const turn of turns) {
		messages.push({ role: "user", content: turn.question });
		messages.push({ role: "assistant", content: turn.answer });
	}
	messages.push({ role: "user", content: wrapQuestion(question) });
	return { ...body, messages };
}

/**
 * Drop a trailing turn that is not a legal prefix yet.
 *
 * Two shapes qualify: an assistant message still being generated, and an
 * assistant message whose tool calls have not all come back. Both are dropped
 * whole, along with the partial results behind them.
 *
 * No placeholder result is substituted for the missing ones. Claude Code
 * inserts "[Tool result missing due to internal error]" so its fork matches the
 * bytes the main thread would have patched in; pi's main loop never writes such
 * a line, so inserting one here would match nothing and would hand the side
 * question a fabricated tool result to reason from.
 */
export function trimTrailingIncompleteTurn(messages: readonly Message[]): Message[] {
	let index = messages.length - 1;
	while (index >= 0 && messages[index]!.role === "toolResult") index--;

	const last = messages[index];
	if (!last || last.role !== "assistant") return [...messages];

	const answered = new Set<string>();
	for (const message of messages.slice(index + 1)) {
		if (message.role === "toolResult") answered.add(message.toolCallId);
	}
	const unpaired = last.content.some((block) => block.type === "toolCall" && !answered.has(block.id));

	if (last.stopReason === "pending" || unpaired) return messages.slice(0, index);
	return [...messages];
}

function replayMessages(turns: readonly ReplayTurn[], question: string, model: Model<any>): Message[] {
	const now = Date.now();
	const messages: Message[] = [];
	for (const turn of turns) {
		messages.push({ role: "user", content: [{ type: "text", text: turn.question }], timestamp: now });
		messages.push({
			role: "assistant",
			content: [{ type: "text", text: turn.answer }],
			api: model.api,
			provider: model.provider,
			model: model.id,
			usage: {
				input: 0,
				output: 0,
				cacheRead: 0,
				cacheWrite: 0,
				totalTokens: 0,
				cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
			},
			stopReason: "stop",
			timestamp: now,
		});
	}
	messages.push({ role: "user", content: [{ type: "text", text: wrapQuestion(question) }], timestamp: now });
	return messages;
}

/**
 * Rebuild the request from the session transcript.
 *
 * `buildContextEntries()` already resolves compaction and branch summaries, so
 * this is what the model is supposed to see — there is no compaction boundary
 * left to cut at.
 *
 * pi records the system prompt and the tool declarations as system messages in
 * the transcript, so when the transcript has them the context is handed over
 * as-is: those are the same bytes the main loop sends, which is better for the
 * cache than re-rendering the prompt. Sessions written by older pi versions
 * have no such message, and only those fall back to the live prompt and the
 * active tool list.
 */
function rebuildContext(
	ctx: ExtensionCommandContext,
	pi: ExtensionAPI,
	turns: readonly ReplayTurn[],
	question: string,
	model: Model<any>,
): Context {
	const entries = ctx.sessionManager.buildContextEntries();
	const transcript = trimTrailingIncompleteTurn(convertToLlm(entries.flatMap(sessionEntryToContextMessages)));
	const messages = [...transcript, ...replayMessages(turns, question, model)];

	if (transcript[0]?.role === "system") return { messages };

	const schemas = new Map(pi.getAllTools().map((tool) => [tool.name, tool]));
	const tools: Tool[] = [];
	for (const name of pi.getActiveTools()) {
		const info = schemas.get(name);
		if (info?.parameters) tools.push({ name: info.name, description: info.description, parameters: info.parameters });
	}
	return { systemPrompt: ctx.getSystemPrompt(), messages, tools };
}

/** A throwaway context for the snapshot path: `onPayload` discards the body built from it. */
function placeholderContext(question: string): Context {
	return { messages: [{ role: "user", content: [{ type: "text", text: question }], timestamp: Date.now() }] };
}

export interface SideRequestOptions {
	ctx: ExtensionCommandContext;
	pi: ExtensionAPI;
	sessionId: string;
	question: string;
	/** The exchanges as of the moment this question was asked. */
	history: readonly Exchange[];
	signal: AbortSignal;
	/** Called with the full answer text each time it grows. */
	onText: (text: string) => void;
}

/** The message shown when there is no model to ask. Panel chrome, so Chinese. */
export const NO_MODEL = "（当前没有选中模型，旁问没有发出。）";

export async function runSideQuestion(options: SideRequestOptions): Promise<SideResult> {
	const { ctx, pi, sessionId, question, history, signal, onText } = options;

	const model = ctx.model;
	if (!model) return { response: NO_MODEL, synthetic: true, aborted: false };

	try {
		const turns = historyTurns(history);
		const snapshot = getSnapshot(sessionId, model);

		let context: Context;
		let payload: unknown;
		const reusable = snapshot && isMessagesBody(snapshot.payload) && !endsWithSystemMessage(snapshot.payload);
		if (reusable) {
			payload = appendSideTurns(structuredClone(snapshot.payload) as MessagesBody, turns, question);
			markOwnPayload(payload);
			context = placeholderContext(question);
		} else {
			context = rebuildContext(ctx, pi, turns, question, model);
		}

		// The adapter validates the thinking level while building the body it is
		// about to be handed — including the one `onPayload` throws away — so it
		// has to be a level this model accepts.
		const level = ctx.thinkingLevel ?? pi.getThinkingLevel();

		// Auth failures surface synchronously here on some providers, which is
		// why the request is built inside the same guard that reads it.
		const stream = ctx.modelRegistry.streamSimple(model, context, {
			signal,
			...(level === "off" ? {} : { reasoning: level }),
			...(payload === undefined ? {} : { onPayload: () => payload }),
		});

		let text = "";
		for await (const event of stream) {
			if (event.type === "text_delta") {
				text += event.delta;
				onText(text);
			} else if (event.type === "text_end") {
				// `text_end` is authoritative; deltas can have been coalesced.
				text = event.partial.content
					.filter((block): block is { type: "text"; text: string } => block.type === "text")
					.map((block) => block.text)
					.join("\n\n");
				onText(text);
			}
		}
		return extractResult(await stream.result());
	} catch (error) {
		if (signal.aborted) return { response: "", synthetic: false, aborted: true };
		return {
			response: `(API error: ${error instanceof Error ? error.message : String(error)})`,
			synthetic: true,
			aborted: false,
		};
	}
}
