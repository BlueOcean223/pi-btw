/**
 * Building one side-question request, and reading one answer back out.
 *
 * Two ways in. The first rewrites the snapshot of the main loop's last request
 * body: append the earlier exchanges and the new question wherever that API
 * keeps the conversation, then hand the result to the provider through
 * `onPayload`. That is what keeps the prefix byte-identical and the cache warm.
 * The second, used when there is no usable snapshot, rebuilds the context from
 * the session transcript and accepts a cache miss.
 *
 * The tool schemas stay in the request in both paths. Dropping them would save
 * tokens and lose the cache: tools sit in the prefix, so a request without them
 * shares no prefix with the main loop. The side question cannot call them
 * anyway — `maxTurns` is effectively one here, because nothing feeds a tool
 * result back, and a `toolCall` that comes back is reported rather than run.
 */

import type { ExtensionAPI, ExtensionCommandContext } from "@earendil-works/pi-coding-agent";
import { convertToLlm, sessionEntryToContextMessages } from "@earendil-works/pi-coding-agent";
import type {
	AssistantMessage,
	Context,
	ImageContent,
	Message,
	Model,
	ProviderHeaders,
	SimpleStreamOptions,
	TextContent,
	Tool,
} from "@earendil-works/pi-ai";
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
 * Whether a captured body has a model id and a `messages` array beside it.
 *
 * Which API the body belongs to is settled before this runs, by the model the
 * snapshot was taken under; this only confirms the body has the shape that API
 * sends. A completions request without tools has nothing beside `messages` but
 * the model and its options, its system prompt being one of the messages, so
 * the test cannot ask for a top-level `system` or `tools`.
 */
export function isMessagesBody(payload: unknown): payload is MessagesBody {
	if (!payload || typeof payload !== "object") return false;
	const body = payload as Record<string, unknown>;
	return typeof body.model === "string" && Array.isArray(body.messages);
}

/** An OpenAI Responses body, including its opaque reasoning and tool items. */
export interface ResponsesBody {
	model: string;
	input: unknown[];
	[key: string]: unknown;
}

export function isResponsesBody(payload: unknown): payload is ResponsesBody {
	if (!payload || typeof payload !== "object") return false;
	const body = payload as Record<string, unknown>;
	return typeof body.model === "string" && Array.isArray(body.input);
}

/** Append Responses items without reserializing the captured prefix or its options. */
export function appendResponsesSideTurns(
	body: ResponsesBody,
	turns: readonly ReplayTurn[],
	question: string,
): ResponsesBody {
	const input = [...body.input];
	const user = (text: string) => ({ role: "user", content: [{ type: "input_text", text }] });
	for (const turn of turns) {
		input.push(user(turn.question));
		// Stored answers have text, not provider output IDs. Responses accepts
		// this plain assistant input form without inventing an output item ID.
		input.push({ role: "assistant", content: turn.answer });
	}
	input.push(user(wrapQuestion(question)));
	return { ...body, input };
}

/** A Google Gemini or Vertex body, as Pi's Google adapters hand it to the SDK. */
export interface GoogleBody {
	model: string;
	contents: unknown[];
	config?: Record<string, unknown>;
	[key: string]: unknown;
}

export function isGoogleBody(payload: unknown): payload is GoogleBody {
	if (!payload || typeof payload !== "object") return false;
	const body = payload as Record<string, unknown>;
	if (body.config !== undefined && (!body.config || typeof body.config !== "object")) return false;
	return typeof body.model === "string" && Array.isArray(body.contents);
}

/**
 * Append Gemini contents, and give the request its own abort signal.
 *
 * The SDK takes the signal from `config.abortSignal`, inside the body. The
 * snapshot was stored without the main loop's, and putting none back would
 * leave the side question impossible to cancel.
 */
export function appendGoogleSideTurns(
	body: GoogleBody,
	turns: readonly ReplayTurn[],
	question: string,
	signal: AbortSignal,
): GoogleBody {
	const contents = [...body.contents];
	for (const turn of turns) {
		contents.push({ role: "user", parts: [{ text: turn.question }] });
		contents.push({ role: "model", parts: [{ text: turn.answer }] });
	}
	contents.push({ role: "user", parts: [{ text: wrapQuestion(question) }] });
	return { ...body, contents, config: { ...body.config, abortSignal: signal } };
}

/** A Bedrock Converse command input. */
export interface BedrockBody {
	modelId: string;
	messages: unknown[];
	[key: string]: unknown;
}

/**
 * Whether a captured body is a Converse command input.
 *
 * Converse also has `messages` and `system`, but each message's content is a
 * list of `{ text }` blocks, and the model id is `modelId`. A `{ role,
 * content: string }` entry is not something Converse accepts.
 */
export function isBedrockBody(payload: unknown): payload is BedrockBody {
	if (!payload || typeof payload !== "object") return false;
	const body = payload as Record<string, unknown>;
	return typeof body.modelId === "string" && Array.isArray(body.messages);
}

/**
 * Append Converse messages without leaving two user turns in a row.
 *
 * A request the main loop sends ends on a user turn: the prompt, or tool
 * results, which the adapter has already folded into a user message. Converse
 * rejects the next message when it is also user. The question text is therefore
 * added as another content block on that message, after the blocks already
 * there, including the cache point. The checkpoint stays where the adapter put
 * it, and the new text sits past it, which is the same arrangement Converse uses
 * to cache a prefix and then ask something about it. Later history then
 * alternates assistant and user as new messages.
 */
export function appendBedrockSideTurns(body: BedrockBody, turns: readonly ReplayTurn[], question: string): BedrockBody {
	const messages = [...body.messages];
	const pieces: { role: "user" | "assistant"; text: string }[] = [];
	for (const turn of turns) {
		pieces.push({ role: "user", text: turn.question });
		pieces.push({ role: "assistant", text: turn.answer });
	}
	pieces.push({ role: "user", text: wrapQuestion(question) });

	const last = messages[messages.length - 1];
	let start = 0;
	if (isConverseUserMessage(last) && pieces[0]?.role === "user") {
		messages[messages.length - 1] = { ...last, content: [...last.content, { text: pieces[0].text }] };
		start = 1;
	}
	for (const piece of pieces.slice(start)) {
		messages.push({ role: piece.role, content: [{ text: piece.text }] });
	}
	return { ...body, messages };
}

function isConverseUserMessage(message: unknown): message is { role: "user"; content: unknown[] } {
	if (!message || typeof message !== "object") return false;
	const record = message as { role?: unknown; content?: unknown };
	return record.role === "user" && Array.isArray(record.content);
}

/** A pi-messages body: Pi's own context, sent as it is, with the request options beside it. */
export interface PiMessagesBody {
	model: string;
	context: { messages: unknown[]; [key: string]: unknown };
	[key: string]: unknown;
}

export function isPiMessagesBody(payload: unknown): payload is PiMessagesBody {
	if (!payload || typeof payload !== "object") return false;
	const body = payload as Record<string, unknown>;
	const context = body.context as Record<string, unknown> | undefined;
	return typeof body.model === "string" && !!context && typeof context === "object" && Array.isArray(context.messages);
}

/**
 * Append Pi messages to the captured context.
 *
 * The server converts this context for whichever provider sits behind it, so
 * the appended turns are Pi messages, exactly as the rebuild path would add
 * them. `options`, which carries the session id, is left as the main loop sent
 * it.
 */
export function appendPiMessagesSideTurns(
	body: PiMessagesBody,
	turns: readonly ReplayTurn[],
	question: string,
	model: Model<any>,
): PiMessagesBody {
	const messages = [...body.context.messages, ...replayMessages(turns, question, model)];
	return { ...body, context: { ...body.context, messages } };
}

/**
 * The captured body with the side turns appended, or undefined when it cannot
 * be reused and the context has to be rebuilt.
 *
 * The API decides where the conversation lives, not the field names: Bedrock
 * also has `messages` and `system`, with entries of another shape. An API not
 * listed here, such as one an extension registered, is rebuilt.
 */
export function extendCapturedBody(
	api: string,
	payload: unknown,
	turns: readonly ReplayTurn[],
	question: string,
	model: Model<any>,
	signal: AbortSignal,
): unknown {
	switch (api) {
		case "anthropic-messages":
		case "openai-completions":
		case "mistral-conversations":
			if (!isMessagesBody(payload) || endsWithSystemMessage(payload)) return undefined;
			return appendSideTurns(structuredClone(payload), turns, question);
		case "openai-responses":
		case "azure-openai-responses":
		case "openai-codex-responses":
			if (!isResponsesBody(payload)) return undefined;
			return appendResponsesSideTurns(structuredClone(payload), turns, question);
		case "google-generative-ai":
		case "google-vertex":
			if (!isGoogleBody(payload)) return undefined;
			return appendGoogleSideTurns(structuredClone(payload), turns, question, signal);
		case "bedrock-converse-stream":
			if (!isBedrockBody(payload)) return undefined;
			return appendBedrockSideTurns(structuredClone(payload), turns, question);
		case "pi-messages":
			if (!isPiMessagesBody(payload)) return undefined;
			return appendPiMessagesSideTurns(structuredClone(payload), turns, question, model);
		default:
			return undefined;
	}
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
 * The session transcript as the main loop would send it now.
 *
 * This reads pi's own projection instead of converting the context entries one
 * by one. A `context_edit` entry omits or replaces an earlier message in model
 * context while its raw text stays in the session: length recovery writes them,
 * and so can extensions at a turn boundary. Converted entry by entry, the edit
 * contributes nothing and the original message goes out as it was. The
 * projection applies the edits and also resolves compaction and branch
 * summaries, so there is no compaction boundary left to cut at.
 *
 * The peer range is open, and a pi without the projection also has no context
 * edits, so there the entries are still converted directly.
 */
export function sessionTranscript(sessionManager: ExtensionCommandContext["sessionManager"]): Message[] {
	const messages =
		sessionManager.buildSessionProjection?.().messages ??
		sessionManager.buildContextEntries().flatMap(sessionEntryToContextMessages);
	return trimTrailingIncompleteTurn(convertToLlm(messages));
}

/**
 * Rebuild the request from the session transcript.
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
	const transcript = sessionTranscript(ctx.sessionManager);
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

/** The smallest valid PNG, for a placeholder that only has to contain an image. */
const PLACEHOLDER_IMAGE =
	"iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII=";

/** Image part types in the bodies Pi sends: Anthropic, Chat Completions, Responses. */
const IMAGE_PART_TYPES = new Set(["image", "image_url", "input_image"]);

/** Wire shapes of a tool result: Anthropic block, Chat Completions message, Responses item. */
const TOOL_RESULT_TYPES = new Set(["tool_result", "function_call_output", "custom_tool_call_output"]);

export function bodyHasImages(value: unknown): boolean {
	if (Array.isArray(value)) return value.some(bodyHasImages);
	if (!value || typeof value !== "object") return false;
	const record = value as Record<string, unknown>;
	if (typeof record.type === "string" && IMAGE_PART_TYPES.has(record.type)) return true;
	return Object.values(record).some(bodyHasImages);
}

function isToolResultShape(value: unknown): boolean {
	if (!value || typeof value !== "object") return false;
	const record = value as Record<string, unknown>;
	if (record.role === "tool" || record.role === "toolResult") return true;
	if (typeof record.type === "string" && TOOL_RESULT_TYPES.has(record.type)) return true;
	return "toolResult" in record;
}

/**
 * Whether the captured request ends on a tool result.
 *
 * Copilot decides `X-Initiator` from the last message of the context it is
 * handed, before `onPayload` replaces the body. A main request that stopped on
 * a tool result is `agent`. The wire form of that ending is not always role
 * `user`: Anthropic puts `tool_result` blocks inside a user message, Chat
 * Completions uses role `tool`, Responses uses a `function_call_output` item.
 */
export function bodyEndsOnToolResult(payload: unknown): boolean {
	if (!payload || typeof payload !== "object") return false;
	const body = payload as Record<string, unknown>;
	const list = Array.isArray(body.input) ? body.input : Array.isArray(body.messages) ? body.messages : undefined;
	const last = list?.[list.length - 1];
	if (!last || typeof last !== "object") return false;
	if (isToolResultShape(last)) return true;
	const content = (last as { content?: unknown }).content;
	return Array.isArray(content) && content.some(isToolResultShape);
}

export interface PlaceholderOptions {
	/** Copilot refuses an image body that lacks `Copilot-Vision-Request`. */
	withImage?: boolean;
	/** Copilot's `X-Initiator` is `agent` when the last message is not a user turn. */
	agentInitiated?: boolean;
}

/**
 * A throwaway context for the snapshot path: `onPayload` discards the body
 * built from it.
 *
 * The body goes, but not everything the adapter worked out from this context
 * goes with it. GitHub Copilot's adapters set `Copilot-Vision-Request` from
 * whether the context has an image, and `X-Initiator` from the last message's
 * role, both before `onPayload` swaps the body. The placeholder has to produce
 * the same two answers the main request's context did.
 */
export function placeholderContext(question: string, options: PlaceholderOptions = {}): Context {
	const timestamp = Date.now();
	const content: (TextContent | ImageContent)[] = [{ type: "text", text: question }];
	if (options.withImage) content.push({ type: "image", data: PLACEHOLDER_IMAGE, mimeType: "image/png" });
	const messages: Message[] = [{ role: "user", content, timestamp }];
	if (options.agentInitiated) {
		messages.push({
			role: "toolResult",
			toolCallId: "btw",
			toolName: "btw",
			content: [{ type: "text", text: " " }],
			isError: false,
			timestamp,
		});
	}
	return { messages };
}

/**
 * Headers for the side question: the main request's, under the credentials
 * resolved now.
 *
 * The main loop's request wrapper adds headers the adapter never sees as
 * options: OpenCode's session routing, attribution, and whatever
 * `before_provider_headers` handlers set. The side question cannot run that
 * wrapper, so it replays what the wrapper produced for the captured request.
 * A header the provider's auth supplies this time wins over the copy, compared
 * without regard to case, so a token refreshed since then is not sent stale.
 */
export function replayHeaders(captured: ProviderHeaders, fresh: ProviderHeaders): ProviderHeaders {
	const freshNames = new Set(Object.keys(fresh).map((name) => name.toLowerCase()));
	const merged: ProviderHeaders = {};
	for (const [name, value] of Object.entries(captured)) {
		if (!freshNames.has(name.toLowerCase())) merged[name] = value;
	}
	return { ...merged, ...fresh };
}

/**
 * The transport for a side question.
 *
 * Only the Codex adapter reads it. On `auto` that adapter reuses the session's
 * WebSocket and, when the new input does not continue the last response,
 * drops the connection's continuation state and stores the side question's in
 * its place. The main loop's next request then has to resend everything. SSE
 * leaves that WebSocket alone, and the prompt cache still matches on
 * `prompt_cache_key` and the prefix, which the side question keeps.
 */
export function sideTransport(
	model: Model<any>,
	configured: SimpleStreamOptions["transport"],
): SimpleStreamOptions["transport"] {
	return model.api === "openai-codex-responses" ? "sse" : configured;
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
		const payload = snapshot && extendCapturedBody(snapshot.api, snapshot.payload, turns, question, model, signal);

		let context: Context;
		if (snapshot && payload !== undefined) {
			markOwnPayload(payload);
			// Read the captured body, not `payload`: appending the question makes the
			// last turn a user message, which would hide a tool result the main
			// request ended on.
			const copilot = model.provider === "github-copilot";
			context = placeholderContext(question, {
				withImage: copilot && bodyHasImages(snapshot.payload),
				agentInitiated: copilot && bodyEndsOnToolResult(snapshot.payload),
			});
		} else {
			context = rebuildContext(ctx, pi, turns, question, model);
		}

		// The adapter validates the thinking level while building the body it is
		// about to be handed — including the one `onPayload` throws away — so it
		// has to be a level this model accepts.
		const level = ctx.thinkingLevel ?? pi.getThinkingLevel();

		// What the main loop's request wrapper adds from settings. Only the
		// rebuild path's body depends on the thinking budgets; on the snapshot
		// path the captured body already has them.
		const settings = pi.getSettings?.();
		const transport = sideTransport(model, settings?.transport);
		const capturedHeaders = snapshot?.headers;

		// Auth failures surface synchronously here on some providers, which is
		// why the request is built inside the same guard that reads it.
		const stream = ctx.modelRegistry.streamSimple(model, context, {
			signal,
			// Adapters also use this for HTTP session routing; preserving the body
			// alone cannot preserve those headers. The rebuild path needs it too.
			sessionId,
			...(level === "off" ? {} : { reasoning: level }),
			...(transport === undefined ? {} : { transport }),
			...(settings?.thinkingBudgets === undefined ? {} : { thinkingBudgets: settings.thinkingBudgets }),
			...(capturedHeaders === undefined
				? {}
				: { transformHeaders: (fresh: ProviderHeaders) => replayHeaders(capturedHeaders, fresh) }),
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
