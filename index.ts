/**
 * pi-btw — `/btw`, a side question answered from the main conversation's own
 * context while the main task keeps running.
 *
 * `/btw what is WBC in this diff` opens a panel where the editor was and
 * answers in one shot. The main agent is not steered, not interrupted, and
 * never sees the question: nothing here reaches the session file or the
 * transcript, so the next main turn carries the same bytes it would have
 * carried anyway.
 *
 * The side question shares the main request's prefix — same system prompt, same
 * tool schemas, same messages — because that prefix is already in the provider's
 * cache. It cannot call those tools; the schemas are there for the cache, and
 * nothing feeds a tool result back. An empty `/btw` reopens the last one, or
 * reattaches to one still in flight.
 *
 * Registered as an extension command, which pi dispatches before the prompt is
 * queued and therefore while the agent is mid-tool. A user message or
 * `sendUserMessage` would instead become a steer or a follow-up.
 */

import type { ExtensionAPI, ExtensionCommandContext } from "@earendil-works/pi-coding-agent";
import { copyToClipboard } from "@earendil-works/pi-coding-agent";
import {
	appendExchange,
	clearHistory,
	clearInflight,
	type Exchange,
	getHistory,
	getInflight,
	type Inflight,
	notifyInflight,
	setInflight,
} from "./history.js";
import { createPanel, type PanelExit, type PanelItem } from "./panel.js";
import { runSideQuestion, shouldRemember, type SideResult } from "./request.js";
import { recordSnapshot } from "./snapshot.js";

const USAGE = "用法：/btw <问题>";
const TUI_ONLY = "/btw 只在交互界面可用，这次没有发出请求";

/**
 * The session the user is looking at right now.
 *
 * Tracked here rather than read back off a captured context because an answer
 * can land after a session switch, and by then the context that started it
 * belongs to a runtime pi has already replaced. Module state survives that
 * replacement, which is also what lets a session keep its exchanges when the
 * user switches away and back inside one process.
 */
let activeSessionId: string | undefined;

export default function btw(pi: ExtensionAPI) {
	pi.on("session_start", (_event, ctx) => {
		activeSessionId = ctx.sessionManager.getSessionId();
	});

	pi.on("before_provider_request", (event, ctx) => {
		const model = ctx.model;
		if (!model) return;
		try {
			recordSnapshot({
				sessionId: ctx.sessionManager.getSessionId(),
				provider: model.provider,
				modelId: model.id,
				api: model.api,
				payload: event.payload,
			});
		} catch {
			// A body that will not clone is a body this extension cannot reuse.
			// The rebuild path covers it; the main request must not be disturbed.
		}
		// Observe only. Returning a payload here would rewrite the main request.
		return undefined;
	});

	pi.registerCommand("btw", {
		description: "Ask a side question without interrupting the main task",
		handler: async (args: string, ctx: ExtensionCommandContext) => {
			const question = args.trim();
			if (ctx.mode !== "tui") {
				ctx.ui.notify(TUI_ONLY, "warning");
				return;
			}
			activeSessionId = ctx.sessionManager.getSessionId();
			if (!question) {
				await reopen(ctx);
				return;
			}
			await ask(pi, ctx, question);
		},
	});
}

/** Start a side question, or reattach to the identical one already running. */
async function ask(pi: ExtensionAPI, ctx: ExtensionCommandContext, question: string): Promise<void> {
	const sessionId = ctx.sessionManager.getSessionId();
	const open = getInflight(sessionId);

	let request: Inflight;
	if (open && open.question === question && !open.result) {
		request = open;
	} else {
		// Two side questions at once would race for the same panel and file two
		// exchanges in an order neither of them chose. The newer one wins; an
		// exchange already stored stays stored.
		open?.controller.abort();
		request = start(pi, ctx, sessionId, question);
	}

	await showPanel(ctx, sessionId, request);
}

/** Empty `/btw`: reattach to an open request, or reopen the last answer. */
async function reopen(ctx: ExtensionCommandContext): Promise<void> {
	const sessionId = ctx.sessionManager.getSessionId();
	const request = getInflight(sessionId);
	if (!request && getHistory(sessionId).length === 0) {
		ctx.ui.notify(USAGE, "info");
		return;
	}
	await showPanel(ctx, sessionId, request);
}

function start(pi: ExtensionAPI, ctx: ExtensionCommandContext, sessionId: string, question: string): Inflight {
	const controller = new AbortController();
	// The exchanges as of now. A later `x` clears the stored list without
	// changing what this request already committed to sending.
	const history = getHistory(sessionId);

	const request: Inflight = {
		sessionId,
		question,
		controller,
		text: "",
		listeners: new Set(),
		settled: undefined as unknown as Promise<SideResult>,
	};

	request.settled = (async () => {
		let result: SideResult;
		try {
			result = await runSideQuestion({
				ctx,
				pi,
				sessionId,
				question,
				history,
				signal: controller.signal,
				onText: (text) => {
					request.text = text;
					notifyInflight(request);
				},
			});
		} catch (error) {
			result = {
				response: `(API error: ${error instanceof Error ? error.message : String(error)})`,
				synthetic: true,
				aborted: false,
			};
		}

		request.result = result;
		// An answer that lands after the user moved to another session belongs to
		// neither: filing it under the current one would put a question that was
		// never asked there into that session's replay list.
		if (shouldRemember(result) && activeSessionId === sessionId) {
			appendExchange(sessionId, {
				question,
				response: result.response,
				...(result.fallbackNotice ? { fallbackNotice: result.fallbackNotice } : {}),
				createdAt: Date.now(),
			});
		}
		clearInflight(request);
		notifyInflight(request);
		reportUsage(ctx, result);
		return result;
	})();

	setInflight(request);
	return request;
}

/**
 * Cache accounting, off unless `PI_BTW_DEBUG` is set.
 *
 * The numbers are how you tell whether the shared prefix actually hit: a second
 * side question should read most of the context from cache and write almost
 * nothing. A `cacheRead` of zero next to a `cacheWrite` the size of the whole
 * conversation means the prefix diverged from the main request's.
 */
function reportUsage(ctx: ExtensionCommandContext, result: SideResult): void {
	if (!process.env.PI_BTW_DEBUG || !result.usage) return;
	try {
		ctx.ui.notify(`/btw cacheRead=${result.usage.cacheRead} cacheWrite=${result.usage.cacheWrite}`, "info");
	} catch {
		// The session this answer belongs to is gone. Nothing to report it to.
	}
}

function storedItem(exchange: Exchange): PanelItem {
	return {
		question: exchange.question,
		text: () => exchange.response,
		pending: () => false,
		notice: () => exchange.fallbackNotice,
	};
}

function liveItem(request: Inflight): PanelItem {
	return {
		question: request.question,
		text: () => request.result?.response ?? request.text,
		pending: () => request.result === undefined,
		notice: () => request.result?.fallbackNotice,
	};
}

async function showPanel(ctx: ExtensionCommandContext, sessionId: string, request: Inflight | undefined): Promise<void> {
	const items = getHistory(sessionId).map(storedItem);
	if (request) items.push(liveItem(request));
	if (items.length === 0) {
		ctx.ui.notify(USAGE, "info");
		return;
	}

	const exit: PanelExit = await ctx.ui.custom<PanelExit>(
		createPanel({
			items,
			subscribe: request
				? (onChange) => {
						request.listeners.add(onChange);
						return () => request.listeners.delete(onChange);
					}
				: undefined,
			clearHistory: () => clearHistory(sessionId),
			copy: (text) => {
				void copyToClipboard(text).catch(() => ctx.ui.notify("复制失败", "error"));
			},
			notify: (message, type) => ctx.ui.notify(message, type),
		}),
	);

	// Escape is the only key that cancels. Collapsing leaves the request open so
	// an empty `/btw` can pick the same answer back up.
	if (exit === "dismiss" && request && request.result === undefined) {
		request.controller.abort();
		clearInflight(request);
	}
}
