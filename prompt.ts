/**
 * The fixed text of a side question: the reminder wrapped around the live
 * question, and the four sentences used when an answer is shown once and then
 * replayed into the next request.
 *
 * These strings are read by the model, not the user, so they are English and
 * fixed rather than prose to polish. The panel chrome the user reads is
 * Chinese and lives in panel.ts.
 */

/**
 * Wrapped around the current question only. Replayed questions in the history
 * are plain user messages — re-wrapping them would change bytes the model has
 * already seen and say "answer this one" about a question already answered.
 */
export const SIDE_QUESTION_REMINDER = `<system-reminder>This is a side question from the user. You must answer this question directly in a single response.

IMPORTANT CONTEXT:
- You are a separate, lightweight agent spawned to answer this one question
- The main agent is NOT interrupted - it continues working independently in the background
- You share the conversation context but are a completely separate instance
- Do NOT reference being interrupted or what you were "previously doing" - that framing is incorrect

CRITICAL CONSTRAINTS:
- You have NO tools available - you cannot read files, run commands, search, or take any actions
- Do NOT write tool calls or tool output as text (for example invoke or function_calls XML blocks) - nothing you write here is executed; if answering would need reading files, running commands, or searching, say that can't be checked from a side question and suggest asking in the main conversation
- This is a one-off response - there will be no follow-up turns
- You can ONLY provide information based on what you already know from the conversation context
- NEVER say things like "Let me try...", "I'll now...", "Let me check...", or promise to take any action
- If you don't know the answer, say so - do not offer to look it up or investigate

Simply answer the question with the information you have.</system-reminder>`;

/** The user message sent for the question being asked right now. */
export function wrapQuestion(question: string): string {
	return `${SIDE_QUESTION_REMINDER}\n\n${question}`;
}

/**
 * A tool call drawn in prose rather than emitted as a protocol block. Both the
 * opening and the closing tag count, and the `antml:` prefix is optional.
 *
 * Deliberately not global: a `g` regex carries `lastIndex` between `.test()`
 * calls, so the same answer would match and then not match.
 */
export const FABRICATED_TOOL_CALL =
	/<(?:antml:)?(?:function_calls>|invoke name=)|<\/(?:antml:)?(?:function_calls|invoke)>/;

/**
 * Appended once, under an answer that drew tool calls as text. The user has to
 * see what the model wrote — it may still hold the answer — but has to be told
 * none of it ran.
 */
export const NO_TOOLS_NOTICE =
	"_/btw can't run tools: any tool calls or tool output shown above were not executed and may not reflect your actual files or data. Ask in the main conversation to check._";

/** Appended once, under an answer the token limit cut short. */
export const CUT_OFF_TAIL = "_(This answer was cut off before it finished. Ask again to retry.)_";

/** Replaces the whole answer when it is replayed into a later side question. */
export const OMIT_FABRICATED = "(That answer wrote tool calls as text. Nothing was executed, so it is omitted here.)";
export const OMIT_CUT_OFF = "(That answer was cut off before it finished, so it is omitted here.)";

/**
 * Rewrite a stored answer for replay as an assistant message.
 *
 * Shown once, omitted afterwards. A fabricated tool block replayed verbatim
 * reads to the next side question like a tool that actually ran, and half a
 * sentence replayed verbatim reads like a claim the model finished making.
 */
export function replayResponse(response: string): string {
	if (FABRICATED_TOOL_CALL.test(response)) return OMIT_FABRICATED;
	if (response.endsWith(CUT_OFF_TAIL)) return OMIT_CUT_OFF;
	return response;
}

/** The assistant message text for one stored exchange, notice included. */
export function replayAssistant(exchange: { response: string; fallbackNotice?: string }): string {
	const body = replayResponse(exchange.response);
	return exchange.fallbackNotice ? `⚠ ${exchange.fallbackNotice}\n\n${body}` : body;
}
