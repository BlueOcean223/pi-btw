/**
 * The replay rewrite and the reminder around the live question.
 *
 * What is pinned here is the rule "shown once, omitted on replay" and the shape
 * of the fabricated-tool-call test. Both exist because a later side question
 * reads the earlier answers as ordinary assistant turns: whatever survives the
 * rewrite is something the model will treat as having happened.
 */

import { describe, expect, test } from "bun:test";
import {
	CUT_OFF_TAIL,
	FABRICATED_TOOL_CALL,
	NO_TOOLS_NOTICE,
	OMIT_CUT_OFF,
	OMIT_FABRICATED,
	replayAssistant,
	replayResponse,
	SIDE_QUESTION_REMINDER,
	wrapQuestion,
} from "./prompt.js";

describe("FABRICATED_TOOL_CALL", () => {
	const NS = "antml:";

	test("matches an opening tag, with or without the namespace prefix", () => {
		expect(FABRICATED_TOOL_CALL.test("<function_calls>")).toBe(true);
		expect(FABRICATED_TOOL_CALL.test(`<${NS}function_calls>`)).toBe(true);
		expect(FABRICATED_TOOL_CALL.test('<invoke name="Read">')).toBe(true);
		expect(FABRICATED_TOOL_CALL.test(`<${NS}invoke name="Read">`)).toBe(true);
	});

	test("matches a closing tag on its own, for an answer that got cut mid-block", () => {
		expect(FABRICATED_TOOL_CALL.test("</function_calls>")).toBe(true);
		expect(FABRICATED_TOOL_CALL.test(`</${NS}invoke>`)).toBe(true);
	});

	test("leaves prose about tools alone", () => {
		expect(FABRICATED_TOOL_CALL.test("I would call the read tool, but I cannot.")).toBe(false);
		expect(FABRICATED_TOOL_CALL.test("`function_calls` is the block name.")).toBe(false);
	});

	test("is not global — a `g` flag would make repeat tests alternate", () => {
		expect(FABRICATED_TOOL_CALL.flags).not.toContain("g");
		const answer = "<function_calls>";
		expect(FABRICATED_TOOL_CALL.test(answer)).toBe(true);
		expect(FABRICATED_TOOL_CALL.test(answer)).toBe(true);
	});
});

describe("replayResponse", () => {
	test("passes an ordinary answer through unchanged", () => {
		expect(replayResponse("WBC is the white blood cell count.")).toBe("WBC is the white blood cell count.");
	});

	test("omits an answer that drew tool calls, notice and all", () => {
		const shown = `Here is what I would run:\n<invoke name="Read">\n\n${NO_TOOLS_NOTICE}`;
		expect(replayResponse(shown)).toBe(OMIT_FABRICATED);
	});

	test("omits an answer the token limit cut off", () => {
		expect(replayResponse(`The three parts are\n\n${CUT_OFF_TAIL}`)).toBe(OMIT_CUT_OFF);
	});

	test("only omits a cut-off answer when the notice is the tail", () => {
		expect(replayResponse(`${CUT_OFF_TAIL} and then some more`)).not.toBe(OMIT_CUT_OFF);
	});

	test("prefers the fabricated omission when an answer is both", () => {
		const shown = `<function_calls>\n\n${CUT_OFF_TAIL}`;
		expect(replayResponse(shown)).toBe(OMIT_FABRICATED);
	});
});

describe("replayAssistant", () => {
	test("puts a fallback warning above the answer", () => {
		const text = replayAssistant({ response: "42", fallbackNotice: "Answered by another model." });
		expect(text).toBe("⚠ Answered by another model.\n\n42");
	});

	test("leaves an answer without a warning alone", () => {
		expect(replayAssistant({ response: "42" })).toBe("42");
	});

	test("still omits an unusable answer that carries a warning", () => {
		const text = replayAssistant({ response: "<function_calls>", fallbackNotice: "Rerouted." });
		expect(text).toBe(`⚠ Rerouted.\n\n${OMIT_FABRICATED}`);
	});
});

describe("wrapQuestion", () => {
	test("puts the reminder first and the user's own words last", () => {
		const wrapped = wrapQuestion("WBC 在这份 diff 里是什么");
		expect(wrapped.startsWith(SIDE_QUESTION_REMINDER)).toBe(true);
		expect(wrapped.endsWith("WBC 在这份 diff 里是什么")).toBe(true);
	});

	test("tells the model it has no tools and must not draw them", () => {
		expect(SIDE_QUESTION_REMINDER).toContain("You have NO tools available");
		expect(SIDE_QUESTION_REMINDER).toContain("function_calls");
	});

	test("tells the model the main agent was not interrupted", () => {
		expect(SIDE_QUESTION_REMINDER).toContain("The main agent is NOT interrupted");
	});
});
