/**
 * The in-memory exchange list: the cap, the clear, and the per-session split.
 *
 * The split is the one with teeth. Claude Code shipped a bug where one
 * session's side questions showed up in another's; here the only thing keeping
 * them apart is that every read and write is keyed by session id.
 */

import { beforeEach, describe, expect, test } from "bun:test";
import {
	appendExchange,
	clearHistory,
	clearInflight,
	getHistory,
	getInflight,
	type Inflight,
	MAX_HISTORY,
	notifyInflight,
	resetAllHistory,
	setInflight,
} from "./history.js";
import type { SideResult } from "./request.js";

function exchange(question: string) {
	return { question, response: `answer to ${question}`, createdAt: 0 };
}

beforeEach(() => {
	resetAllHistory();
});

describe("exchange list", () => {
	test("keeps exchanges in the order they were answered", () => {
		appendExchange("s1", exchange("one"));
		appendExchange("s1", exchange("two"));
		expect(getHistory("s1").map((e) => e.question)).toEqual(["one", "two"]);
	});

	test(`keeps only the most recent ${MAX_HISTORY}`, () => {
		for (let i = 0; i < MAX_HISTORY + 5; i++) appendExchange("s1", exchange(`q${i}`));
		const stored = getHistory("s1");
		expect(stored).toHaveLength(MAX_HISTORY);
		expect(stored[0]?.question).toBe("q5");
		expect(stored[stored.length - 1]?.question).toBe(`q${MAX_HISTORY + 4}`);
	});

	test("clear drops the whole list for that session and nobody else's", () => {
		appendExchange("s1", exchange("mine"));
		appendExchange("s2", exchange("theirs"));
		clearHistory("s1");
		expect(getHistory("s1")).toEqual([]);
		expect(getHistory("s2").map((e) => e.question)).toEqual(["theirs"]);
	});

	test("reads return a copy, so a caller cannot edit the stored list", () => {
		appendExchange("s1", exchange("one"));
		getHistory("s1").push(exchange("smuggled"));
		expect(getHistory("s1")).toHaveLength(1);
	});

	test("a session with no exchanges reads as empty, not undefined", () => {
		expect(getHistory("never-used")).toEqual([]);
	});
});

describe("open requests", () => {
	function open(sessionId: string, question: string): Inflight {
		return {
			sessionId,
			question,
			controller: new AbortController(),
			text: "",
			listeners: new Set(),
			settled: Promise.resolve({ response: "", synthetic: true, aborted: false } satisfies SideResult),
		};
	}

	test("one open request per session, looked up by id", () => {
		const a = open("s1", "a");
		const b = open("s2", "b");
		setInflight(a);
		setInflight(b);
		expect(getInflight("s1")).toBe(a);
		expect(getInflight("s2")).toBe(b);
	});

	test("clearing a superseded request leaves the newer one in place", () => {
		const first = open("s1", "first");
		const second = open("s1", "second");
		setInflight(first);
		setInflight(second);
		clearInflight(first);
		expect(getInflight("s1")).toBe(second);
	});

	test("every subscribed panel is told when the answer grows", () => {
		const request = open("s1", "q");
		let calls = 0;
		request.listeners.add(() => calls++);
		request.listeners.add(() => calls++);
		notifyInflight(request);
		expect(calls).toBe(2);
	});
});
