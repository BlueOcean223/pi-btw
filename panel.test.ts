/**
 * The panel's two pieces of arithmetic, kept out of the component so they can
 * be checked without a terminal.
 */

import { describe, expect, test } from "bun:test";
import { bodyHeight, createPanel, HISTORY_WINDOW, historyWindow, type PanelItem } from "./panel.js";

describe("historyWindow", () => {
	/** Cursor on the live question, i.e. past every earlier one. */
	const atLive = (count: number) => count;

	test("lists every earlier question while they fit", () => {
		expect(historyWindow(3, atLive(3))).toEqual({ start: 0, shown: 3, hidden: 0 });
		expect(historyWindow(HISTORY_WINDOW, atLive(HISTORY_WINDOW))).toEqual({
			start: 0,
			shown: HISTORY_WINDOW,
			hidden: 0,
		});
	});

	test("collapses the rest into a count once past the window", () => {
		expect(historyWindow(19, atLive(19))).toEqual({
			start: 19 - HISTORY_WINDOW,
			shown: HISTORY_WINDOW,
			hidden: 19 - HISTORY_WINDOW,
		});
	});

	test("a first question has nothing above it", () => {
		expect(historyWindow(0, atLive(0))).toEqual({ start: 0, shown: 0, hidden: 0 });
	});

	/**
	 * Paging back past the window has to bring the window with it. Otherwise the
	 * answer area changes while every drawn row stays unmarked, and there is
	 * nothing on screen saying which question produced what is being read.
	 */
	test("the window slides back to keep the cursor drawn", () => {
		const { start, shown, hidden } = historyWindow(19, 2);
		expect(start).toBe(2);
		expect(shown).toBe(HISTORY_WINDOW);
		expect(hidden).toBe(2);
		expect(2 >= start && 2 < start + shown).toBe(true);
	});

	test("the cursor is inside the drawn window wherever it sits", () => {
		for (let cursor = 0; cursor <= 19; cursor++) {
			const { start, shown } = historyWindow(19, cursor);
			const drawn = cursor >= start && cursor < start + shown;
			// The live question is drawn by its own row, not from this window.
			expect(drawn || cursor === 19).toBe(true);
		}
	});
});

describe("bodyHeight", () => {
	test("grows with the terminal", () => {
		expect(bodyHeight(30)).toBeGreaterThan(bodyHeight(20));
	});

	test("still leaves an answer area on a short terminal", () => {
		expect(bodyHeight(8)).toBeGreaterThanOrEqual(3);
	});

	test("stops growing, so the panel never swallows the transcript", () => {
		expect(bodyHeight(200)).toBe(bodyHeight(400));
	});
});

/** A panel wired to stub deps, with the calls it made recorded. */
function mount(questions: string[]) {
	const items: PanelItem[] = questions.map((question) => ({
		question,
		text: () => `answer to ${question}`,
		pending: () => false,
		notice: () => undefined,
	}));
	const calls = { cleared: [] as number[], copied: [] as string[], exits: [] as string[] };
	const factory = createPanel({
		items,
		clearHistory: (keptIndex) => calls.cleared.push(keptIndex),
		copy: (text) => calls.copied.push(text),
		notify: () => {},
	});
	const tui = { terminal: { rows: 40, columns: 80 }, requestRender: () => {} };
	const theme = { fg: (_c: string, t: string) => t, bold: (t: string) => t, italic: (t: string) => t };
	const panel = factory(tui as never, theme as never, undefined as never, (exit) => calls.exits.push(exit));
	return { panel, items, calls };
}

describe("clearing the column", () => {
	test("keeps the item being viewed, not whichever one is last", () => {
		const { panel, items, calls } = mount(["old one", "old two", "current"]);
		panel.handleInput?.("["); // page back onto "old two"
		panel.handleInput?.("x");

		expect(calls.cleared).toEqual([1]);
		expect(items.map((i) => i.question)).toEqual(["old two"]);
	});

	test("keeps the live item when that is what is on screen", () => {
		const { panel, items, calls } = mount(["old one", "current"]);
		panel.handleInput?.("x");

		expect(calls.cleared).toEqual([1]);
		expect(items.map((i) => i.question)).toEqual(["current"]);
	});
});

/**
 * Under the Kitty keyboard protocol and xterm's modifyOtherKeys, a plain
 * letter does not arrive as that letter. These are the sequences those two
 * terminals send, and the letter keys have to keep working in them — the hint
 * row advertises them either way.
 */
describe("letter keys under terminal keyboard protocols", () => {
	const kitty = (codepoint: number) => `\x1b[${codepoint}u`;
	const modifyOtherKeys = (codepoint: number) => `\x1b[27;1;${codepoint}~`;

	test("copy fires for the raw byte and for both encodings", () => {
		for (const data of ["c", kitty(99), modifyOtherKeys(99)]) {
			const { panel, calls } = mount(["q"]);
			panel.handleInput?.(data);
			expect(calls.copied).toEqual(["answer to q"]);
		}
	});

	test("clear fires for the raw byte and for both encodings", () => {
		for (const data of ["x", kitty(120), modifyOtherKeys(120)]) {
			const { panel, calls } = mount(["q"]);
			panel.handleInput?.(data);
			expect(calls.cleared).toEqual([0]);
		}
	});

	test("the bracket keys page for the raw byte and for both encodings", () => {
		for (const data of ["[", kitty(91), modifyOtherKeys(91)]) {
			const { panel, items, calls } = mount(["older", "newer"]);
			panel.handleInput?.(data);
			panel.handleInput?.("x");
			expect(calls.cleared).toEqual([0]);
			expect(items.map((i) => i.question)).toEqual(["older"]);
		}
	});
});
