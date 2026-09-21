/**
 * The panel's two pieces of arithmetic, kept out of the component so they can
 * be checked without a terminal.
 */

import { describe, expect, test } from "bun:test";
import { bodyHeight, createPanel, HISTORY_WINDOW, historyWindow, type PanelItem } from "./panel.js";

describe("historyWindow", () => {
	test("lists every earlier question while they fit", () => {
		expect(historyWindow(3)).toEqual({ shown: 3, hidden: 0 });
		expect(historyWindow(HISTORY_WINDOW)).toEqual({ shown: HISTORY_WINDOW, hidden: 0 });
	});

	test("collapses the rest into a count once past the window", () => {
		expect(historyWindow(19)).toEqual({ shown: HISTORY_WINDOW, hidden: 19 - HISTORY_WINDOW });
	});

	test("a first question has nothing above it", () => {
		expect(historyWindow(0)).toEqual({ shown: 0, hidden: 0 });
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

describe("clearing the column", () => {
	/**
	 * `x` drops the stored exchanges but not the answer on screen. The panel
	 * edits its own item list in place, so this checks the list rather than the
	 * rendering.
	 */
	test("keeps the open item and drops the ones above it", () => {
		const item = (question: string): PanelItem => ({
			question,
			text: () => `answer to ${question}`,
			pending: () => false,
			notice: () => undefined,
		});
		const items = [item("old one"), item("old two"), item("current")];

		let cleared = false;
		const factory = createPanel({
			items,
			clearHistory: () => {
				cleared = true;
			},
			copy: () => {},
			notify: () => {},
		});

		const tui = { terminal: { rows: 40, columns: 80 }, requestRender: () => {} };
		const theme = { fg: (_c: string, t: string) => t, bold: (t: string) => t };
		const panel = factory(tui as never, theme as never, undefined as never, () => {});
		panel.handleInput?.("x");

		expect(cleared).toBe(true);
		expect(items.map((i) => i.question)).toEqual(["current"]);
	});
});
