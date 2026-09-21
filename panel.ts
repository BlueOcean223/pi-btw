/**
 * The side-question panel: one column where the editor sits, holding the
 * question, the answer, and a short list of earlier questions to page through.
 *
 * It takes the editor's slot rather than floating over the transcript. The main
 * agent is still working while this is up, and covering the stream it is
 * producing would hide the thing the user chose not to interrupt. Pi restores
 * the editor and its unsent draft when the panel closes.
 *
 * The keys differ from Claude Code's on purpose. There, dismissing the panel
 * and merely putting it away are separate paths and only the first aborts. Here
 * the panel owns the keyboard while it is up, so the user has to close it to
 * reach the main view at all, and closing must not mean cancelling. Escape
 * cancels; Enter, Space and Ctrl+] step out of the way and leave the request
 * running, which an empty `/btw` reattaches to.
 */

import type { Theme } from "@earendil-works/pi-coding-agent";
import { getMarkdownTheme } from "@earendil-works/pi-coding-agent";
import {
	type Component,
	Key,
	type KeybindingsManager,
	Markdown,
	matchesKey,
	type TUI,
	truncateToWidth,
	visibleWidth,
} from "@earendil-works/pi-tui";

/** How the panel was closed. `collapse` leaves an open request running. */
export type PanelExit = "dismiss" | "collapse";

/** Claude Code's `je`: how many earlier questions are listed above the current one. */
export const HISTORY_WINDOW = 5;

const COLLAPSE_KEY = "ctrl+]";
const MIN_BODY_LINES = 3;
const MAX_BODY_LINES = 24;
/** Rows the panel needs around the answer: question list, rules, hint row. */
const CHROME_LINES = 8;

/** One row of the panel's list: a stored exchange, or the question being asked now. */
export interface PanelItem {
	question: string;
	/** Read on every render, so a streaming answer keeps redrawing. */
	text(): string;
	/** True while the request behind this item is still open. */
	pending(): boolean;
	/** A model-fallback warning to show above the answer, when there is one. */
	notice(): string | undefined;
}

export interface PanelDeps {
	/** Oldest first. The last entry is the one the panel opens on. */
	items: PanelItem[];
	/** Redraw hook for a live request. Returns an unsubscribe function. */
	subscribe?(onChange: () => void): () => void;
	/** The `x` key. Drops the stored exchanges; the open item stays on screen. */
	clearHistory(): void;
	copy(text: string): void;
	notify(message: string, type?: "info" | "warning" | "error"): void;
}

/** How many rows the answer area gets, given the terminal height. */
export function bodyHeight(terminalRows: number): number {
	return Math.max(MIN_BODY_LINES, Math.min(MAX_BODY_LINES, terminalRows - CHROME_LINES));
}

/**
 * The earlier questions to list, and how many are hidden behind them.
 *
 * `count` is every item except the one at the end. Past the window the rest
 * collapse into a single line, so a session holding twenty exchanges still
 * opens a panel that fits on screen.
 */
export function historyWindow(count: number, window: number = HISTORY_WINDOW): { shown: number; hidden: number } {
	const shown = Math.min(Math.max(0, count), window);
	return { shown, hidden: Math.max(0, count) - shown };
}

export function createPanel(deps: PanelDeps) {
	return (
		tui: TUI,
		theme: Theme,
		keybindings: KeybindingsManager,
		done: (exit: PanelExit) => void,
	): Component & { dispose(): void } => {
		const markdownTheme = getMarkdownTheme();
		const items = deps.items;

		let cursor = Math.max(0, items.length - 1);
		let scroll = 0;
		/** Follow a streaming answer down until the user scrolls away from the end. */
		let stickToEnd = true;
		let cachedWidth = -1;
		let cachedLines: string[] | undefined;

		// Pi always supplies the manager. The duck-typed check, rather than a
		// truthiness test, is what keeps a host passing a partial object from
		// throwing on every keystroke instead of falling back to defaults.
		const bound = (id: Parameters<KeybindingsManager["matches"]>[1], fallback: Parameters<typeof matchesKey>[1]) => {
			const resolver = typeof keybindings?.matches === "function" ? keybindings : undefined;
			return (data: string) => (resolver ? resolver.matches(data, id) : matchesKey(data, fallback));
		};

		const isCancel = bound("tui.select.cancel", "escape");
		const isUp = bound("tui.select.up", "up");
		const isDown = bound("tui.select.down", "down");
		const isConfirm = bound("tui.select.confirm", "enter");
		const isSubmit = bound("tui.input.submit", "enter");

		function refresh() {
			cachedLines = undefined;
			tui.requestRender();
		}

		const unsubscribe = deps.subscribe?.(refresh);

		function current(): PanelItem | undefined {
			return items[cursor];
		}

		function move(delta: number) {
			const next = Math.min(items.length - 1, Math.max(0, cursor + delta));
			if (next === cursor) return;
			cursor = next;
			scroll = 0;
			stickToEnd = true;
			refresh();
		}

		/** The answer as rendered markdown, so the body can be scrolled by line. */
		function answerLines(width: number): string[] {
			const item = current();
			if (!item) return [];
			const lines: string[] = [];
			const notice = item.notice();
			if (notice) lines.push(...new Markdown(`⚠ ${notice}`, 0, 0, markdownTheme).render(width));

			const text = item.text();
			if (text) lines.push(...new Markdown(text, 0, 0, markdownTheme).render(width));
			if (item.pending()) lines.push(theme.fg("dim", "正在回答…"));
			else if (!text) lines.push(theme.fg("dim", "（没有内容）"));
			return lines;
		}

		/**
		 * The key hints, dropped from the least important end until they fit.
		 * A truncated hint row is worse than a short one: it cuts mid-word and
		 * the key it was about to name is the one the reader was looking for.
		 */
		function hintRow(width: number): string {
			const hints = ["Esc 中止", "Ctrl+] 收起", "↑↓ 滚动", "[ ] 切换", "c 复制", "x 清历史"];
			let row = "";
			for (const hint of hints) {
				const next = row ? `${row}  ·  ${hint}` : hint;
				if (visibleWidth(next) > width) break;
				row = next;
			}
			return theme.fg("dim", row);
		}

		function rule(width: number): string {
			return theme.fg("borderMuted", "─".repeat(Math.max(1, width)));
		}

		function render(width: number): string[] {
			if (cachedLines && cachedWidth === width) return cachedLines;
			cachedWidth = width;

			const lines: string[] = [];
			const earlier = items.slice(0, Math.max(0, items.length - 1));
			const { shown, hidden } = historyWindow(earlier.length);
			if (hidden > 0) lines.push(theme.fg("dim", `(+${hidden} earlier /btw)`));

			const firstListed = earlier.length - shown;
			earlier.slice(firstListed).forEach((item, offset) => {
				const label = truncateToWidth(item.question.replace(/\s+/g, " "), Math.max(4, width - 2));
				const row = `· ${label}`;
				lines.push(firstListed + offset === cursor ? theme.bold(row) : theme.fg("muted", row));
			});

			const live = items[items.length - 1];
			if (live) {
				const label = truncateToWidth(live.question.replace(/\s+/g, " "), Math.max(4, width - 6));
				const row = `${theme.fg("accent", "/btw")} ${label}`;
				lines.push(cursor === items.length - 1 ? theme.bold(row) : theme.fg("muted", row));
			}

			lines.push(rule(width));

			const body = answerLines(width);
			const height = bodyHeight(tui.terminal.rows);
			const maxScroll = Math.max(0, body.length - height);
			scroll = stickToEnd ? maxScroll : Math.min(scroll, maxScroll);
			lines.push(...body.slice(scroll, scroll + height));
			if (maxScroll > 0) lines.push(theme.fg("dim", `— ${scroll + 1}/${maxScroll + 1} —`));

			lines.push(rule(width));
			lines.push(hintRow(width));

			cachedLines = lines.map((line) => (visibleWidth(line) > width ? truncateToWidth(line, width) : line));
			return cachedLines;
		}

		function scrollBy(delta: number) {
			const body = answerLines(cachedWidth > 0 ? cachedWidth : 80);
			const maxScroll = Math.max(0, body.length - bodyHeight(tui.terminal.rows));
			scroll = Math.min(maxScroll, Math.max(0, scroll + delta));
			stickToEnd = scroll >= maxScroll;
			refresh();
		}

		return {
			render,
			invalidate() {
				cachedLines = undefined;
			},
			dispose() {
				unsubscribe?.();
			},
			handleInput(data: string): void {
				if (isCancel(data)) {
					done("dismiss");
					return;
				}
				// Enter and Space never abort. While the answer is still coming they
				// put the panel away and leave the request running; once it has
				// landed there is nothing left to leave running.
				if (isConfirm(data) || isSubmit(data) || matchesKey(data, Key.space) || matchesKey(data, COLLAPSE_KEY)) {
					done("collapse");
					return;
				}
				if (isUp(data)) return scrollBy(-1);
				if (isDown(data)) return scrollBy(1);
				if (data === "[" || matchesKey(data, Key.shift("left"))) return move(-1);
				if (data === "]" || matchesKey(data, Key.shift("right"))) return move(1);
				if (data === "c") {
					const text = current()?.text() ?? "";
					if (text) {
						deps.copy(text);
						deps.notify("已复制这条回答", "info");
					}
					return;
				}
				if (data === "x") {
					deps.clearHistory();
					// Keep the item the panel is showing; drop the column above it.
					const live = items[items.length - 1];
					items.length = 0;
					if (live) items.push(live);
					cursor = Math.max(0, items.length - 1);
					refresh();
				}
			},
		};
	};
}
