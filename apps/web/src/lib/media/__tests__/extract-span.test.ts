import { describe, expect, test } from "bun:test";
import {
	SpanExtractionError,
	extractSpan,
	rebaseSampleTiming,
	resolveSpanBounds,
	spanFileName,
} from "@/lib/media/extract-span";

describe("resolveSpanBounds", () => {
	test("passes a span that sits inside the source through untouched", () => {
		expect(
			resolveSpanBounds({ start: 230, end: 244, sourceDuration: 774 }),
		).toEqual({ start: 230, end: 244, duration: 14 });
	});

	test("clamps a negative start to zero", () => {
		expect(
			resolveSpanBounds({ start: -2, end: 5, sourceDuration: 774 }),
		).toEqual({ start: 0, end: 5, duration: 5 });
	});

	test("clamps an end past the source duration", () => {
		expect(
			resolveSpanBounds({ start: 770, end: 800, sourceDuration: 774 }),
		).toEqual({ start: 770, end: 774, duration: 4 });
	});

	test("trusts the caller's end when the duration is unknown", () => {
		expect(
			resolveSpanBounds({ start: 10, end: 20, sourceDuration: 0 }),
		).toEqual({ start: 10, end: 20, duration: 10 });
	});

	test("rejects a span that starts at or after the source end", () => {
		expect(() =>
			resolveSpanBounds({ start: 800, end: 810, sourceDuration: 774 }),
		).toThrow(SpanExtractionError);
	});

	test("rejects a zero-length span", () => {
		expect(() =>
			resolveSpanBounds({ start: 5, end: 5, sourceDuration: 774 }),
		).toThrow(SpanExtractionError);
	});

	test("rejects non-finite bounds", () => {
		expect(() =>
			resolveSpanBounds({
				start: 0,
				end: Number.POSITIVE_INFINITY,
				sourceDuration: 774,
			}),
		).toThrow(SpanExtractionError);
	});

	test("reports empty-span as the failure reason", () => {
		try {
			resolveSpanBounds({ start: 5, end: 4, sourceDuration: 774 });
			throw new Error("expected resolveSpanBounds to throw");
		} catch (error) {
			expect(error).toBeInstanceOf(SpanExtractionError);
			expect((error as SpanExtractionError).reason).toBe("empty-span");
		}
	});
});

describe("rebaseSampleTiming", () => {
	const fallbackDuration = 1 / 30;

	test("shifts a frame inside the span onto a zero-based timeline", () => {
		expect(
			rebaseSampleTiming({
				timestamp: 231.5,
				duration: 0.04,
				spanStart: 230,
				fallbackDuration,
			}),
		).toEqual({ timestamp: 1.5, duration: 0.04 });
	});

	test("pins the frame straddling the cut to zero and trims its lead-in", () => {
		// Decoding starts at the keyframe before 230s, so the first frame
		// handed back starts at 229.98 and covers the cut.
		const timing = rebaseSampleTiming({
			timestamp: 229.98,
			duration: 0.04,
			spanStart: 230,
			fallbackDuration,
		});
		expect(timing.timestamp).toBe(0);
		expect(timing.duration).toBeCloseTo(0.02, 10);
	});

	test("reports a non-positive duration for frames that end before the cut", () => {
		const timing = rebaseSampleTiming({
			timestamp: 229.9,
			duration: 0.04,
			spanStart: 230,
			fallbackDuration,
		});
		expect(timing.duration).toBeLessThanOrEqual(0);
	});

	test("substitutes the fallback duration when the source reports none", () => {
		expect(
			rebaseSampleTiming({
				timestamp: 231,
				duration: 0,
				spanStart: 230,
				fallbackDuration,
			}),
		).toEqual({ timestamp: 1, duration: fallbackDuration });
	});

	test("keeps a frame that starts exactly on the cut at zero", () => {
		expect(
			rebaseSampleTiming({
				timestamp: 230,
				duration: 0.04,
				spanStart: 230,
				fallbackDuration,
			}),
		).toEqual({ timestamp: 0, duration: 0.04 });
	});

	test("produces monotonically increasing timestamps across a frame run", () => {
		const spanStart = 230;
		const timestamps = [229.98, 230.02, 230.06, 230.1].map(
			(timestamp) =>
				rebaseSampleTiming({
					timestamp,
					duration: 0.04,
					spanStart,
					fallbackDuration,
				}).timestamp,
		);
		for (let i = 1; i < timestamps.length; i++) {
			expect(timestamps[i]).toBeGreaterThan(timestamps[i - 1]);
		}
	});
});

describe("spanFileName", () => {
	test("keeps the source stem and always ends in .mp4", () => {
		expect(
			spanFileName({ sourceName: "ticket-2min.mp4", start: 230, end: 244 }),
		).toBe("ticket-2min-span-230.00-244.00.mp4");
	});

	test("handles a source name with no extension", () => {
		expect(spanFileName({ sourceName: "clip", start: 0, end: 1.5 })).toBe(
			"clip-span-0.00-1.50.mp4",
		);
	});

	test("falls back to a generic stem for an empty name", () => {
		expect(spanFileName({ sourceName: "", start: 0, end: 1 })).toBe(
			"video-span-0.00-1.00.mp4",
		);
	});
});

describe("extractSpan", () => {
	test("is exported as a function (WebCodecs path needs a browser)", () => {
		expect(typeof extractSpan).toBe("function");
	});
});
