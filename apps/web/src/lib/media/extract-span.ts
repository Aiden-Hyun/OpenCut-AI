import {
	ALL_FORMATS,
	BlobSource,
	BufferTarget,
	Input,
	Mp4OutputFormat,
	Output,
	QUALITY_HIGH,
	VideoSampleSink,
	VideoSampleSource,
	getFirstEncodableVideoCodec,
} from "mediabunny";

/** Frame rate assumed when the source's packet stats are unusable. */
const FALLBACK_FRAME_RATE = 30;
/** Packets sampled to estimate the source frame rate. */
const FRAME_RATE_SAMPLE_SIZE = 100;

export type SpanExtractionFailure =
	/** The file has no video track to cut from. */
	| "no-video-track"
	/** The source codec cannot be decoded by this browser's WebCodecs. */
	| "undecodable-source"
	/** No mp4-compatible video codec this browser can encode. */
	| "no-encodable-codec"
	/** start/end do not describe a positive-length range inside the source. */
	| "empty-span"
	/** Decoding produced nothing inside the requested range. */
	| "no-frames"
	/** Decode/encode blew up part way through. */
	| "transcode-failed";

/** Thrown by {@link extractSpan}. Callers switch on `reason` to decide
 *  whether to fall back to uploading the whole file. */
export class SpanExtractionError extends Error {
	readonly reason: SpanExtractionFailure;

	constructor(reason: SpanExtractionFailure, message: string) {
		super(message);
		this.name = "SpanExtractionError";
		this.reason = reason;
	}
}

/** Clamp a requested [start, end] into the source and reject empty ranges.
 *  `sourceDuration <= 0` means "unknown", in which case `end` is trusted. */
export function resolveSpanBounds({
	start,
	end,
	sourceDuration,
}: {
	start: number;
	end: number;
	sourceDuration: number;
}): { start: number; end: number; duration: number } {
	if (!Number.isFinite(start) || !Number.isFinite(end)) {
		throw new SpanExtractionError(
			"empty-span",
			"Span start and end must be finite numbers.",
		);
	}

	const clampedStart = Math.max(0, start);
	const clampedEnd = sourceDuration > 0 ? Math.min(end, sourceDuration) : end;

	if (clampedEnd - clampedStart <= 0) {
		throw new SpanExtractionError(
			"empty-span",
			`Span ${start}s–${end}s is empty after clamping to the source.`,
		);
	}

	return {
		start: clampedStart,
		end: clampedEnd,
		duration: clampedEnd - clampedStart,
	};
}

/** Re-base one decoded frame onto a span-local timeline.
 *
 *  Decoding starts at the keyframe preceding `spanStart`, so the first frame
 *  handed back straddles the cut: it is pinned to 0 and loses the part that
 *  falls outside the span. Frames that end at or before `spanStart` are
 *  reported with a non-positive duration so the caller drops them. */
export function rebaseSampleTiming({
	timestamp,
	duration,
	spanStart,
	fallbackDuration,
}: {
	timestamp: number;
	duration: number;
	spanStart: number;
	fallbackDuration: number;
}): { timestamp: number; duration: number } {
	const effectiveDuration = duration > 0 ? duration : fallbackDuration;
	const shifted = timestamp - spanStart;

	if (shifted >= 0) {
		return { timestamp: shifted, duration: effectiveDuration };
	}

	return { timestamp: 0, duration: effectiveDuration + shifted };
}

/** Name for an extracted span, derived from the source so the cleaned clip
 *  is recognisable in the media panel. */
export function spanFileName({
	sourceName,
	start,
	end,
}: {
	sourceName: string;
	start: number;
	end: number;
}): string {
	const base = (sourceName || "video").replace(/\.[^.]+$/, "") || "video";
	const stamp = (seconds: number) => Math.max(0, seconds).toFixed(2);
	return `${base}-span-${stamp(start)}-${stamp(end)}.mp4`;
}

/** Cut [start, end] out of a video file and re-encode just that span to mp4.
 *
 *  Video only — the overlay clip this feeds is muted and the original's audio
 *  keeps playing underneath, so decoding and re-encoding audio would be pure
 *  waste. Resolution, frame rate and rotation are carried over from the
 *  source.
 *
 *  Throws {@link SpanExtractionError} when the browser cannot decode the
 *  source or encode the result, so callers can fall back to a server-side
 *  path. */
export async function extractSpan({
	file,
	start,
	end,
	signal,
}: {
	file: File;
	start: number;
	end: number;
	signal?: AbortSignal;
}): Promise<File> {
	const input = new Input({
		source: new BlobSource(file),
		formats: ALL_FORMATS,
	});

	// Released in the `finally` below: callers run this once per segment, so
	// holding demuxers open across a run would pile up.
	try {
		return await extractSpanFromInput({
			input,
			file,
			start,
			end,
			signal,
		});
	} finally {
		input.dispose();
	}
}

async function extractSpanFromInput({
	input,
	file,
	start,
	end,
	signal,
}: {
	input: Input;
	file: File;
	start: number;
	end: number;
	signal?: AbortSignal;
}): Promise<File> {
	let sourceDuration = 0;
	try {
		sourceDuration = await input.computeDuration();
	} catch {
		// Unknown duration: trust the caller's `end` instead of clamping.
	}

	const bounds = resolveSpanBounds({ start, end, sourceDuration });

	const videoTrack = await input.getPrimaryVideoTrack();
	if (!videoTrack) {
		throw new SpanExtractionError(
			"no-video-track",
			"The source file has no video track to extract from.",
		);
	}

	if (!(await videoTrack.canDecode())) {
		throw new SpanExtractionError(
			"undecodable-source",
			"This browser cannot decode the source video codec.",
		);
	}

	const format = new Mp4OutputFormat();
	const codec = await getFirstEncodableVideoCodec(
		format.getSupportedVideoCodecs(),
		{
			width: videoTrack.codedWidth,
			height: videoTrack.codedHeight,
		},
	);
	if (!codec) {
		throw new SpanExtractionError(
			"no-encodable-codec",
			"This browser cannot encode an mp4 video track.",
		);
	}

	let frameRate = FALLBACK_FRAME_RATE;
	try {
		const stats = await videoTrack.computePacketStats(FRAME_RATE_SAMPLE_SIZE);
		if (
			Number.isFinite(stats.averagePacketRate) &&
			stats.averagePacketRate > 0
		) {
			frameRate = stats.averagePacketRate;
		}
	} catch {
		// Keep the fallback rate; it only affects container metadata and the
		// duration substituted for frames that report none.
	}
	const fallbackDuration = 1 / frameRate;

	const output = new Output({ format, target: new BufferTarget() });
	const videoSource = new VideoSampleSource({ codec, bitrate: QUALITY_HIGH });
	output.addVideoTrack(videoSource, {
		frameRate,
		rotation: videoTrack.rotation,
	});
	await output.start();

	let emitted = 0;
	try {
		const sink = new VideoSampleSink(videoTrack);
		for await (const sample of sink.samples(bounds.start, bounds.end)) {
			try {
				if (signal?.aborted) {
					throw new DOMException("Span extraction aborted", "AbortError");
				}

				const timing = rebaseSampleTiming({
					timestamp: sample.timestamp,
					duration: sample.duration,
					spanStart: bounds.start,
					fallbackDuration,
				});
				if (timing.duration <= 0) continue;

				sample.setTimestamp(timing.timestamp);
				sample.setDuration(timing.duration);
				await videoSource.add(sample);
				emitted++;
			} finally {
				sample.close();
			}
		}
	} catch (error) {
		await output.cancel();
		if (error instanceof DOMException && error.name === "AbortError") {
			throw error;
		}
		throw new SpanExtractionError(
			"transcode-failed",
			error instanceof Error
				? `Failed to extract the span: ${error.message}`
				: "Failed to extract the span.",
		);
	}

	if (emitted === 0) {
		await output.cancel();
		throw new SpanExtractionError(
			"no-frames",
			`No frames decoded between ${bounds.start}s and ${bounds.end}s.`,
		);
	}

	videoSource.close();
	await output.finalize();

	const buffer = output.target.buffer;
	if (!buffer) {
		throw new SpanExtractionError(
			"transcode-failed",
			"The extracted span produced no output data.",
		);
	}

	return new File(
		[buffer],
		spanFileName({
			sourceName: file.name,
			start: bounds.start,
			end: bounds.end,
		}),
		{ type: "video/mp4" },
	);
}
