import type { TimelineTrack } from "@/types/timeline";
import type {
	AnimationInterpolation,
	AnimationPropertyPath,
} from "@/types/animation";
import { canTracktHaveAudio } from "@/lib/timeline";
import { canElementHaveAudio } from "@/lib/timeline/element-utils";
import { mergeTimeRanges, type TimeRange } from "@/lib/text-timeline-sync";

export interface MuteKeyframe {
	trackId: string;
	elementId: string;
	propertyPath: AnimationPropertyPath;
	time: number;
	value: number;
	interpolation: AnimationInterpolation;
}

const MIN_MUTE_SPAN_SECONDS = 0.02;

/**
 * Computes volume keyframes that silence every audio-bearing element under
 * the given timeline windows (dubbed narrator ranges), with short linear
 * fades at the edges. Elements outside the windows keep their static volume.
 *
 * Windows are merged first so recover/duck ramps of neighboring windows
 * cannot interleave. Keyframe times are element-local (timeline time minus
 * element start), matching the animation system's convention.
 */
export function computeMuteKeyframes({
	tracks,
	windows,
	excludeTrackIds,
	fadeSeconds = 0.08,
}: {
	tracks: TimelineTrack[];
	windows: TimeRange[];
	excludeTrackIds: Set<string>;
	fadeSeconds?: number;
}): MuteKeyframe[] {
	const merged = mergeTimeRanges(windows, fadeSeconds * 2 + 0.01);
	if (merged.length === 0) return [];

	const keyframes: MuteKeyframe[] = [];

	for (const track of tracks) {
		if (excludeTrackIds.has(track.id)) continue;
		if (!canTracktHaveAudio(track)) continue;
		if (track.muted) continue;

		for (const element of track.elements) {
			if (!canElementHaveAudio(element)) continue;
			if ("muted" in element && element.muted) continue;
			if (element.duration <= 0) continue;

			const baseVolume =
				"volume" in element ? ((element as { volume?: number }).volume ?? 1) : 1;
			if (baseVolume === 0) continue;

			const elementStart = element.startTime;
			const elementEnd = element.startTime + element.duration;

			for (const window of merged) {
				const overlapStart = Math.max(window.start, elementStart);
				const overlapEnd = Math.min(window.end, elementEnd);
				if (overlapEnd - overlapStart < MIN_MUTE_SPAN_SECONDS) continue;

				const muteStartLocal = overlapStart - elementStart;
				const muteEndLocal = overlapEnd - elementStart;

				const push = (time: number, value: number) => {
					keyframes.push({
						trackId: track.id,
						elementId: element.id,
						propertyPath: "volume",
						time,
						value,
						interpolation: "linear",
					});
				};

				const preFadeTime = muteStartLocal - fadeSeconds;
				if (preFadeTime > MIN_MUTE_SPAN_SECONDS) {
					push(preFadeTime, baseVolume);
					push(muteStartLocal, 0);
				} else {
					// Window reaches (or precedes) the element start: begin muted.
					push(0, 0);
				}

				const postFadeTime = muteEndLocal + fadeSeconds;
				if (postFadeTime < element.duration - MIN_MUTE_SPAN_SECONDS) {
					push(muteEndLocal, 0);
					push(postFadeTime, baseVolume);
				} else {
					// Window reaches the element end: stay muted to the edge.
					push(Math.min(muteEndLocal, element.duration), 0);
				}
			}
		}
	}

	return keyframes;
}
