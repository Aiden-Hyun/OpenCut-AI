import type { NumberAnimationChannel } from "@/types/animation";
import { normalizeChannel } from "./interpolation";

/**
 * Creates a monotonic sampler over a volume keyframe channel. Matches
 * getNumberChannelValueAtTime semantics (segment interpolation is governed
 * by the left keyframe; values clamp to the first/last keyframe outside the
 * keyframe range) but walks a cursor instead of scanning, so it can be
 * called once per audio sample. Times passed to the returned function must
 * be non-decreasing.
 */
export function createVolumeSampler({
	channel,
	fallbackValue,
}: {
	channel: NumberAnimationChannel | undefined;
	fallbackValue: number;
}): (localTime: number) => number {
	if (!channel || channel.keyframes.length === 0) {
		return () => fallbackValue;
	}

	const keyframes = normalizeChannel({ channel }).keyframes;
	const lastIndex = keyframes.length - 1;
	let cursor = 0;

	return (localTime: number): number => {
		if (localTime <= keyframes[0].time) {
			return keyframes[0].value;
		}
		if (localTime >= keyframes[lastIndex].time) {
			return keyframes[lastIndex].value;
		}

		while (cursor < lastIndex - 1 && localTime >= keyframes[cursor + 1].time) {
			cursor++;
		}

		const left = keyframes[cursor];
		const right = keyframes[cursor + 1];
		if (left.interpolation === "hold") {
			return left.value;
		}

		const span = right.time - left.time;
		if (span <= 0) {
			return right.value;
		}

		const progress = (localTime - left.time) / span;
		return left.value + (right.value - left.value) * progress;
	};
}

/**
 * Schedules Web Audio automation on a gain param for the slice of a volume
 * channel covering [fromLocalTime, toLocalTime] (element-local seconds).
 * Keyframe values replace the element's base volume (same semantics as
 * resolveVolumeAtTime); gainMultiplier is applied on top of every value.
 *
 * contextTimeAtFrom is the AudioContext time at which fromLocalTime plays.
 * Events that would land in the past are clamped to `now`.
 */
export function scheduleVolumeAutomation({
	param,
	channel,
	fromLocalTime,
	toLocalTime,
	contextTimeAtFrom,
	now,
	gainMultiplier = 1,
	playbackRate = 1,
}: {
	param: AudioParam;
	channel: NumberAnimationChannel;
	fromLocalTime: number;
	toLocalTime: number;
	contextTimeAtFrom: number;
	now: number;
	gainMultiplier?: number;
	playbackRate?: number;
}): void {
	const keyframes = normalizeChannel({ channel }).keyframes;
	if (keyframes.length === 0) return;

	const rate = playbackRate > 0 ? playbackRate : 1;
	const toContextTime = (localTime: number): number =>
		Math.max(now, contextTimeAtFrom + (localTime - fromLocalTime) / rate);

	const sampler = createVolumeSampler({ channel, fallbackValue: keyframes[0].value });
	const initialValue = sampler(fromLocalTime) * gainMultiplier;
	param.setValueAtTime(initialValue, toContextTime(fromLocalTime));

	let previous: (typeof keyframes)[number] | undefined;
	for (const keyframe of keyframes) {
		if (keyframe.time <= fromLocalTime) {
			previous = keyframe;
			continue;
		}
		if (keyframe.time > toLocalTime) {
			// One event past the window keeps a ramp in progress accurate up to
			// the window edge; sources stop at the edge so overshoot is inaudible.
			if (previous && previous.interpolation === "linear") {
				param.linearRampToValueAtTime(
					keyframe.value * gainMultiplier,
					toContextTime(keyframe.time),
				);
			}
			break;
		}

		const value = keyframe.value * gainMultiplier;
		const time = toContextTime(keyframe.time);
		if (previous && previous.interpolation === "linear") {
			param.linearRampToValueAtTime(value, time);
		} else {
			param.setValueAtTime(value, time);
		}
		previous = keyframe;
	}
}
