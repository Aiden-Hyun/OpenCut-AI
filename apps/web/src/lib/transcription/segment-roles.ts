import type { SegmentRole, TranscriptionSegment } from "@/types/ai";

export interface RoleClassification {
	/** Speaker ID judged to be the narrator, or null when undecidable. */
	narratorSpeaker: string | null;
	/** Proposed role per segment id. */
	roles: Record<number, SegmentRole>;
	/** 0–1: how clearly the narrator stood out from the runner-up speaker. */
	confidence: number;
}

interface SpeakerStats {
	totalDuration: number;
	firstStart: number;
	lastEnd: number;
	confidenceSum: number;
	wordCount: number;
}

/**
 * Proposes narrator/field roles from diarized transcript segments.
 *
 * The narrator of a voiceover-style video is the speaker who (a) talks the
 * most overall, (b) appears spread across the whole timeline rather than in
 * one cluster, and (c) tends to transcribe with higher word confidence
 * (studio audio vs. field audio). Segments from every other speaker — and
 * segments diarization failed to label — are proposed as "field", which is
 * the safe default: a mislabeled field segment would get muted and dubbed,
 * while a mislabeled narrator segment merely keeps its original audio.
 */
export function proposeSegmentRoles({
	segments,
}: {
	segments: TranscriptionSegment[];
}): RoleClassification {
	const bySpeaker = new Map<string, SpeakerStats>();
	let timelineStart = Number.POSITIVE_INFINITY;
	let timelineEnd = Number.NEGATIVE_INFINITY;

	for (const segment of segments) {
		timelineStart = Math.min(timelineStart, segment.start);
		timelineEnd = Math.max(timelineEnd, segment.end);
		if (!segment.speaker) continue;

		const stats = bySpeaker.get(segment.speaker) ?? {
			totalDuration: 0,
			firstStart: segment.start,
			lastEnd: segment.end,
			confidenceSum: 0,
			wordCount: 0,
		};
		stats.totalDuration += Math.max(0, segment.end - segment.start);
		stats.firstStart = Math.min(stats.firstStart, segment.start);
		stats.lastEnd = Math.max(stats.lastEnd, segment.end);
		for (const word of segment.words ?? []) {
			stats.confidenceSum += word.confidence ?? 0;
			stats.wordCount++;
		}
		bySpeaker.set(segment.speaker, stats);
	}

	if (bySpeaker.size === 0) {
		return { narratorSpeaker: null, roles: {}, confidence: 0 };
	}

	const timelineSpan = Math.max(timelineEnd - timelineStart, 1e-6);
	const scores = new Map<string, number>();
	for (const [speaker, stats] of bySpeaker) {
		const spread = Math.min(
			1,
			Math.max(0, (stats.lastEnd - stats.firstStart) / timelineSpan),
		);
		const avgConfidence =
			stats.wordCount > 0 ? stats.confidenceSum / stats.wordCount : 0.5;
		const score =
			stats.totalDuration * (0.5 + 0.5 * spread) * (0.75 + 0.25 * avgConfidence);
		scores.set(speaker, score);
	}

	const ranked = [...scores.entries()].sort((a, b) => b[1] - a[1]);
	const [narratorSpeaker, topScore] = ranked[0];
	const runnerUpScore = ranked[1]?.[1] ?? 0;
	const confidence =
		topScore > 0 ? Math.min(1, (topScore - runnerUpScore) / topScore) : 0;

	const roles: Record<number, SegmentRole> = {};
	for (const segment of segments) {
		roles[segment.id] =
			segment.speaker === narratorSpeaker ? "narrator" : "field";
	}

	return { narratorSpeaker, roles, confidence };
}
