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
 * A narrator delivers uninterrupted monologue: their segments cluster in
 * solo blocks (often opening the video), while field speakers interleave in
 * rapid back-and-forth dialogue. Narration is also studio-quality audio, so
 * it transcribes with markedly higher word confidence than bodycam speech.
 * Scoring therefore favors (a) low interleaving with other speakers,
 * (b) high transcription confidence, (c) speaking time, and (d) opening
 * the video. Raw duration alone is deliberately NOT decisive — in bodycam
 * footage the officer often out-talks the narrator.
 *
 * Segments from every other speaker — and segments diarization failed to
 * label — are proposed as "field", the safe default: a mislabeled field
 * segment would get muted and dubbed, while a mislabeled narrator segment
 * merely keeps its original audio.
 */
export function proposeSegmentRoles({
	segments,
}: {
	segments: TranscriptionSegment[];
}): RoleClassification {
	const ordered = [...segments].sort((a, b) => a.start - b.start);
	const bySpeaker = new Map<
		string,
		SpeakerStats & { neighborCount: number; interleavedCount: number }
	>();

	for (let i = 0; i < ordered.length; i++) {
		const segment = ordered[i];
		if (!segment.speaker) continue;

		const stats = bySpeaker.get(segment.speaker) ?? {
			totalDuration: 0,
			firstStart: segment.start,
			lastEnd: segment.end,
			confidenceSum: 0,
			wordCount: 0,
			neighborCount: 0,
			interleavedCount: 0,
		};
		stats.totalDuration += Math.max(0, segment.end - segment.start);
		stats.firstStart = Math.min(stats.firstStart, segment.start);
		stats.lastEnd = Math.max(stats.lastEnd, segment.end);
		for (const word of segment.words ?? []) {
			stats.confidenceSum += word.confidence ?? 0;
			stats.wordCount++;
		}
		// Interleaving: how often this speaker's segments border a different
		// labeled speaker. Dialogue participants alternate constantly; a
		// narrator's segments sit in contiguous solo blocks.
		for (const neighbor of [ordered[i - 1], ordered[i + 1]]) {
			if (!neighbor?.speaker) continue;
			stats.neighborCount++;
			if (neighbor.speaker !== segment.speaker) stats.interleavedCount++;
		}
		bySpeaker.set(segment.speaker, stats);
	}

	if (bySpeaker.size === 0) {
		return { narratorSpeaker: null, roles: {}, confidence: 0 };
	}

	const firstSpeaker = ordered.find((seg) => seg.speaker)?.speaker;
	const scores = new Map<string, number>();
	for (const [speaker, stats] of bySpeaker) {
		const avgConfidence =
			stats.wordCount > 0 ? stats.confidenceSum / stats.wordCount : 0.5;
		const interleaveRatio =
			stats.neighborCount > 0
				? stats.interleavedCount / stats.neighborCount
				: 0;
		// Sub-linear duration so sheer talk time cannot outvote structure.
		const durationTerm = Math.sqrt(Math.max(stats.totalDuration, 1e-6));
		const monologueTerm = (1 - interleaveRatio) ** 2;
		const confidenceTerm = avgConfidence ** 2;
		const opensVideo = speaker === firstSpeaker ? 1.25 : 1;
		const score =
			durationTerm * (0.15 + 0.85 * monologueTerm) * (0.25 + 0.75 * confidenceTerm) * opensVideo;
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
