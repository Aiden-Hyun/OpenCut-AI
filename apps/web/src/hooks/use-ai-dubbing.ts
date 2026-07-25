import { useCallback, useState } from "react";
import { useEditor } from "@/hooks/use-editor";
import { useTranscriptStore } from "@/stores/transcript-store";
import { useBackgroundTasksStore } from "@/stores/background-tasks-store";
import { aiClient } from "@/lib/ai-client";
import { generateUUID } from "@/utils/id";
import {
	SARVAM_TTS_SUPPORTED_CODES,
	toSarvamCode,
} from "@/constants/sarvam-constants";
import { computeMuteKeyframes } from "@/lib/timeline/dub-replacement";
import type { TimeRange } from "@/lib/text-timeline-sync";
import { toast } from "sonner";

export type DubbingEngine = "sarvam" | "smallest" | "local";

export type DubbingScope = "narrator" | "all";

export interface DubbingOptions {
	targetLanguage: string;
	engine: DubbingEngine;
	voiceId: string;
	pace?: number;
	segmentIndices?: number[];
	/**
	 * Which segments to dub when segmentIndices is not given:
	 * "narrator" dubs only segments with role === "narrator" (falling back to
	 * all segments when no roles are assigned); "all" dubs everything.
	 */
	scope?: DubbingScope;
	/**
	 * Mute the original audio under each dubbed range (default true). Field
	 * segments and everything outside dubbed ranges keep their original audio.
	 */
	replaceOriginal?: boolean;
	/** Reference audio path for XTTS voice cloning (local engine only). */
	speakerWav?: string;
}

export interface DubbingProgress {
	currentSegment: number;
	totalSegments: number;
	currentText: string;
	phase: "translating" | "generating" | "placing" | "done";
}

/** Dub may run this much longer than its slot before a speed-up retry. */
const FIT_TOLERANCE = 1.08;
/** Upper bound for corrective TTS speed-up — beyond this it sounds rushed. */
const MAX_FIT_SPEED = 1.35;

function getAudioDuration(url: string): Promise<number> {
	return new Promise((resolve) => {
		const audio = new Audio(url);
		audio.addEventListener("loadedmetadata", () => {
			resolve(audio.duration);
		});
		audio.addEventListener("error", () => {
			resolve(5);
		});
	});
}

export function useAIDubbing() {
	const editor = useEditor();
	const segments = useTranscriptStore((s) => s.segments);
	const language = useTranscriptStore((s) => s.language);
	const addTask = useBackgroundTasksStore((s) => s.addTask);
	const updateTask = useBackgroundTasksStore((s) => s.updateTask);

	const [isDubbing, setIsDubbing] = useState(false);
	const [progress, setProgress] = useState<DubbingProgress | null>(null);

	const runDubbing = useCallback(
		async (options: DubbingOptions) => {
			const scope = options.scope ?? "narrator";
			const narratorSegments = segments.filter(
				(seg) => seg.role === "narrator",
			);
			const targetSegments = options.segmentIndices
				? options.segmentIndices.map((i) => segments[i]).filter(Boolean)
				: scope === "narrator" && narratorSegments.length > 0
					? narratorSegments
					: segments;

			if (targetSegments.length === 0) {
				toast.error("No segments to dub");
				return;
			}

			setIsDubbing(true);
			const taskId = `dubbing-${Date.now()}`;
			addTask({
				id: taskId,
				type: "dubbing",
				label: `Dubbing to ${options.targetLanguage}`,
				progress: `0/${targetSegments.length} segments`,
			});

			const totalSegments = targetSegments.length;
			let completed = 0;

			// Dubs get their own named track so reruns reuse it and the
			// mute pass can exclude it.
			const dubTrackName = `Dub (${options.targetLanguage})`;
			const tracks = editor.timeline.getTracks();
			let trackId = tracks.find(
				(t) => t.type === "audio" && t.name === dubTrackName,
			)?.id;
			if (!trackId) {
				trackId = editor.timeline.addTrack({ type: "audio" });
				editor.timeline.renameTrack({ trackId, name: dubTrackName });
			}

			const dubbedRanges: TimeRange[] = [];

			try {
				for (const seg of targetSegments) {
					setProgress({
						currentSegment: completed + 1,
						totalSegments,
						currentText: seg.text.slice(0, 50),
						phase: "translating",
					});

					let translatedText: string;
					const isSarvamLang = SARVAM_TTS_SUPPORTED_CODES.has(
						options.targetLanguage,
					);

					if (isSarvamLang && language !== options.targetLanguage) {
						const srcCode = toSarvamCode(language) ?? "en-IN";
						const tgtCode = toSarvamCode(options.targetLanguage) ?? "hi-IN";
						const result = await aiClient.sarvamTranslate(
							seg.text,
							srcCode,
							tgtCode,
						);
						translatedText = result.translated_text;
					} else if (language !== options.targetLanguage) {
						translatedText = await aiClient.translateText(
							seg.text,
							options.targetLanguage,
						);
					} else {
						translatedText = seg.text;
					}

					setProgress({
						currentSegment: completed + 1,
						totalSegments,
						currentText: translatedText.slice(0, 50),
						phase: "generating",
					});

					const isSarvam = options.engine === "sarvam" && isSarvamLang;
					const isSmallest = options.engine === "smallest";
					const slotDuration = Math.max(0.2, seg.end - seg.start);

					const generateLocal = async (speed: number): Promise<Blob> =>
						aiClient.generateSpeechBlob({
							text: translatedText,
							language: options.targetLanguage,
							speaker: options.voiceId,
							speakerWav: options.speakerWav,
							speed,
						});

					let audioBlob: Blob;
					if (isSarvam) {
						const sarvamCode =
							toSarvamCode(options.targetLanguage) ?? "hi-IN";
						audioBlob = await aiClient.sarvamTTS(
							translatedText,
							sarvamCode,
							options.voiceId,
							options.pace ?? 1.0,
						);
					} else if (isSmallest) {
						audioBlob = await aiClient.smallestTTS(
							translatedText,
							options.voiceId,
							options.targetLanguage,
							options.pace ?? 1.0,
						);
					} else {
						audioBlob = await generateLocal(options.pace ?? 1.0);
					}

					const ext = isSarvam || isSmallest ? "mp3" : "wav";
					const mimeType = isSarvam || isSmallest ? "audio/mpeg" : "audio/wav";

					let file = new File([audioBlob], `dub_${generateUUID()}.${ext}`, {
						type: mimeType,
					});
					let audioUrl = URL.createObjectURL(file);
					let duration = await getAudioDuration(audioUrl);

					// Fit-to-slot: if the dub overflows its slot, retry once at a
					// proportionally higher speed (local XTTS engine only).
					if (
						!isSarvam &&
						!isSmallest &&
						duration > slotDuration * FIT_TOLERANCE
					) {
						const fitSpeed = Math.min(
							MAX_FIT_SPEED,
							(duration / slotDuration) * (options.pace ?? 1.0),
						);
						try {
							const refitBlob = await generateLocal(fitSpeed);
							URL.revokeObjectURL(audioUrl);
							file = new File([refitBlob], `dub_${generateUUID()}.${ext}`, {
								type: mimeType,
							});
							audioUrl = URL.createObjectURL(file);
							duration = await getAudioDuration(audioUrl);
						} catch {
							// Keep the first take if the refit attempt fails.
						}
					}

					setProgress({
						currentSegment: completed + 1,
						totalSegments,
						currentText: translatedText.slice(0, 50),
						phase: "placing",
					});

					const dubDuration = duration || slotDuration;
					editor.timeline.insertElement({
						placement: { mode: "explicit", trackId },
						element: {
							type: "audio",
							sourceType: "library",
							sourceUrl: audioUrl,
							name: `Dub [${options.targetLanguage}]: ${seg.text.slice(0, 20)}...`,
							startTime: seg.start,
							duration: dubDuration,
							trimStart: 0,
							trimEnd: 0,
							sourceDuration: dubDuration,
							volume: 1,
						},
					});
					dubbedRanges.push({
						start: seg.start,
						end: seg.start + Math.max(dubDuration, slotDuration),
					});

					completed++;
					updateTask(taskId, {
						progress: `${completed}/${totalSegments} segments`,
					});
					setProgress({
						currentSegment: completed,
						totalSegments,
						currentText: translatedText.slice(0, 50),
						phase: completed === totalSegments ? "done" : "translating",
					});
				}

				// Replace, don't layer: silence the original audio exactly under
				// the dubbed ranges. Field audio outside the ranges is untouched.
				if (options.replaceOriginal !== false && dubbedRanges.length > 0) {
					const muteKeyframes = computeMuteKeyframes({
						tracks: editor.timeline.getTracks(),
						windows: dubbedRanges,
						excludeTrackIds: new Set([trackId]),
					});
					if (muteKeyframes.length > 0) {
						editor.timeline.upsertKeyframes({ keyframes: muteKeyframes });
					}
				}

				updateTask(taskId, {
					status: "completed",
					progress: `${totalSegments}/${totalSegments} segments`,
					completedAt: Date.now(),
				});

				toast.success(
					`Dubbing complete: ${totalSegments} segments in ${options.targetLanguage}`,
				);
			} catch (error) {
				const message =
					error instanceof Error ? error.message : "Dubbing failed";
				updateTask(taskId, {
					status: "error",
					error: message,
					completedAt: Date.now(),
				});
				toast.error("Dubbing failed", { description: message });
			} finally {
				setIsDubbing(false);
				setProgress(null);
			}
		},
		[editor, segments, language, addTask, updateTask],
	);

	return { runDubbing, isDubbing, progress };
}
