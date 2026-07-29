import { Button } from "@/components/ui/button";
import { PanelView } from "@/components/editor/panels/assets/views/base-view";
import {
	Select,
	SelectContent,
	SelectGroup,
	SelectItem,
	SelectLabel,
	SelectTrigger,
	SelectValue,
} from "@/components/ui/select";
import { useState, useRef, useEffect } from "react";
import { useEditor } from "@/hooks/use-editor";
import { DEFAULT_TEXT_ELEMENT } from "@/constants/text-constants";
import { WHISPER_LANGUAGES } from "@/constants/transcription-constants";
import { LANGUAGES } from "@/constants/language-constants";
import {
	SARVAM_STT_LANGUAGES,
	SARVAM_LANGUAGE_MAP,
	SARVAM_SUPPORTED_CODES,
	isSarvamSTTSupported,
} from "@/constants/sarvam-constants";
import { SMALLEST_STT_LANGUAGES } from "@/constants/smallest-constants";
import type { TranscriptionLanguage, TranscriptionEngine } from "@/types/transcription";

import { Spinner } from "@/components/ui/spinner";
import { Checkbox } from "@/components/ui/checkbox";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import { cn } from "@/utils/ui";
import { useTranscriptStore } from "@/stores/transcript-store";
import { usePreviewStore } from "@/stores/preview-store";
import { getElementsAtTime, hasMediaId } from "@/lib/timeline";
import { processMediaAssets } from "@/lib/media/processing";
import { extractSpan, SpanExtractionError } from "@/lib/media/extract-span";
import { toast } from "sonner";
import { aiClient } from "@/lib/ai-client";
import type {
	SubtitleRegion,
	SubtitleScheduleEntry,
	SubtitleTimelineDetection,
} from "@/lib/ai-client";
import { formatTimeCode } from "@/lib/time";
import type { TimelineElement } from "@/types/timeline";
import { useBackgroundTasksStore } from "@/stores/background-tasks-store";

interface SubtitleTrackInfo {
	trackId: string;
	language: string;
}

/** Video track the cleaned spans are placed on. Reused across runs so a
 *  second pass does not scatter overlays over several tracks. */
const CLEAN_TRACK_NAME = "Subtitle removal";

/** Poll interval for inpaint job status, in ms. */
const INPAINT_POLL_MS = 3000;
/** Consecutive status-poll failures tolerated before giving up. */
const MAX_POLL_FAILURES = 5;

export function Captions() {
	const [selectedEngine, setSelectedEngine] = useState<TranscriptionEngine>("whisper");
	const [selectedLanguage, setSelectedLanguage] =
		useState<TranscriptionLanguage>("auto");
	const [isProcessing, setIsProcessing] = useState(false);
	const [processingStep, setProcessingStep] = useState("");
	const [error, setError] = useState<string | null>(null);
	const [subtitleTracks, setSubtitleTracks] = useState<SubtitleTrackInfo[]>([]);
	const [splitOnTranscribe, setSplitOnTranscribe] = useState(false);
	const [translateLanguage, setTranslateLanguage] = useState("es");
	const [isTranslating, setIsTranslating] = useState(false);
	const [translatingStep, setTranslatingStep] = useState("");
	const [inpaintRegion, setInpaintRegion] = useState({
		x1: "0.05",
		y1: "0.75",
		x2: "0.95",
		y2: "0.98",
	});
	const [showRegionOnPreview, setShowRegionOnPreview] = useState(true);
	const [isInpainting, setIsInpainting] = useState(false);
	const [inpaintProgress, setInpaintProgress] = useState(0);
	const [inpaintStep, setInpaintStep] = useState("");
	const [inpaintError, setInpaintError] = useState<string | null>(null);
	const [inpaintJobId, setInpaintJobId] = useState<string | null>(null);
	const [isCancellingInpaint, setIsCancellingInpaint] = useState(false);
	const [isDetectingRegion, setIsDetectingRegion] = useState(false);
	// Detected timeline of burned-in regions. While it is set the panel is in
	// schedule mode: the reviewable segment list replaces the manual region
	// inputs and the job runs off the enabled segments. "Clear detection"
	// drops it and returns to the manual single-region flow.
	const [detection, setDetection] = useState<SubtitleTimelineDetection | null>(
		null,
	);
	const [enabledSegments, setEnabledSegments] = useState<boolean[]>([]);
	// Segment the playhead currently sits in — drives the preview boxes and
	// the highlighted row.
	const [activeSegment, setActiveSegment] = useState<number | null>(null);
	// Set by the Stop button; the segment loop and the polling loop both exit
	// on their next tick.
	const inpaintStopRequestedRef = useRef(false);
	// Aborts an in-flight client-side span extraction, which is the one phase
	// that does not poll and so cannot notice the stop flag on its own.
	const inpaintAbortRef = useRef<AbortController | null>(null);
	const containerRef = useRef<HTMLDivElement>(null);
	const segments = useTranscriptStore((s) => s.segments);
	const setInpaintRegionOverlays = usePreviewStore(
		(s) => s.setInpaintRegionOverlays,
	);
	const editor = useEditor();

	// Mirror the removal regions onto the preview as highlight boxes: the
	// active segment's regions in schedule mode, the manual box otherwise.
	useEffect(() => {
		if (!showRegionOnPreview) {
			setInpaintRegionOverlays([]);
			return;
		}
		if (detection) {
			const segment =
				activeSegment === null ? undefined : detection.segments[activeSegment];
			setInpaintRegionOverlays(segment?.regions ?? []);
			return;
		}
		const region = {
			x1: Number.parseFloat(inpaintRegion.x1),
			y1: Number.parseFloat(inpaintRegion.y1),
			x2: Number.parseFloat(inpaintRegion.x2),
			y2: Number.parseFloat(inpaintRegion.y2),
		};
		const isValid =
			[region.x1, region.y1, region.x2, region.y2].every(
				(v) => !Number.isNaN(v) && v >= 0 && v <= 1,
			) &&
			region.x2 > region.x1 &&
			region.y2 > region.y1;
		setInpaintRegionOverlays(isValid ? [region] : []);
	}, [
		showRegionOnPreview,
		inpaintRegion,
		detection,
		activeSegment,
		setInpaintRegionOverlays,
	]);

	// Follow the playhead while a detection is loaded so scrubbing shows the
	// boxes for the stretch on screen. State only changes when the segment
	// does, so this does not re-render the panel every frame.
	useEffect(() => {
		if (!detection) {
			setActiveSegment(null);
			return;
		}
		let running = true;
		let frame = 0;
		const tick = () => {
			if (!running) return;
			const time = editor.playback.getCurrentTime();
			const index = detection.segments.findIndex(
				(segment) => time >= segment.start && time < segment.end,
			);
			setActiveSegment(index === -1 ? null : index);
			frame = requestAnimationFrame(tick);
		};
		frame = requestAnimationFrame(tick);
		return () => {
			running = false;
			cancelAnimationFrame(frame);
		};
	}, [detection, editor]);

	// Remove the highlight when leaving the Captions view
	useEffect(() => {
		return () => {
			usePreviewStore.getState().setInpaintRegionOverlays([]);
		};
	}, []);

	// Determine which languages to show based on engine
	const availableLanguages = selectedEngine === "sarvam"
		? SARVAM_STT_LANGUAGES
		: selectedEngine === "smallest"
			? SMALLEST_STT_LANGUAGES
			: WHISPER_LANGUAGES;

	// Filter out tracks that no longer exist on the timeline (user may have deleted them)
	const timelineTracks = editor.timeline.getTracks();
	const timelineTrackIds = new Set(timelineTracks.map((t) => t.id));
	const activeSubtitleTracks = subtitleTracks.filter((t) =>
		timelineTrackIds.has(t.trackId),
	);

	// Sync state if tracks were removed externally
	if (activeSubtitleTracks.length !== subtitleTracks.length) {
		// Use a microtask to avoid setState during render
		queueMicrotask(() => setSubtitleTracks(activeSubtitleTracks));
	}

	/** Determine the effective engine for a given language code */
	const getEffectiveEngine = (langCode: string): TranscriptionEngine => {
		if (selectedEngine === "sarvam") return "sarvam";
		if (selectedEngine === "smallest") return "smallest";
		// Auto-switch to Sarvam if an Indian language is explicitly selected with Whisper
		if (langCode !== "auto" && isSarvamSTTSupported(langCode) && !WHISPER_LANGUAGES.some(l => l.code === langCode)) {
			return "sarvam";
		}
		return "whisper";
	};

	const handleGenerateTranscript = async () => {
		const taskId = `transcription-${Date.now()}`;
		const bgTasks = useBackgroundTasksStore.getState();

		try {
			setIsProcessing(true);
			setError(null);

			const engine = getEffectiveEngine(selectedLanguage);
			const engineLabel = engine === "sarvam" ? "Sarvam AI" : engine === "smallest" ? "Smallest AI" : "Whisper";

			bgTasks.addTask({
				id: taskId,
				type: "transcription",
				label: `Transcription (${engineLabel})`,
				progress: "Starting...",
			});

			// Remove existing subtitle tracks before re-transcribing
			for (const track of activeSubtitleTracks) {
				try {
					editor.timeline.removeTrack({ trackId: track.trackId });
				} catch {
					// Track may already be gone
				}
			}
			setSubtitleTracks([]);

			// Find the media file from the timeline
			setProcessingStep("Finding media...");
			bgTasks.updateTask(taskId, { progress: "Finding media..." });
			const tracks = editor.timeline.getTracks();
			let foundMediaId: string | null = null;

			for (const track of tracks) {
				for (const element of track.elements) {
					if (
						(track.type === "video" || track.type === "audio") &&
						hasMediaId(element as TimelineElement)
					) {
						foundMediaId = (element as TimelineElement & { mediaId: string }).mediaId;
						break;
					}
				}
				if (foundMediaId) break;
			}

			if (!foundMediaId) {
				setError("No video or audio found on the timeline. Import a file first.");
				return;
			}

			const mediaAsset = editor.media
				.getAssets()
				.find((asset) => asset.id === foundMediaId);

			if (!mediaAsset?.file) {
				setError("Cannot access the media file for transcription.");
				return;
			}

			// Send to appropriate transcription service
			setProcessingStep(`Transcribing via ${engineLabel}...`);
			bgTasks.updateTask(taskId, { progress: `Transcribing via ${engineLabel}...` });

			// Ensure the file has a proper extension — the backend rejects files without one
			const mimeToExt: Record<string, string> = {
				"video/mp4": ".mp4",
				"video/webm": ".webm",
				"video/quicktime": ".mov",
				"video/x-matroska": ".mkv",
				"video/avi": ".avi",
				"audio/mpeg": ".mp3",
				"audio/wav": ".wav",
				"audio/x-wav": ".wav",
				"audio/ogg": ".ogg",
				"audio/flac": ".flac",
				"audio/aac": ".aac",
				"audio/mp4": ".m4a",
			};

			let file = mediaAsset.file;
			const fileName = file.name || "";
			const hasExtension = fileName.includes(".") && fileName.split(".").pop()!.length > 0;

			if (!hasExtension) {
				const ext = mimeToExt[file.type] || ".mp4";
				const newName = fileName ? `${fileName}${ext}` : `media${ext}`;
				file = new File([file], newName, { type: file.type || "video/mp4" });
			}

			let result;
			if (engine === "sarvam") {
				// Use Sarvam AI for Indian languages
				const sarvamLangCode = selectedLanguage === "auto"
					? undefined
					: SARVAM_LANGUAGE_MAP[selectedLanguage] || undefined;
				result = await aiClient.sarvamTranscribe(file, sarvamLangCode);
			} else if (engine === "smallest") {
				// Use Smallest AI Pulse for multilingual STT
				const language = selectedLanguage === "auto" ? "en" : selectedLanguage;
				result = await aiClient.smallestTranscribe(file, language);
			} else {
				// Use Whisper (local)
				const language = selectedLanguage === "auto" ? undefined : selectedLanguage;
				result = await aiClient.transcribe(file, language);
			}

			setProcessingStep("Processing segments...");
			bgTasks.updateTask(taskId, { progress: "Processing segments..." });

			const timelineDuration = editor.timeline.getTotalDuration();

			// Filter out hallucinated segments beyond the actual duration
			// For Sarvam results, also allow Indic Unicode ranges through
			const validSegments = result.segments.filter((seg) => {
				if (seg.start >= timelineDuration) return false;
				// Keep Latin, Cyrillic, CJK, Kana, Devanagari, Bengali, Gurmukhi, Gujarati,
			// Oriya, Tamil, Telugu, Kannada, Malayalam, Sinhala, Arabic/Nastaliq (Urdu),
			// Meitei (Manipuri), Ol Chiki (Santali)
			const cleanText = seg.text.replace(/[^a-zA-Z0-9\u00C0-\u024F\u0400-\u04FF\u4E00-\u9FFF\u3040-\u309F\u30A0-\u30FF\u0900-\u097F\u0980-\u09FF\u0A00-\u0A7F\u0A80-\u0AFF\u0B00-\u0B7F\u0B80-\u0BFF\u0C00-\u0C7F\u0C80-\u0CFF\u0D00-\u0D7F\u0D80-\u0DFF\u0600-\u06FF\u0750-\u077F\uFB50-\uFDFF\uFE70-\uFEFF\uABC0-\uABFF\u1C50-\u1C7F]/g, "").trim();
				if (cleanText.length === 0) return false;
				if (seg.end <= seg.start) return false;
				return true;
			}).map((seg) => ({
				...seg,
				end: Math.min(seg.end, timelineDuration),
			}));

			if (validSegments.length === 0) {
				setError("No speech detected in the video. Try a different language or check that the video has audio.");
				return;
			}

			// Populate transcript store — segments already have word-level detail from backend
			const transcriptSegments = validSegments.map((seg, index) => ({
				id: index,
				text: seg.text,
				start: seg.start,
				end: seg.end,
				words: seg.words && seg.words.length > 0
					? seg.words.map((w) => ({
						word: w.word,
						start: w.start,
						end: w.end,
						confidence: w.confidence,
					}))
					: seg.text.trim().split(/\s+/).map((word, wordIndex, arr) => {
						const segDuration = seg.end - seg.start;
						const wordDuration = segDuration / arr.length;
						return {
							word,
							start: seg.start + wordIndex * wordDuration,
							end: seg.start + (wordIndex + 1) * wordDuration,
							confidence: 0.9,
						};
					}),
			}));

			useTranscriptStore.getState().setSegments(transcriptSegments);
			useTranscriptStore.getState().setLanguage(result.language ?? "en");

			// ── Auto Speaker Diarization + Emotion Detection ──
			// Run both in parallel: speaker labels and emotion annotations.
			let speakerChangeTimes: number[] = [];
			setProcessingStep("Detecting speakers & emotions...");
			bgTasks.updateTask(taskId, { progress: "Detecting speakers & emotions..." });

			const speakerPromise = aiClient.analyzeSpeakers(file).catch((err) => {
				console.warn("Speaker diarization failed:", err);
				return null;
			});
			const emotionPromise = aiClient.analyzeEmotions(file).catch((err) => {
				console.warn("Emotion detection failed:", err);
				return null;
			});

			const [speakerResult, emotionResult] = await Promise.all([speakerPromise, emotionPromise]);

			if (speakerResult && speakerResult.segments.length > 0) {
				useTranscriptStore.getState().applySpeakerDiarization(speakerResult.segments);

				// Collect speaker change boundaries for auto-cuts
				for (let i = 1; i < speakerResult.segments.length; i++) {
					const prev = speakerResult.segments[i - 1];
					const curr = speakerResult.segments[i];
					if (prev.speaker !== curr.speaker) {
						const boundary = curr.start;
						if (boundary > 0 && boundary < timelineDuration) {
							speakerChangeTimes.push(boundary);
						}
					}
				}

				const numSpeakers = speakerResult.num_speakers;
				if (numSpeakers > 1) {
					toast.success(`Detected ${numSpeakers} speakers`, {
						description: `Method: ${speakerResult.method}. Click speaker names in the transcript to rename them.`,
					});
				}
			}

			if (emotionResult && emotionResult.emotions.length > 0) {
				useTranscriptStore.getState().setEmotions(emotionResult.emotions);
			}

			// Split video at segment boundaries AND speaker change points.
			// Opt-in: on longer real-world videos this produces 100+ synchronous
			// splits (each with a full timeline re-render) and freezes the tab.
			if (splitOnTranscribe && validSegments.length > 1) {
				try {
					const allTimes = new Set<number>();
					for (const seg of validSegments) {
						if (seg.start > 0 && seg.start < timelineDuration) {
							allTimes.add(seg.start);
						}
						if (seg.end > 0 && seg.end < timelineDuration) {
							allTimes.add(seg.end);
						}
					}
					// Add speaker change boundaries
					for (const t of speakerChangeTimes) {
						allTimes.add(t);
					}

					let splitCount = 0;
					const reversed = [...allTimes].sort((a, b) => b - a);
					for (const time of reversed) {
						const elementsAtTime = getElementsAtTime({
							tracks: editor.timeline.getTracks(),
							time,
						});
						if (elementsAtTime.length > 0) {
							editor.timeline.splitElements({
								elements: elementsAtTime,
								splitTime: time,
							});
							splitCount++;
						}
					}

					if (splitCount > 0) {
						toast.success(`Video split into ${splitCount + 1} segments`, {
							description: "Delete or reorder segments in the transcript panel to edit the video.",
						});
					}
				} catch (splitError) {
					console.error("Failed to split video:", splitError);
				}
			}

			// Auto-separate audio from video so it appears as its own track.
			// Tied to the split toggle: mutating the timeline mid-flow is only
			// needed for the text-based-editing workflow.
			const tracksAfterSplit = splitOnTranscribe
				? editor.timeline.getTracks()
				: [];
			for (const track of tracksAfterSplit) {
				if (track.type !== "video") continue;
				for (const el of track.elements) {
					const videoEl = el as TimelineElement & { mediaId?: string; muted?: boolean };
					if (!videoEl.mediaId || videoEl.muted) continue;

					// Mute the video element and create a matching audio element
					editor.timeline.updateElements({
						updates: [{
							trackId: track.id,
							elementId: el.id,
							updates: { muted: true },
						}],
					});

					editor.timeline.insertElement({
						element: {
							type: "audio",
							sourceType: "upload",
							mediaId: videoEl.mediaId,
							name: `${el.name} (audio)`,
							startTime: el.startTime,
							duration: el.duration,
							trimStart: el.trimStart,
							trimEnd: el.trimEnd,
							sourceDuration: el.sourceDuration,
							volume: 1,
						},
						placement: { mode: "auto" },
					});
				}
			}

			setSubtitleTracks([]);

			toast.success(`Transcription complete (${engineLabel})`, {
				description: `${validSegments.length} segments detected`,
			});

			bgTasks.updateTask(taskId, {
				status: "completed",
				progress: `${validSegments.length} segments`,
				completedAt: Date.now(),
			});
		} catch (err) {
			console.error("Transcription failed:", err);
			const message = err instanceof Error ? err.message : "An unexpected error occurred";
			if (message.includes("Cannot connect") || message.includes("connection_refused")) {
				setError("Cannot connect to AI backend. Make sure it is running (docker compose up -d).");
			} else if (message.includes("Sarvam API key")) {
				setError("Sarvam API key is not configured. Add OPENCUTAI_SARVAM_API_KEY to your environment.");
			} else if (message.includes("Smallest AI API key")) {
				setError("Smallest AI API key is not configured. Add it in Settings > API Keys, or set OPENCUTAI_SMALLEST_API_KEY in your environment.");
			} else {
				setError(message);
			}
			bgTasks.updateTask(taskId, {
				status: "error",
				error: message,
				completedAt: Date.now(),
			});
		} finally {
			setIsProcessing(false);
			setProcessingStep("");
		}
	};

	const addSubtitleTrack = ({
		subtitleSegments,
		languageLabel,
		yOffset = 0.38,
	}: {
		subtitleSegments: {
			text: string;
			start: number;
			end: number;
			words?: { word: string; start: number; end: number }[];
		}[];
		languageLabel: string;
		yOffset?: number;
	}) => {
		const trackId = editor.timeline.addTrack({ type: "text", index: 0 });
		editor.timeline.renameTrack({
			trackId,
			name: `Subs: ${languageLabel}`,
		});
		const canvasSize = editor.project.getActive().settings.canvasSize;
		const subtitleY = canvasSize.height * yOffset;

		for (let i = 0; i < subtitleSegments.length; i++) {
			const seg = subtitleSegments[i];

			// Build word timings relative to the element's local time (0-based)
			const wordTimings = seg.words?.map((w) => ({
				word: w.word,
				start: w.start - seg.start,
				end: w.end - seg.start,
			}));

			editor.timeline.insertElement({
				placement: { mode: "explicit", trackId },
				element: {
					...DEFAULT_TEXT_ELEMENT,
					name: `${languageLabel} ${i + 1}`,
					content: seg.text,
					duration: seg.end - seg.start,
					startTime: seg.start,
					fontSize: 4,
					fontWeight: "bold",
					color: "#ffffff",
					highlightColor: "#FACC15",
					...(wordTimings && wordTimings.length > 0 ? { wordTimings } : {}),
					textAlign: "center",
					background: {
						enabled: true,
						color: "#000000",
						cornerRadius: 4,
						paddingX: 12,
						paddingY: 6,
						offsetX: 0,
						offsetY: 0,
					},
					opacity: 0.95,
					transform: {
						scale: 1,
						position: { x: 0, y: subtitleY },
						rotate: 0,
					},
				},
			});
		}

		return trackId;
	};

	const handleAddSubtitles = () => {
		const currentSegments = useTranscriptStore.getState().segments;
		if (currentSegments.length === 0) return;

		const trackId = addSubtitleTrack({
			subtitleSegments: currentSegments.map((seg) => ({
				text: seg.text,
				start: seg.start,
				end: seg.end,
				words: seg.words,
			})),
			languageLabel: "Subtitle",
		});

		setSubtitleTracks((prev) => [...prev, { trackId, language: "original" }]);
		toast.success("Subtitles added with word highlighting");
	};

	/** Check if we should use Sarvam for translation (Indian language pair).
	 *  Requires at least one side to be an Indian language (not just "en"),
	 *  AND the other side must also be Sarvam-supported.
	 */
	const shouldUseSarvamTranslation = (sourceLang: string, targetLang: string): boolean => {
		const sourceIsSarvam = SARVAM_SUPPORTED_CODES.has(sourceLang);
		const targetIsSarvam = SARVAM_SUPPORTED_CODES.has(targetLang);
		// Both must be Sarvam-supported, and at least one must be a non-English Indian language
		const sourceIsIndian = sourceIsSarvam && sourceLang !== "en";
		const targetIsIndian = targetIsSarvam && targetLang !== "en";
		return (sourceIsIndian || targetIsIndian) && sourceIsSarvam && targetIsSarvam;
	};

	const handleTranslateAndAdd = async () => {
		const currentSegments = useTranscriptStore.getState().segments;
		if (currentSegments.length === 0) return;

		const targetLang = LANGUAGES.find((l) => l.code === translateLanguage);
		if (!targetLang) return;

		const transcriptLang = useTranscriptStore.getState().language;
		const useSarvam = shouldUseSarvamTranslation(transcriptLang, translateLanguage);

		const taskId = `translation-${targetLang.code}-${Date.now()}`;
		const bgTasks = useBackgroundTasksStore.getState();

		setIsTranslating(true);
		setError(null);

		const translationEngine = useSarvam ? "Sarvam AI" : "Local LLM";

		bgTasks.addTask({
			id: taskId,
			type: "translation",
			label: `${targetLang.name} translation (${translationEngine})`,
			progress: "Starting...",
		});

		try {
			const translatedSegments: { text: string; start: number; end: number }[] = [];

			if (useSarvam) {
				// Use Sarvam translation API — translate segment by segment
				const sourceSarvamCode = SARVAM_LANGUAGE_MAP[transcriptLang] || `${transcriptLang}-IN`;
				const targetSarvamCode = SARVAM_LANGUAGE_MAP[translateLanguage] || `${translateLanguage}-IN`;

				for (let i = 0; i < currentSegments.length; i++) {
					const seg = currentSegments[i];
					const stepText = `Translating to ${targetLang.name} via Sarvam... (${i + 1}/${currentSegments.length})`;
					setTranslatingStep(stepText);
					bgTasks.updateTask(taskId, { progress: stepText });

					try {
						const result = await aiClient.sarvamTranslate(
							seg.text,
							sourceSarvamCode,
							targetSarvamCode,
						);
						translatedSegments.push({
							text: result.translated_text || seg.text,
							start: seg.start,
							end: seg.end,
						});
					} catch (translationErr) {
						console.warn(`Sarvam translation failed for segment ${i}, using original:`, translationErr);
						translatedSegments.push({
							text: seg.text,
							start: seg.start,
							end: seg.end,
						});
					}
				}
			} else {
				// Use local LLM (original approach)
				const BATCH_SIZE = 5;

				for (let i = 0; i < currentSegments.length; i += BATCH_SIZE) {
					const batch = currentSegments.slice(i, i + BATCH_SIZE);
					const batchIndex = Math.floor(i / BATCH_SIZE) + 1;
					const totalBatches = Math.ceil(currentSegments.length / BATCH_SIZE);
					const stepText = `Translating to ${targetLang.name}... (${batchIndex}/${totalBatches})`;
					setTranslatingStep(stepText);
					bgTasks.updateTask(taskId, { progress: stepText });

					// Send batch as numbered lines for reliable parsing
					const numberedLines = batch
						.map((seg, idx) => `${idx + 1}. ${seg.text}`)
						.join("\n");

					const translated = await aiClient.translateText(
						numberedLines,
						targetLang.name,
					);

					// Parse response — expect numbered lines back
					const lines = translated
						.split("\n")
						.map((line) => line.replace(/^\d+\.\s*/, "").trim())
						.filter((line) => line.length > 0);

					for (let j = 0; j < batch.length; j++) {
						translatedSegments.push({
							text: lines[j] || batch[j].text,
							start: batch[j].start,
							end: batch[j].end,
						});
					}
				}
			}

			// Place translated subtitles slightly above the original ones
			const yOffset = activeSubtitleTracks.length > 0 ? 0.28 : 0.38;
			const trackId = addSubtitleTrack({
				subtitleSegments: translatedSegments,
				languageLabel: `${targetLang.name}`,
				yOffset,
			});

			setSubtitleTracks((prev) => [
				...prev,
				{ trackId, language: targetLang.code },
			]);

			// Store translation in transcript store for the tabbed panel
			useTranscriptStore.getState().addTranslation({
				languageCode: targetLang.code,
				languageName: targetLang.name,
				segments: translatedSegments.map((seg, idx) => ({
					id: idx,
					text: seg.text,
					start: seg.start,
					end: seg.end,
					words: [],
				})),
			});

			bgTasks.updateTask(taskId, {
				status: "completed",
				progress: `${translatedSegments.length} segments`,
				completedAt: Date.now(),
			});
		} catch (err) {
			console.error("Translation failed:", err);
			const message =
				err instanceof Error ? err.message : "Translation failed";
			if (
				message.includes("Cannot connect") ||
				message.includes("connection_refused")
			) {
				setError(
					"Cannot connect to AI backend. Make sure it is running and an LLM model is loaded.",
				);
			} else if (message.includes("Sarvam API key")) {
				setError("Sarvam API key is not configured. Add OPENCUTAI_SARVAM_API_KEY to your environment.");
			} else if (message.includes("Smallest AI API key")) {
				setError("Smallest AI API key is not configured. Add it in Settings > API Keys, or set OPENCUTAI_SMALLEST_API_KEY in your environment.");
			} else {
				setError(message);
			}
			bgTasks.updateTask(taskId, {
				status: "error",
				error: message,
				completedAt: Date.now(),
			});
		} finally {
			setIsTranslating(false);
			setTranslatingStep("");
		}
	};

	const handleRemoveSubtitles = () => {
		for (const track of subtitleTracks) {
			try {
				editor.timeline.removeTrack({ trackId: track.trackId });
			} catch {
				// Track may already have been removed manually
			}
		}
		setSubtitleTracks([]);
		toast.success("All subtitle tracks removed");
	};

	const handleRemoveSingleTrack = (trackId: string) => {
		try {
			editor.timeline.removeTrack({ trackId });
		} catch {
			// Already removed
		}
		setSubtitleTracks((prev) => prev.filter((t) => t.trackId !== trackId));
		toast.success("Subtitle track removed");
	};

	/** Resolve the first video media file on the timeline (shared by the
	 *  auto-detect and Remove handlers), ensuring it has a file extension
	 *  since the backend rejects files without one. Also reports the track it
	 *  sits on so cleaned spans can be stacked directly above it. */
	const resolveTimelineVideoFile = ():
		| { file: File; trackId: string }
		| { error: string } => {
		const tracks = editor.timeline.getTracks();
		let foundMediaId: string | null = null;
		let foundTrackId: string | null = null;

		for (const track of tracks) {
			for (const element of track.elements) {
				if (track.type === "video" && hasMediaId(element as TimelineElement)) {
					foundMediaId = (element as TimelineElement & { mediaId: string })
						.mediaId;
					foundTrackId = track.id;
					break;
				}
			}
			if (foundMediaId) break;
		}

		if (!foundMediaId || !foundTrackId) {
			return { error: "No video found on the timeline. Import a video file first." };
		}

		const mediaAsset = editor.media
			.getAssets()
			.find((asset) => asset.id === foundMediaId);

		if (!mediaAsset?.file) {
			return { error: "Cannot access the media file for subtitle removal." };
		}

		let file = mediaAsset.file;
		const fileName = file.name || "";
		const hasExtension =
			fileName.includes(".") && (fileName.split(".").pop() ?? "").length > 0;
		if (!hasExtension) {
			const newName = fileName ? `${fileName}.mp4` : "media.mp4";
			file = new File([file], newName, { type: file.type || "video/mp4" });
		}

		return { file, trackId: foundTrackId };
	};

	/** Find, or create, the video track cleaned spans are placed on.
	 *
	 *  buildScene reverses the tracks array before painting nodes in order
	 *  (scene-builder.ts), so a LOWER array index is painted LATER and ends up
	 *  on top. Inserting at the source track's index therefore drops the
	 *  overlay directly above it — and still below the text subtitle tracks,
	 *  which are added at index 0. */
	const ensureCleanTrack = ({
		sourceTrackId,
	}: {
		sourceTrackId: string;
	}): string => {
		const tracks = editor.timeline.getTracks();
		const existing = tracks.find(
			(track) => track.type === "video" && track.name === CLEAN_TRACK_NAME,
		);
		if (existing) return existing.id;

		const sourceIndex = tracks.findIndex((track) => track.id === sourceTrackId);
		const trackId = editor.timeline.addTrack({
			type: "video",
			index: sourceIndex === -1 ? 0 : sourceIndex,
		});
		editor.timeline.renameTrack({ trackId, name: CLEAN_TRACK_NAME });
		return trackId;
	};

	/** Human label for a cleaned span, e.g. `Clean 3:50–4:04`. */
	const cleanSpanLabel = ({ start, end }: { start: number; end: number }) =>
		`Clean ${formatTimeCode({ timeInSeconds: start, format: "MM:SS" })}–${formatTimeCode(
			{ timeInSeconds: end, format: "MM:SS" },
		)}`;

	/** Segments the Remove job would run on: enabled and with regions. */
	const scheduledSegments = detection
		? detection.segments.filter(
				(segment, index) =>
					enabledSegments[index] && segment.regions.length > 0,
			)
		: [];

	/** Segments that CAN be checked — the ones that found something to erase. */
	const selectableSegmentCount = detection
		? detection.segments.filter((segment) => segment.regions.length > 0).length
		: 0;
	const allSegmentsEnabled =
		scheduledSegments.length === selectableSegmentCount;
	const noSegmentsEnabled = scheduledSegments.length === 0;

	const handleDetectSubtitles = async () => {
		try {
			setIsDetectingRegion(true);
			setInpaintError(null);

			const resolved = resolveTimelineVideoFile();
			if ("error" in resolved) {
				setInpaintError(resolved.error);
				return;
			}

			const result = await aiClient.detectSubtitleTimeline(resolved.file);
			const withRegions = result.segments.filter(
				(segment) => segment.regions.length > 0,
			);
			if (withRegions.length === 0) {
				toast.error("No recurring text band detected — set the region manually");
				return;
			}
			setDetection(result);
			// Stretches with nothing to erase start disabled; there is no work
			// to do there and enabling them would only cost time.
			setEnabledSegments(
				result.segments.map((segment) => segment.regions.length > 0),
			);
			const areas = withRegions.reduce(
				(sum, segment) => sum + segment.regions.length,
				0,
			);
			toast.success(
				`${withRegions.length} stretch${withRegions.length === 1 ? "" : "es"} with burned-in text (${areas} area${areas === 1 ? "" : "s"}) — review, then Remove`,
			);
		} catch (err) {
			console.error("Subtitle detection failed:", err);
			const message =
				err instanceof Error ? err.message : "Subtitle detection failed";
			if (
				message.includes("Cannot connect") ||
				message.includes("connection_refused") ||
				message.includes("not available")
			) {
				setInpaintError(
					"Cannot connect to the inpaint service. Make sure it is running (docker compose up -d inpaint-service).",
				);
			} else {
				setInpaintError(message);
			}
		} finally {
			setIsDetectingRegion(false);
		}
	};

	/** Drop the detection and go back to the manual single-region flow. */
	const handleClearDetection = () => {
		setDetection(null);
		setEnabledSegments([]);
		setActiveSegment(null);
	};

	/** Scrub the playhead to a segment so its boxes show on the preview. */
	const handleSelectSegment = (index: number) => {
		const segment = detection?.segments[index];
		if (!segment) return;
		setActiveSegment(index);
		editor.playback.seek({ time: segment.start });
	};

	const handleToggleSegment = (index: number) => {
		setEnabledSegments((prev) =>
			prev.map((enabled, i) => (i === index ? !enabled : enabled)),
		);
	};

	/** Check/uncheck every segment that has something to erase. Segments with
	 *  no regions stay off — there is no work to do there. */
	const handleSetAllSegments = (enabled: boolean) => {
		if (!detection) return;
		setEnabledSegments(
			detection.segments.map(
				(segment) => enabled && segment.regions.length > 0,
			),
		);
	};

	/** Abort the whole run. Segments already placed on the timeline are left
	 *  alone — they are finished, valid work. */
	const handleStopInpaint = async () => {
		setIsCancellingInpaint(true);
		inpaintStopRequestedRef.current = true;
		inpaintAbortRef.current?.abort();
		setInpaintStep("Cancelling...");
		if (!inpaintJobId) return;
		try {
			await aiClient.cancelInpaintJob(inpaintJobId);
		} catch (err) {
			// The polling loop still exits via the stop flag; the orphaned
			// job keeps running server-side at worst.
			console.warn("Cancel request failed:", err);
		}
	};

	/** Upload one file to the inpaint service, poll it to completion and hand
	 *  back the cleaned video. Returns null when the run was stopped or the
	 *  job was cancelled server-side. */
	const runInpaintJob = async ({
		file,
		target,
		onPhase,
	}: {
		file: File;
		target: SubtitleRegion | { schedule: SubtitleScheduleEntry[] };
		onPhase: (phase: string, percent?: number) => void;
	}): Promise<Blob | null> => {
		onPhase("Uploading...", 0);
		const { job_id } = await aiClient.removeSubtitles(file, target);
		setInpaintJobId(job_id);

		try {
			let consecutiveFailures = 0;
			for (;;) {
				await new Promise((resolve) => setTimeout(resolve, INPAINT_POLL_MS));
				if (inpaintStopRequestedRef.current) return null;

				let status: Awaited<ReturnType<typeof aiClient.inpaintJobStatus>>;
				try {
					status = await aiClient.inpaintJobStatus(job_id);
					consecutiveFailures = 0;
				} catch (pollError) {
					consecutiveFailures++;
					if (consecutiveFailures >= MAX_POLL_FAILURES) throw pollError;
					continue;
				}

				if (status.status === "cancelled") return null;
				if (status.status === "error") {
					throw new Error(status.error || "Subtitle removal failed.");
				}

				const percent = Math.round((status.progress ?? 0) * 100);
				onPhase(status.message || `Processing... ${percent}%`, percent);

				if (status.status === "done") break;
			}

			onPhase("Downloading result...");
			const response = await fetch(aiClient.inpaintResultUrl(job_id));
			if (!response.ok) {
				throw new Error(`Failed to download result (${response.status})`);
			}
			return await response.blob();
		} finally {
			setInpaintJobId(null);
		}
	};

	/** Import a cleaned clip into the media panel and return its media id. */
	const importCleanedClip = async ({
		blob,
		name,
	}: {
		blob: Blob;
		name: string;
	}): Promise<{ mediaId: string; duration?: number }> => {
		const cleanFile = new File([blob], name, { type: "video/mp4" });
		const processed = await processMediaAssets({ files: [cleanFile] });
		if (processed.length === 0) {
			throw new Error("Failed to process the cleaned video.");
		}
		const mediaId = await editor.media.addMediaAsset({
			projectId: editor.project.getActive().metadata.id,
			asset: processed[0],
		});
		return { mediaId, duration: processed[0].duration };
	};

	/** Span-only removal.
	 *
	 *  For every enabled segment: cut that stretch out of the source in the
	 *  browser, send just that clip to the inpaint service, and drop the
	 *  cleaned result onto a dedicated track above the original as a muted
	 *  overlay. Nothing else is decoded, re-encoded or replaced.
	 *
	 *  Segments run one at a time — the service has a single worker, so
	 *  parallelism would only queue. Returns how many overlays were placed and
	 *  the extraction error, if any, that ended the run early. */
	const runSpanRemoval = async ({
		file,
		sourceTrackId,
		taskId,
		bgTasks,
	}: {
		file: File;
		sourceTrackId: string;
		taskId: string;
		bgTasks: ReturnType<typeof useBackgroundTasksStore.getState>;
	}): Promise<{
		placed: number;
		extractionFailure: SpanExtractionError | null;
	}> => {
		const total = scheduledSegments.length;
		let placed = 0;
		let cleanTrackId: string | null = null;

		for (let index = 0; index < total; index++) {
			if (inpaintStopRequestedRef.current) break;

			const segment = scheduledSegments[index];
			const label = `Segment ${index + 1} of ${total}`;
			const phase = (text: string, percent?: number) => {
				setInpaintStep(`${label} — ${text}`);
				if (percent !== undefined) setInpaintProgress(percent);
				bgTasks.updateTask(taskId, { progress: `${label} — ${text}` });
			};

			let span: File;
			try {
				phase("Extracting span...", 0);
				span = await extractSpan({
					file,
					start: segment.start,
					end: segment.end,
					signal: inpaintAbortRef.current?.signal,
				});
			} catch (err) {
				if (err instanceof SpanExtractionError) {
					return { placed, extractionFailure: err };
				}
				throw err;
			}

			if (inpaintStopRequestedRef.current) break;

			// Region coordinates are fractions of the frame, so they carry over
			// to the span unchanged. The span itself is the whole schedule: one
			// entry covering the entire clip.
			const spanDuration = segment.end - segment.start;
			const blob = await runInpaintJob({
				file: span,
				target: {
					schedule: [
						{ start: 0, end: spanDuration, regions: segment.regions },
					],
				},
				onPhase: phase,
			});
			if (!blob) break;

			phase("Importing cleaned span...");
			const imported = await importCleanedClip({
				blob,
				name: `clean-${span.name}`,
			});

			// Trust the imported clip's own duration where it is known: the
			// re-encode can land a frame either side of the requested span, and
			// stretching the element to match would freeze or drop a frame.
			const duration = imported.duration ?? spanDuration;

			const supportsTransaction =
				typeof editor.command.beginTransaction === "function";
			if (supportsTransaction) editor.command.beginTransaction();
			try {
				if (!cleanTrackId) {
					cleanTrackId = ensureCleanTrack({ sourceTrackId });
				}
				editor.timeline.insertElement({
					placement: { mode: "explicit", trackId: cleanTrackId },
					element: {
						type: "video",
						mediaId: imported.mediaId,
						name: cleanSpanLabel({ start: segment.start, end: segment.end }),
						startTime: segment.start,
						duration,
						trimStart: 0,
						trimEnd: 0,
						sourceDuration: duration,
						// The original underneath keeps playing its audio; the
						// span carries none anyway.
						muted: true,
						opacity: 1,
						transform: { scale: 1, position: { x: 0, y: 0 }, rotate: 0 },
					},
				});
				if (supportsTransaction) editor.command.commitTransaction();
			} catch (placementError) {
				if (supportsTransaction) editor.command.rollbackTransaction();
				throw placementError;
			}

			placed++;
		}

		return { placed, extractionFailure: null };
	};

	/** Whole-video removal: upload the entire source, re-encode every frame
	 *  and import the result as a new media asset. Kept for manual
	 *  single-region mode and as the fallback when the browser cannot cut the
	 *  source itself. */
	const runWholeVideoRemoval = async ({
		file,
		target,
		taskId,
		bgTasks,
	}: {
		file: File;
		target: SubtitleRegion | { schedule: SubtitleScheduleEntry[] };
		taskId: string;
		bgTasks: ReturnType<typeof useBackgroundTasksStore.getState>;
	}): Promise<boolean> => {
		const blob = await runInpaintJob({
			file,
			target,
			onPhase: (text, percent) => {
				setInpaintStep(text);
				if (percent !== undefined) setInpaintProgress(percent);
				bgTasks.updateTask(taskId, { progress: text });
			},
		});
		if (!blob) return false;

		setInpaintStep("Importing result...");
		bgTasks.updateTask(taskId, { progress: "Importing result..." });
		const baseName = file.name.replace(/\.[^.]+$/, "");
		const resultName = `${baseName}-clean.mp4`;
		await importCleanedClip({ blob, name: resultName });

		toast.success("Burned-in subtitles removed", {
			description: `${resultName} added to your media.`,
		});
		bgTasks.updateTask(taskId, {
			status: "completed",
			progress: resultName,
			completedAt: Date.now(),
		});
		return true;
	};

	const handleRemoveBurnedSubtitles = async () => {
		const taskId = `inpaint-${Date.now()}`;
		const bgTasks = useBackgroundTasksStore.getState();

		// Schedule mode runs on the enabled detected segments; manual mode on
		// the four inputs, which are validated the same way as before.
		const region = {
			x1: Number.parseFloat(inpaintRegion.x1),
			y1: Number.parseFloat(inpaintRegion.y1),
			x2: Number.parseFloat(inpaintRegion.x2),
			y2: Number.parseFloat(inpaintRegion.y2),
		};
		if (detection) {
			if (scheduledSegments.length === 0) {
				setInpaintError(
					"No segments selected. Enable at least one detected segment, or clear the detection to set a region manually.",
				);
				return;
			}
		} else if (
			[region.x1, region.y1, region.x2, region.y2].some((v) =>
				Number.isNaN(v),
			) ||
			region.x1 < 0 ||
			region.x2 > 1 ||
			region.x1 >= region.x2 ||
			region.y1 < 0 ||
			region.y2 > 1 ||
			region.y1 >= region.y2
		) {
			setInpaintError(
				"Region values must be fractions between 0 and 1, with x1 < x2 and y1 < y2.",
			);
			return;
		}

		try {
			setIsInpainting(true);
			setInpaintError(null);
			setInpaintProgress(0);
			inpaintStopRequestedRef.current = false;
			inpaintAbortRef.current = new AbortController();

			// Find the first video media asset on the timeline (same media
			// resolution as the transcribe handler, restricted to video)
			setInpaintStep("Finding media...");
			const resolved = resolveTimelineVideoFile();
			if ("error" in resolved) {
				setInpaintError(resolved.error);
				return;
			}
			const { file, trackId: sourceTrackId } = resolved;

			bgTasks.addTask({
				id: taskId,
				type: "inpaint",
				label: "Remove burned-in subtitles (STTN)",
				progress: "Starting...",
			});

			// Manual single-region mode has no per-segment timing to cut on, so
			// it stays on the whole-video path.
			if (!detection) {
				const finished = await runWholeVideoRemoval({
					file,
					target: region,
					taskId,
					bgTasks,
				});
				if (!finished) {
					bgTasks.removeTask(taskId);
					toast("Subtitle removal cancelled");
				}
				return;
			}

			const { placed, extractionFailure } = await runSpanRemoval({
				file,
				sourceTrackId,
				taskId,
				bgTasks,
			});

			// Nothing placed and the browser could not cut the source: fall all
			// the way back to re-encoding the whole video, as before.
			if (extractionFailure && placed === 0) {
				toast.warning("Falling back to whole-video processing", {
					description: `${extractionFailure.message} The entire video will be re-encoded, which is much slower.`,
				});
				const finished = await runWholeVideoRemoval({
					file,
					target: { schedule: scheduledSegments },
					taskId,
					bgTasks,
				});
				if (!finished) {
					bgTasks.removeTask(taskId);
					toast("Subtitle removal cancelled");
				}
				return;
			}

			if (placed === 0) {
				bgTasks.removeTask(taskId);
				toast("Subtitle removal cancelled");
				return;
			}

			if (extractionFailure) {
				// Some segments landed before extraction broke — keep them and
				// say what is left undone rather than discarding valid work.
				setInpaintError(
					`Stopped after ${placed} of ${scheduledSegments.length} segment(s): ${extractionFailure.message}`,
				);
			}

			// Stopping part way through still leaves finished overlays behind,
			// so report what landed either way.
			const stopped = inpaintStopRequestedRef.current;
			toast.success(
				`Placed ${placed} cleaned segment${placed === 1 ? "" : "s"} on the "${CLEAN_TRACK_NAME}" track`,
				{
					description: stopped
						? `Stopped after ${placed} of ${scheduledSegments.length}. The original clip is untouched — hide or delete the track to undo.`
						: "They overlay the original clip, which is untouched — hide or delete the track to undo.",
				},
			);
			bgTasks.updateTask(taskId, {
				status: "completed",
				progress: `${placed} segment${placed === 1 ? "" : "s"} placed`,
				completedAt: Date.now(),
			});
		} catch (err) {
			// Stop mid-extraction surfaces as an AbortError; that is a
			// cancellation, not a failure.
			if (err instanceof DOMException && err.name === "AbortError") {
				bgTasks.removeTask(taskId);
				toast("Subtitle removal cancelled");
				return;
			}

			console.error("Subtitle removal failed:", err);
			const message =
				err instanceof Error ? err.message : "An unexpected error occurred";
			if (
				message.includes("Cannot connect") ||
				message.includes("connection_refused") ||
				message.includes("not available")
			) {
				setInpaintError(
					"Cannot connect to the inpaint service. Make sure it is running (docker compose up -d inpaint-service).",
				);
			} else {
				setInpaintError(message);
			}
			bgTasks.updateTask(taskId, {
				status: "error",
				error: message,
				completedAt: Date.now(),
			});
		} finally {
			setIsInpainting(false);
			setInpaintStep("");
			setInpaintProgress(0);
			setInpaintJobId(null);
			setIsCancellingInpaint(false);
			inpaintStopRequestedRef.current = false;
			inpaintAbortRef.current = null;
		}
	};

	const handleLanguageChange = ({ value }: { value: string }) => {
		if (value === "auto") {
			setSelectedLanguage("auto");
			return;
		}
		setSelectedLanguage(value as TranscriptionLanguage);
	};

	const handleEngineChange = (value: string) => {
		const engine = value as TranscriptionEngine;
		setSelectedEngine(engine);
		// Reset language to auto when switching engines
		setSelectedLanguage("auto");
	};

	return (
		<PanelView title="Transcript" ref={containerRef}>
			<div className="flex flex-col gap-5">
				<p className="text-xs text-muted-foreground leading-relaxed">
					Transcribe your video to edit it like a document. Delete sections, remove filler words, or reorder segments.
				</p>

				{/* ── Engine Selector ── */}
				<div className="flex flex-col gap-2">
					<Label className="text-xs">Transcription engine</Label>
					<Select value={selectedEngine} onValueChange={handleEngineChange}>
						<SelectTrigger>
							<SelectValue />
						</SelectTrigger>
						<SelectContent>
							<SelectItem value="whisper">
								Whisper (Local)
							</SelectItem>
							<SelectItem value="sarvam">
								Sarvam AI (Indian Languages)
							</SelectItem>
							<SelectItem value="smallest">
								Smallest AI Pulse (39 Languages)
							</SelectItem>
						</SelectContent>
					</Select>
					<p className="text-[10px] text-muted-foreground">
						{selectedEngine === "sarvam"
							? "Cloud-based, optimized for 22 Indian regional languages"
							: selectedEngine === "smallest"
								? "Cloud-based, 39 languages with speaker diarization & emotion detection"
								: "On-device, best for global languages (English, Spanish, French, etc.)"}
					</p>
				</div>

				{/* ── Language Selector ── */}
				<div className="flex flex-col gap-2">
					<Label className="text-xs">Language</Label>
					<Select
						value={selectedLanguage}
						onValueChange={(value) => handleLanguageChange({ value })}
					>
						<SelectTrigger>
							<SelectValue placeholder="Select a language" />
						</SelectTrigger>
						<SelectContent>
							<SelectItem value="auto">Auto detect</SelectItem>
							{selectedEngine === "sarvam" ? (
								<>
									<SelectGroup>
										<SelectLabel className="text-[10px] text-muted-foreground px-2">
											Indian Regional Languages
										</SelectLabel>
										{SARVAM_STT_LANGUAGES.filter(l => l.code !== "en").map((language) => (
											<SelectItem key={language.code} value={language.code}>
												{language.name}
											</SelectItem>
										))}
									</SelectGroup>
									<SelectGroup>
										<SelectLabel className="text-[10px] text-muted-foreground px-2">
											English
										</SelectLabel>
										<SelectItem value="en">
											English (Indian)
										</SelectItem>
									</SelectGroup>
								</>
							) : (
								availableLanguages.map((language) => (
									<SelectItem key={language.code} value={language.code}>
										{language.name}
									</SelectItem>
								))
							)}
						</SelectContent>
					</Select>
				</div>

				{error && (
					<div className="bg-destructive/10 border-destructive/20 rounded-md border p-3">
						<p className="text-destructive text-sm">{error}</p>
					</div>
				)}

				{segments.length > 0 && activeSubtitleTracks.length === 0 && (
					<p className="text-[11px] text-muted-foreground leading-relaxed rounded-md bg-muted/50 px-3 py-2">
						Transcript is ready. Add subtitles below, or re-transcribe if the video changed.
					</p>
				)}

				<div className="flex items-center justify-between gap-2">
					<div className="flex flex-col">
						<Label className="text-xs" htmlFor="split-on-transcribe">
							Split clips at segment boundaries
						</Label>
						<span className="text-[10px] text-muted-foreground">
							For text-based editing. Slow on long videos.
						</span>
					</div>
					<Switch
						id="split-on-transcribe"
						checked={splitOnTranscribe}
						onCheckedChange={setSplitOnTranscribe}
					/>
				</div>

				<Button
					className="w-full"
					variant={segments.length > 0 ? "outline" : "default"}
					onClick={handleGenerateTranscript}
					disabled={isProcessing}
				>
					{isProcessing && <Spinner className="mr-1" />}
					{isProcessing
						? processingStep
						: segments.length > 0
							? "Re-transcribe"
							: "Generate transcript"}
				</Button>

				{segments.length > 0 && (
					<>
						{/* ── Subtitle Tracks ── */}
						<div className="border-t pt-4 flex flex-col gap-3">
							<div className="flex items-center justify-between">
								<Label className="text-xs">Subtitle tracks</Label>
								{activeSubtitleTracks.length > 0 && (
									<span className="text-[10px] text-muted-foreground tabular-nums">
										{activeSubtitleTracks.length} active
									</span>
								)}
							</div>

							{activeSubtitleTracks.length > 0 ? (
								<div className="flex flex-col gap-1.5">
									{activeSubtitleTracks.map((track) => {
										const langName =
											track.language === "original"
												? "Original"
												: LANGUAGES.find((l) => l.code === track.language)
														?.name ?? track.language;
										return (
											<div
												key={track.trackId}
												className="flex items-center justify-between rounded-md border px-3 py-2"
											>
												<div className="flex items-center gap-2">
													<span className="bg-primary size-1.5 rounded-full shrink-0" />
													<span className="text-sm font-medium">
														{langName}
													</span>
												</div>
												<Button
													variant="ghost"
													size="sm"
													className="h-6 px-2 text-xs text-muted-foreground hover:text-destructive"
													onClick={() =>
														handleRemoveSingleTrack(track.trackId)
													}
												>
													Remove
												</Button>
											</div>
										);
									})}
									<Button
										variant="outline"
										size="sm"
										className="w-full text-destructive hover:text-destructive"
										onClick={handleRemoveSubtitles}
									>
										Remove all
									</Button>
								</div>
							) : (
								<Button
									variant="outline"
									className="w-full"
									onClick={handleAddSubtitles}
								>
									Add subtitles
								</Button>
							)}
						</div>

						{/* ── Add Language ── */}
						<div className="border-t pt-4 flex flex-col gap-3">
							<Label className="text-xs">Add language</Label>
							<p className="text-[11px] text-muted-foreground leading-relaxed">
								{shouldUseSarvamTranslation(
									useTranscriptStore.getState().language,
									translateLanguage,
								)
									? "Translate using Sarvam AI and add as a new subtitle track."
									: "Translate using the local LLM and add as a new subtitle track."}
							</p>
							<div className="flex gap-2">
								<Select
									value={translateLanguage}
									onValueChange={setTranslateLanguage}
								>
									<SelectTrigger className="flex-1">
										<SelectValue />
									</SelectTrigger>
									<SelectContent>
										<SelectGroup>
											<SelectLabel className="text-[10px] text-muted-foreground px-2">
												Global Languages
											</SelectLabel>
											{LANGUAGES.filter(
												(lang) =>
													!SARVAM_SUPPORTED_CODES.has(lang.code) &&
													!activeSubtitleTracks.some(
														(t) => t.language === lang.code,
													),
											).map((lang) => (
												<SelectItem key={lang.code} value={lang.code}>
													{lang.name}
												</SelectItem>
											))}
										</SelectGroup>
										<SelectGroup>
											<SelectLabel className="text-[10px] text-muted-foreground px-2">
												Indian Languages (via Sarvam AI)
											</SelectLabel>
											{LANGUAGES.filter(
												(lang) =>
													SARVAM_SUPPORTED_CODES.has(lang.code) &&
													lang.code !== "en" &&
													!activeSubtitleTracks.some(
														(t) => t.language === lang.code,
													),
											).map((lang) => (
												<SelectItem key={lang.code} value={lang.code}>
													{lang.name}
												</SelectItem>
											))}
										</SelectGroup>
									</SelectContent>
								</Select>
								<Button
									onClick={handleTranslateAndAdd}
									disabled={isTranslating}
									className="shrink-0"
								>
									{isTranslating && <Spinner className="mr-1" />}
									{isTranslating ? "..." : "Add"}
								</Button>
							</div>
							{isTranslating && translatingStep && (
								<p className="text-[11px] text-muted-foreground animate-pulse">
									{translatingStep}
								</p>
							)}
						</div>
					</>
				)}

				{/* ── Burned-in Subtitles ── */}
				<div className="border-t pt-4 flex flex-col gap-3">
					<Label className="text-xs">Burned-in subtitles</Label>
					<p className="text-[11px] text-muted-foreground leading-relaxed">
						Erase hardcoded subtitles with AI inpainting (STTN). Runs locally.
					</p>

					<Button
						variant="outline"
						size="sm"
						className="w-full"
						onClick={handleDetectSubtitles}
						disabled={isDetectingRegion || isInpainting}
					>
						{isDetectingRegion && <Spinner className="mr-1" />}
						{isDetectingRegion
							? "Scanning video for text..."
							: "Auto-detect subtitles"}
					</Button>

					{detection ? (
						<div className="flex flex-col gap-2">
							<div className="flex items-center justify-between">
								<Label className="text-xs">
									Detected segments
									<span className="ml-1 text-[10px] text-muted-foreground tabular-nums">
										{detection.windows} windows
									</span>
								</Label>
								<div className="flex items-center gap-0.5">
									<Button
										variant="ghost"
										size="sm"
										className="h-6 px-2 text-xs text-muted-foreground"
										onClick={() => handleSetAllSegments(true)}
										disabled={isInpainting || allSegmentsEnabled}
									>
										All
									</Button>
									<Button
										variant="ghost"
										size="sm"
										className="h-6 px-2 text-xs text-muted-foreground"
										onClick={() => handleSetAllSegments(false)}
										disabled={isInpainting || noSegmentsEnabled}
									>
										None
									</Button>
									<Button
										variant="ghost"
										size="sm"
										className="h-6 px-2 text-xs text-muted-foreground"
										onClick={handleClearDetection}
										disabled={isInpainting}
									>
										Clear
									</Button>
								</div>
							</div>

							<div className="flex max-h-52 flex-col gap-1 overflow-y-auto">
								{detection.segments.map((segment, index) => (
									<div
										key={`${segment.start}-${segment.end}`}
										className={cn(
											"flex items-center gap-2 rounded-md border px-2 py-1.5",
											index === activeSegment && "border-primary bg-muted/50",
										)}
									>
										<Checkbox
											id={`inpaint-segment-${index}`}
											checked={enabledSegments[index] ?? false}
											disabled={isInpainting || segment.regions.length === 0}
											onCheckedChange={() => handleToggleSegment(index)}
										/>
										<button
											type="button"
											className="flex min-w-0 flex-1 items-center justify-between gap-2 text-left"
											onClick={() => handleSelectSegment(index)}
										>
											<span className="text-[11px] tabular-nums">
												{formatTimeCode({
													timeInSeconds: segment.start,
													format: "MM:SS",
												})}
												–
												{formatTimeCode({
													timeInSeconds: segment.end,
													format: "MM:SS",
												})}
											</span>
											<span className="text-[10px] text-muted-foreground shrink-0">
												{segment.regions.length === 0
													? "none"
													: `${segment.regions.length} area${
															segment.regions.length === 1 ? "" : "s"
														}`}
											</span>
										</button>
									</div>
								))}
							</div>
							<p className="text-[10px] text-muted-foreground">
								Click a row to jump there and check its boxes on the preview.
							</p>
						</div>
					) : (
						<div className="grid grid-cols-2 gap-2">
							{(
								[
									["x1", "Left (x1)"],
									["y1", "Top (y1)"],
									["x2", "Right (x2)"],
									["y2", "Bottom (y2)"],
								] as const
							).map(([key, label]) => (
								<div key={key} className="flex flex-col gap-1">
									<Label
										className="text-[10px] text-muted-foreground"
										htmlFor={`inpaint-${key}`}
									>
										{label}
									</Label>
									<Input
										id={`inpaint-${key}`}
										type="number"
										min={0}
										max={1}
										step={0.01}
										value={inpaintRegion[key]}
										disabled={isInpainting}
										onChange={(e) =>
											setInpaintRegion((prev) => ({
												...prev,
												[key]: e.target.value,
											}))
										}
									/>
								</div>
							))}
						</div>
					)}

					<div className="flex items-center justify-between gap-2">
						<div className="flex flex-col">
							<Label className="text-xs" htmlFor="show-inpaint-region">
								Show on preview
							</Label>
							<span className="text-[10px] text-muted-foreground">
								{detection
									? "Green boxes show what will be erased at the playhead."
									: "The green box shows what will be erased."}
							</span>
						</div>
						<Switch
							id="show-inpaint-region"
							checked={showRegionOnPreview}
							onCheckedChange={setShowRegionOnPreview}
						/>
					</div>

					{inpaintError && (
						<div className="bg-destructive/10 border-destructive/20 rounded-md border p-3">
							<p className="text-destructive text-sm">{inpaintError}</p>
						</div>
					)}

					{isInpainting ? (
						<div className="flex gap-2">
							<Button className="min-w-0 flex-1" variant="outline" disabled>
								<Spinner className="mr-1" />
								<span className="truncate">
									{`${inpaintStep || "Processing..."}${
										inpaintProgress > 0 ? ` (${inpaintProgress}%)` : ""
									}`}
								</span>
							</Button>
							<Button
								variant="outline"
								className="shrink-0 text-destructive hover:text-destructive"
								onClick={handleStopInpaint}
								disabled={isCancellingInpaint}
							>
								{isCancellingInpaint ? "Stopping..." : "Stop"}
							</Button>
						</div>
					) : (
						<Button
							className="w-full"
							variant="outline"
							onClick={handleRemoveBurnedSubtitles}
							disabled={isDetectingRegion}
						>
							{detection
								? `Remove from ${scheduledSegments.length} segment${
										scheduledSegments.length === 1 ? "" : "s"
									}`
								: "Remove subtitles"}
						</Button>
					)}

					<p className="text-[10px] text-muted-foreground">
						{detection
							? `Only the ${scheduledSegments.length} selected segment(s) are cut out, cleaned (${scheduledSegments.reduce(
									(sum, segment) => sum + segment.regions.length,
									0,
								)} area(s)) and placed as overlay clips on a "${CLEAN_TRACK_NAME}" track. The original clip is never re-encoded or replaced.`
							: "Will erase the region above for the whole video, re-encoding every frame."}{" "}
						CPU processing is slow — expect several minutes per video minute.
					</p>
				</div>
			</div>
		</PanelView>
	);
}
