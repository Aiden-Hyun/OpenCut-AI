"use client";

import { SubTabView } from "./sub-tab-view";
import { SoundsView } from "./sounds";
import { VoiceoverView } from "./voiceover";
import { PodcastClipsView } from "./podcast-clips";
import { AIDubbingPanel } from "./ai-dubbing";

export function AudioCombinedView() {
	return (
		<SubTabView
			tabs={[
				{ key: "sounds", label: "Sounds", content: <SoundsView /> },
				{ key: "voiceover", label: "Voiceover", content: <VoiceoverView /> },
				{ key: "dubbing", label: "Dubbing", content: <AIDubbingPanel /> },
				{ key: "podcast", label: "Podcast", content: <PodcastClipsView /> },
			]}
		/>
	);
}
