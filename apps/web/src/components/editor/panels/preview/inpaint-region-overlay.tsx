"use client";

import { usePreviewStore } from "@/stores/preview-store";

/**
 * Highlights the burned-in-subtitle removal region on the preview.
 * The region is set by the Captions panel as fractions (0-1) of the
 * frame, so plain percentage positioning maps it onto the letterboxed
 * canvas bounds without any pixel math.
 */
export function InpaintRegionOverlay() {
	const { inpaintRegionOverlay } = usePreviewStore();

	if (!inpaintRegionOverlay) return null;

	const { x1, y1, x2, y2 } = inpaintRegionOverlay;

	return (
		<div
			data-overlay="inpaint-region"
			className="pointer-events-none absolute border-2 border-emerald-500 bg-emerald-500/25"
			style={{
				left: `${x1 * 100}%`,
				top: `${y1 * 100}%`,
				width: `${(x2 - x1) * 100}%`,
				height: `${(y2 - y1) * 100}%`,
			}}
		>
			<span className="absolute top-0 left-0 max-w-full truncate rounded-br-sm bg-emerald-500 px-1.5 py-0.5 text-[10px] font-medium leading-none text-white">
				Subtitle removal area
			</span>
		</div>
	);
}
