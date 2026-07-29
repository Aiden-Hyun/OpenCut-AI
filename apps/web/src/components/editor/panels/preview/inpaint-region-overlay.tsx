"use client";

import { usePreviewStore } from "@/stores/preview-store";

/**
 * Highlights the burned-in-subtitle removal regions on the preview.
 * Regions are set by the Captions panel as fractions (0-1) of the frame, so
 * plain percentage positioning maps them onto the letterboxed canvas bounds
 * without any pixel math. Manual mode sets a single box; a detected segment
 * can have several active at the same timestamp.
 */
export function InpaintRegionOverlay() {
	const { inpaintRegionOverlays } = usePreviewStore();

	if (inpaintRegionOverlays.length === 0) return null;

	return (
		<>
			{inpaintRegionOverlays.map(({ x1, y1, x2, y2 }) => (
				<div
					key={`${x1}-${y1}-${x2}-${y2}`}
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
			))}
		</>
	);
}
