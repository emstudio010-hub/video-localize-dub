import React, {useEffect, useState} from 'react';
import {
	AbsoluteFill, OffthreadVideo, continueRender, delayRender, interpolate, spring,
	staticFile, useCurrentFrame, useVideoConfig,
} from 'remotion';

type LineStyle = {size: number; color: string; letter_spacing: number};
type Word = {w: string; f: number};
type Card = {from: number; to: number; lines: string[]; words: Word[]};
type Plate = {color: string; opacity?: number; radius?: number; padding?: number} | null;
type Highlight = {color?: string; plate?: string; radius?: number} | null;

export type Props = {
	fps: number; width: number; height: number; durationInFrames: number; video: string; lang: string;
	style: {
		center: number[]; rotationDeg: number; writingMode: string; maxWidth: number;
		fontFamily: string; fontFile: string | null;
		line1: LineStyle; line2: LineStyle; lineGap: number;
		strokePx: number; strokeColor: string; shadow: string;
		plate: Plate; animation: string; highlight: Highlight;
	};
	cards: Card[];
};

const FONT_NAME = 'LocalizeCaptionFont';

const useFont = (file: string | null) => {
	const [handle] = useState(() => (file ? delayRender('caption font') : null));
	useEffect(() => {
		if (!file || handle === null) return;
		const face = new FontFace(FONT_NAME, `url(${staticFile(file)})`);
		face.load()
			.then((f) => { document.fonts.add(f); continueRender(handle); })
			.catch((e) => { console.error('font load failed', e); continueRender(handle); });
	}, [file, handle]);
};

const entrance = (anim: string, local: number, fps: number) => {
	const s = spring({frame: local, fps, config: {damping: 12, stiffness: 220, mass: 0.6}});
	switch (anim) {
		case 'pop': return {scale: interpolate(s, [0, 1], [0.6, 1]), opacity: Math.min(1, local / 2 + 0.5), dy: 0};
		case 'fade': return {scale: 1, opacity: interpolate(local, [0, 5], [0, 1], {extrapolateRight: 'clamp'}), dy: 0};
		case 'slide': return {scale: 1, opacity: Math.min(1, local / 3), dy: interpolate(s, [0, 1], [18, 0])};
		default: return {scale: 1, opacity: 1, dy: 0};
	}
};

export const Localized: React.FC<Props> = ({video, style, cards, lang}) => {
	const frame = useCurrentFrame();
	const {fps} = useVideoConfig();
	useFont(style.fontFile);
	const card = cards.find((c) => frame >= c.from && frame < c.to);
	const family = style.fontFile ? `${FONT_NAME}, ${style.fontFamily}` : style.fontFamily;
	const joiner = ['zh', 'ja', 'ko'].includes(lang) ? '' : ' ';

	let body: React.ReactNode = null;
	if (card) {
		const {scale, opacity, dy} = entrance(style.animation, frame - card.from, fps);
		const active = style.highlight ? card.words.filter((w) => w.f <= frame).length - 1 : -1;
		let wi = 0;
		body = (
			<div
				style={{
					position: 'absolute', left: style.center[0], top: style.center[1],
					transform: `translate(-50%, -50%) translateY(${dy}px) rotate(${style.rotationDeg}deg) scale(${scale})`,
					opacity, writingMode: style.writingMode as React.CSSProperties['writingMode'],
					maxWidth: style.maxWidth, textAlign: 'center', whiteSpace: 'nowrap',
					fontFamily: family, lineHeight: 1 + style.lineGap,
					padding: style.plate ? style.plate.padding ?? 10 : 0,
					borderRadius: style.plate?.radius ?? 0,
					background: style.plate ? style.plate.color : 'transparent',
				}}
			>
				{card.lines.map((line, li) => {
					const ls = li === 0 || card.lines.length === 1 ? style.line1 : style.line2;
					const tokens = joiner ? line.split(' ') : Array.from(line);
					return (
						<div key={li} style={{
							fontSize: ls.size, color: ls.color, letterSpacing: ls.letter_spacing,
							WebkitTextStroke: `${style.strokePx * 2}px ${style.strokeColor}`,
							paintOrder: 'stroke fill', textShadow: style.shadow,
						}}>
							{tokens.map((t, ti) => {
								const idx = wi++;
								const on = idx === active && style.highlight;
								return (
									<span key={ti} style={on ? {
										color: style.highlight!.color ?? ls.color,
										background: style.highlight!.plate ?? 'transparent',
										borderRadius: style.highlight!.radius ?? 6, padding: '0 0.12em',
									} : undefined}>
										{t}{ti < tokens.length - 1 ? joiner : ''}
									</span>
								);
							})}
						</div>
					);
				})}
			</div>
		);
	}

	return (
		<AbsoluteFill style={{backgroundColor: 'black'}}>
			<OffthreadVideo src={staticFile(video)} muted />
			{body}
		</AbsoluteFill>
	);
};
