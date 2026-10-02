import React from 'react';
import {Composition} from 'remotion';
import {Localized, Props} from './Localized';

// Real values come from --props (captions.json written by build_captions.py).
const fallback: Props = {
	fps: 30, width: 1080, height: 1920, durationInFrames: 30, video: 'clean.mp4', lang: 'en',
	style: {
		center: [540, 1500], rotationDeg: 0, writingMode: 'horizontal-tb', maxWidth: 970,
		fontFamily: 'Impact', fontFile: null,
		line1: {size: 34, color: '#FFE600', letter_spacing: 1.2},
		line2: {size: 32, color: '#FFFFFF', letter_spacing: 1},
		lineGap: 0.05, strokePx: 4.5, strokeColor: '#000000', shadow: 'none',
		plate: null, animation: 'pop', highlight: null,
	},
	cards: [],
};

export const RemotionRoot: React.FC = () => (
	<Composition
		id="Localized"
		component={Localized}
		defaultProps={fallback}
		calculateMetadata={({props}) => ({
			durationInFrames: props.durationInFrames,
			fps: props.fps,
			width: props.width,
			height: props.height,
		})}
	/>
);
