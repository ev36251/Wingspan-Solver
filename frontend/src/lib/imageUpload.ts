// Shared screenshot-upload helper: downscale + base64-encode an image file.

// Claude vision works best (and cheapest) at <=1568px on the long edge.
const MAX_DIM = 1568;

export interface ImagePayload {
	media_type: string;
	data: string;
}

export async function fileToImagePayload(file: File): Promise<ImagePayload> {
	try {
		const bitmap = await createImageBitmap(file);
		const scale = Math.min(1, MAX_DIM / Math.max(bitmap.width, bitmap.height));
		if (scale < 1 || file.size > 1_500_000) {
			const canvas = document.createElement('canvas');
			canvas.width = Math.round(bitmap.width * scale);
			canvas.height = Math.round(bitmap.height * scale);
			canvas.getContext('2d')!.drawImage(bitmap, 0, 0, canvas.width, canvas.height);
			const dataUrl = canvas.toDataURL('image/jpeg', 0.85);
			return { media_type: 'image/jpeg', data: dataUrl.split(',')[1] };
		}
	} catch {
		// createImageBitmap unsupported for this file — send it as-is below.
	}
	const b64 = await new Promise<string>((resolve, reject) => {
		const reader = new FileReader();
		reader.onload = () => resolve((reader.result as string).split(',')[1]);
		reader.onerror = () => reject(reader.error);
		reader.readAsDataURL(file);
	});
	return { media_type: file.type || 'image/png', data: b64 };
}
