// Shared screenshot-upload helper: downscale + base64-encode an image file.

// Claude vision works best (and cheapest) at <=1568px on the long edge.
const MAX_DIM = 1568;

export interface ImagePayload {
	media_type: string;
	data: string;
}

const SUPPORTED = ['image/png', 'image/jpeg', 'image/webp', 'image/gif'];

export async function fileToImagePayload(file: File): Promise<ImagePayload> {
	// Always re-encode through a canvas to JPEG when the browser can decode the
	// file. This downscales large images AND normalizes formats the backend
	// doesn't accept (e.g. iPhone HEIC photos, which Safari can decode) into a
	// guaranteed-supported media type.
	try {
		const bitmap = await createImageBitmap(file);
		const scale = Math.min(1, MAX_DIM / Math.max(bitmap.width, bitmap.height));
		const canvas = document.createElement('canvas');
		canvas.width = Math.max(1, Math.round(bitmap.width * scale));
		canvas.height = Math.max(1, Math.round(bitmap.height * scale));
		canvas.getContext('2d')!.drawImage(bitmap, 0, 0, canvas.width, canvas.height);
		const dataUrl = canvas.toDataURL('image/jpeg', 0.85);
		return { media_type: 'image/jpeg', data: dataUrl.split(',')[1] };
	} catch {
		// createImageBitmap couldn't decode it — fall back to raw bytes.
	}
	const b64 = await new Promise<string>((resolve, reject) => {
		const reader = new FileReader();
		reader.onload = () => resolve((reader.result as string).split(',')[1]);
		reader.onerror = () => reject(reader.error);
		reader.readAsDataURL(file);
	});
	const media_type = SUPPORTED.includes(file.type) ? file.type : 'image/png';
	return { media_type, data: b64 };
}
