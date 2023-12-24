import * as piexif from "piexifjs";

interface IImageInfo {
    prompt?: string;
    negative_prompt?: string;
    seed?: number;
    guidance_scale?: number;
}

export function snapImage(imageEl: HTMLImageElement, info: IImageInfo) {
    try {
        const zeroth: { [key: string]: any } = {};
        const exif: { [key: string]: any } = {};
        const gps: { [key: string]: any } = {};
        zeroth[piexif.ImageIFD.Make] = "LCM Image-to-Image ControNet";
        zeroth[piexif.ImageIFD.ImageDescription] = `prompt: ${info?.prompt} | negative_prompt: ${info?.negative_prompt} | seed: ${info?.seed} | guidance_scale: ${info?.guidance_scale}`;
        zeroth[piexif.ImageIFD.Software] = "https://github.com/radames/Real-Time-Latent-Consistency-Model";
        exif[piexif.ExifIFD.DateTimeOriginal] = new Date().toISOString();

        const exifObj = { "0th": zeroth, "Exif": exif, "GPS": gps };
        const exifBytes = piexif.dump(exifObj);

        const canvas = document.createElement("canvas");
        canvas.width = imageEl.naturalWidth;
        canvas.height = imageEl.naturalHeight;
        const ctx = canvas.getContext("2d") as CanvasRenderingContext2D;
        ctx.drawImage(imageEl, 0, 0);
        const dataURL = canvas.toDataURL("image/jpeg");
        const withExif = piexif.insert(exifBytes, dataURL);

        const a = document.createElement("a");
        a.href = withExif;
        a.download = `lcm_txt_2_img${Date.now()}.png`;
        a.click();
    } catch (err) {
        console.log(err);
    }
}
