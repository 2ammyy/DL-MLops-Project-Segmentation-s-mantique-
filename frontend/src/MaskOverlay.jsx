import { useEffect, useRef, useState } from "react";

function colorizeMask(maskBase64) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => {
      const off = document.createElement("canvas");
      off.width = img.width;
      off.height = img.height;
      const ctx = off.getContext("2d");
      ctx.drawImage(img, 0, 0);
      const data = ctx.getImageData(0, 0, off.width, off.height);
      const px = data.data;
      for (let i = 0; i < px.length; i += 4) {
        if (px[i] > 127) {
          px[i] = 255;
          px[i + 1] = 40;
          px[i + 2] = 40;
          px[i + 3] = 255;
        } else {
          px[i + 3] = 0;
        }
      }
      ctx.putImageData(data, 0, 0);
      resolve(off);
    };
    img.onerror = () => reject(new Error("Decodage du masque impossible"));
    img.src = `data:image/png;base64,${maskBase64}`;
  });
}

export default function MaskOverlay({ imageUrl, maskBase64 }) {
  const canvasRef = useRef(null);
  const maskRef = useRef(null);
  const [opacity, setOpacity] = useState(60);

  useEffect(() => {
    if (!imageUrl || !maskBase64) return undefined;
    let cancelled = false;
    colorizeMask(maskBase64)
      .then((maskCanvas) => {
        if (!cancelled) {
          maskRef.current = maskCanvas;
          draw();
        }
      })
      .catch((err) => console.error(err));
    return () => {
      cancelled = true;
    };
  }, [imageUrl, maskBase64]);

  useEffect(() => {
    draw();
  }, [opacity]);

  function draw() {
    const canvas = canvasRef.current;
    if (!canvas || !imageUrl) return;
    const img = new Image();
    img.onload = () => {
      canvas.width = img.width;
      canvas.height = img.height;
      const ctx = canvas.getContext("2d");
      ctx.drawImage(img, 0, 0);
      if (maskRef.current) {
        ctx.globalAlpha = opacity / 100;
        ctx.drawImage(maskRef.current, 0, 0);
        ctx.globalAlpha = 1;
      }
    };
    img.src = imageUrl;
  }

  if (!imageUrl) {
    return <div className="overlay-empty">Aucune image chargee</div>;
  }

  return (
    <div className="mask-overlay">
      <canvas ref={canvasRef} className="overlay-canvas" />
      <div className="overlay-controls">
        <label htmlFor="opacity-slider" className="slider-label">
          Opacite du masque : <strong>{opacity}%</strong>
        </label>
        <input
          id="opacity-slider"
          type="range"
          min="0"
          max="100"
          value={opacity}
          onChange={(e) => setOpacity(Number(e.target.value))}
          className="opacity-slider"
        />
        <div className="slider-scale">
          <span>Image seule</span>
          <span>Masque seul</span>
        </div>
      </div>
    </div>
  );
}
