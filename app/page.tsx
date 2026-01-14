"use client";

import { useEffect, useRef, useState } from "react";
import * as ort from "onnxruntime-web";

type CvType = any;

export default function Home() {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  const [status, setStatus] = useState<string>("ยังไม่เริ่ม");
  const [emotion, setEmotion] = useState<string>("-");
  const [conf, setConf] = useState<number>(0);

  // เพิ่ม Ref เพื่อเช็คสถานะว่ากล้องทำงานอยู่หรือไม่
  const isRunningRef = useRef<boolean>(false);

  const cvRef = useRef<CvType | null>(null);
  const faceCascadeRef = useRef<any>(null);
  const sessionRef = useRef<ort.InferenceSession | null>(null);
  const classesRef = useRef<string[] | null>(null);

  // 1) Load OpenCV
  async function loadOpenCV() {
    if (typeof window === "undefined") return;

    if ((window as any).cv?.Mat) {
      cvRef.current = (window as any).cv;
      return;
    }

    await new Promise<void>((resolve, reject) => {
      const script = document.createElement("script");
      script.src = "/opencv/opencv.js";
      script.async = true;

      script.onload = () => {
        const waitReady = () => {
          if ((window as any).cv?.Mat) {
            cvRef.current = (window as any).cv;
            resolve();
          } else {
            setTimeout(waitReady, 50);
          }
        };

        const cv = (window as any).cv;
        if (cv && "onRuntimeInitialized" in cv) {
          cv.onRuntimeInitialized = () => waitReady();
        } else {
          waitReady();
        }
      };

      script.onerror = () => reject(new Error("โหลด /opencv/opencv.js ไม่สำเร็จ"));
      document.body.appendChild(script);
    });
  }

  // 2) Load Haar cascade
  async function loadCascade() {
    const cv = cvRef.current;
    if (!cv) throw new Error("cv ยังไม่พร้อม");

    const cascadeUrl = "/opencv/haarcascade_frontalface_default.xml";
    const res = await fetch(cascadeUrl);
    if (!res.ok) throw new Error("โหลด cascade ไม่สำเร็จ");
    const data = new Uint8Array(await res.arrayBuffer());

    const cascadePath = "haarcascade_frontalface_default.xml";
    try {
      cv.FS_unlink(cascadePath);
    } catch { }
    cv.FS_createDataFile("/", cascadePath, data, true, false, false);

    const faceCascade = new cv.CascadeClassifier();
    const loaded = faceCascade.load(cascadePath);
    if (!loaded) throw new Error("cascade load() ไม่สำเร็จ");
    faceCascadeRef.current = faceCascade;
  }

  // 3) Load ONNX model
  async function loadModel() {
    const session = await ort.InferenceSession.create(
      "/models/emotion_yolo11n_cls.onnx",
      { executionProviders: ["wasm"] }
    );
    sessionRef.current = session;

    const clsRes = await fetch("/models/classes.json");
    if (!clsRes.ok) throw new Error("โหลด classes.json ไม่สำเร็จ");
    classesRef.current = await clsRes.json();
  }

  // 4) Start camera
  async function startCamera() {
    // ถ้าทำงานอยู่แล้ว ไม่ต้องเริ่มใหม่
    if (isRunningRef.current) return;

    setStatus("ขอสิทธิ์กล้อง...");
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "user" },
        audio: false,
      });
      
      if (!videoRef.current) return;
      videoRef.current.srcObject = stream;
      await videoRef.current.play();
      
      setStatus("กำลังทำงาน...");
      isRunningRef.current = true; // ตั้งค่า flag ว่าเริ่มแล้ว
      requestAnimationFrame(loop);
    } catch (error: any) {
      setStatus(`เปิดกล้องไม่สำเร็จ: ${error.message}`);
    }
  }

  // ฟังก์ชันใหม่: หยุดกล้อง
  function stopCamera() {
    if (!isRunningRef.current) return;

    isRunningRef.current = false; // สั่งหยุด loop
    setStatus("หยุดกล้องแล้ว");
    setEmotion("-");
    setConf(0);

    // หยุด MediaStream
    if (videoRef.current && videoRef.current.srcObject) {
      const stream = videoRef.current.srcObject as MediaStream;
      stream.getTracks().forEach((track) => track.stop());
      videoRef.current.srcObject = null;
    }

    // เคลียร์ Canvas
    if (canvasRef.current) {
      const ctx = canvasRef.current.getContext("2d");
      if (ctx) {
        ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
      }
    }
  }

  // 5) Preprocess face ROI -> tensor
  function preprocessToTensor(faceCanvas: HTMLCanvasElement) {
    const size = 64;
    const tmp = document.createElement("canvas");
    tmp.width = size;
    tmp.height = size;
    const ctx = tmp.getContext("2d")!;
    ctx.drawImage(faceCanvas, 0, 0, size, size);

    const imgData = ctx.getImageData(0, 0, size, size).data;
    const float = new Float32Array(1 * 3 * size * size);

    let idx = 0;
    for (let c = 0; c < 3; c++) {
      for (let i = 0; i < size * size; i++) {
        const r = imgData[i * 4 + 0] / 255;
        const g = imgData[i * 4 + 1] / 255;
        const b = imgData[i * 4 + 2] / 255;
        float[idx++] = c === 0 ? r : c === 1 ? g : b;
      }
    }
    return new ort.Tensor("float32", float, [1, 3, size, size]);
  }

  function softmax(logits: Float32Array) {
    let max = -Infinity;
    for (const v of logits) max = Math.max(max, v);
    const exps = logits.map((v) => Math.exp(v - max));
    const sum = exps.reduce((a, b) => a + b, 0);
    return exps.map((v) => v / sum);
  }

  // 7) Main loop
  async function loop() {
    // ถ้าสั่งหยุดแล้ว ให้เลิกทำทันที
    if (!isRunningRef.current) return;

    try {
      const cv = cvRef.current;
      const faceCascade = faceCascadeRef.current;
      const session = sessionRef.current;
      const classes = classesRef.current;
      const video = videoRef.current;
      const canvas = canvasRef.current;

      if (!cv || !faceCascade || !session || !classes || !video || !canvas) {
        // ของยังไม่พร้อม ให้ลองใหม่รอบหน้า (แต่เช็ค flag ด้วย)
        if (isRunningRef.current) requestAnimationFrame(loop);
        return;
      }

      // ถ้า video หยุดเล่นไปแล้ว (เช่น ถูก stopCamera แย่งปิด)
      if (video.paused || video.ended) {
         if (isRunningRef.current) requestAnimationFrame(loop);
         return;
      }

      const ctx = canvas.getContext("2d")!;
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      ctx.drawImage(video, 0, 0);

      const src = cv.imread(canvas);
      const gray = new cv.Mat();
      cv.cvtColor(src, gray, cv.COLOR_RGBA2GRAY);

      const faces = new cv.RectVector();
      const msize = new cv.Size(0, 0);
      faceCascade.detectMultiScale(gray, faces, 1.1, 3, 0, msize, msize);

      let bestRect: any = null;
      let bestArea = 0;

      for (let i = 0; i < faces.size(); i++) {
        const r = faces.get(i);
        const area = r.width * r.height;
        if (area > bestArea) {
          bestArea = area;
          bestRect = r;
        }
        ctx.strokeStyle = "lime";
        ctx.lineWidth = 2;
        ctx.strokeRect(r.x, r.y, r.width, r.height);
      }

      if (bestRect) {
        const faceCanvas = document.createElement("canvas");
        faceCanvas.width = bestRect.width;
        faceCanvas.height = bestRect.height;
        const fctx = faceCanvas.getContext("2d")!;
        fctx.drawImage(
          canvas,
          bestRect.x, bestRect.y, bestRect.width, bestRect.height,
          0, 0, bestRect.width, bestRect.height
        );

        const input = preprocessToTensor(faceCanvas);
        const feeds: Record<string, ort.Tensor> = {};
        feeds[session.inputNames[0]] = input;

        const out = await session.run(feeds);
        const outName = session.outputNames[0];
        const logits = out[outName].data as Float32Array;

        const probs = softmax(logits);
        let maxIdx = 0;
        for (let i = 1; i < probs.length; i++) {
          if (probs[i] > probs[maxIdx]) maxIdx = i;
        }

        setEmotion(classes[maxIdx] ?? `class_${maxIdx}`);
        setConf(probs[maxIdx] ?? 0);

        ctx.fillStyle = "rgba(0,0,0,0.6)";
        ctx.fillRect(bestRect.x, Math.max(0, bestRect.y - 28), 220, 28);
        ctx.fillStyle = "white";
        ctx.font = "16px sans-serif";
        ctx.fillText(
          `${classes[maxIdx]} ${(probs[maxIdx] * 100).toFixed(1)}%`,
          bestRect.x + 6,
          bestRect.y - 8
        );
      }

      src.delete();
      gray.delete();
      faces.delete();

      // เรียก loop รอบถัดไปเฉพาะถ้า flag ยังเป็น true
      if (isRunningRef.current) {
        requestAnimationFrame(loop);
      }

    } catch (e: any) {
      console.error(e);
      setStatus(`ผิดพลาดขณะทำงาน: ${e?.message ?? e}`);
      isRunningRef.current = false; // หยุด loop ถ้า error ร้ายแรง
    }
  }

  // Boot sequence
  useEffect(() => {
    let mounted = true;
    (async () => {
      try {
        setStatus("กำลังโหลด OpenCV...");
        await loadOpenCV();
        if (!mounted) return;

        setStatus("กำลังโหลด Haar cascade...");
        await loadCascade();
        if (!mounted) return;

        setStatus("กำลังโหลดโมเดล ONNX...");
        await loadModel();
        if (!mounted) return;

        setStatus("พร้อม เริ่มกดปุ่ม Start");
      } catch (e: any) {
        if (mounted) setStatus(`เริ่มต้นไม่สำเร็จ: ${e?.message ?? e}`);
      }
    })();
    return () => { mounted = false; };
  }, []);

  return (
    <main className="min-h-screen p-6 space-y-4">
      <h1 className="text-2xl font-bold">Face Emotion (OpenCV + YOLO11-CLS)</h1>

      <div className="space-y-2">
        <div className="text-sm">สถานะ: {status}</div>
        <div className="text-sm">
          Emotion: <b>{emotion}</b> | Conf: <b>{(conf * 100).toFixed(1)}%</b>
        </div>
      </div>

      <div className="flex gap-3">
        <button
          className="px-4 py-2 rounded bg-green-600 text-white hover:bg-green-700 transition"
          onClick={startCamera}
        >
          Start Camera
        </button>
        
        {/* ปุ่ม Stop Camera ใหม่ */}
        <button
          className="px-4 py-2 rounded bg-red-600 text-white hover:bg-red-700 transition"
          onClick={stopCamera}
        >
          Stop Camera
        </button>
      </div>

      <div className="relative w-full max-w-3xl">
        <video ref={videoRef} className="hidden" playsInline />
        <canvas
          ref={canvasRef}
          className="w-full rounded border bg-black/5"
        />
      </div>

      <p className="text-sm text-stone-500">
        หมายเหตุ: ต้องกดปุ่ม Start เพื่อขอสิทธิ์เปิดกล้อง
      </p>
    </main>
  );
}