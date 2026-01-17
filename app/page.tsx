"use client";

import { useEffect, useRef, useState } from "react";
import * as ort from "onnxruntime-web";

const emotionMap: Record<string, string> = {
  happy: "😁",
  sad: "😢",
  angry: "😡",
  surprise: "😲",
  neutral: "😐",
  fear: "😱",
  disgust: "🤢",
  "-": "🤖"
};

type CvType = any;

interface FaceResult {
  emotion: string;
  conf: number;
}

export default function Home() {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const isRunningRef = useRef<boolean>(false);
  
  const cvRef = useRef<CvType | null>(null);
  const faceCascadeRef = useRef<any>(null);
  const sessionRef = useRef<ort.InferenceSession | null>(null);
  const classesRef = useRef<string[] | null>(null);

  const [status, setStatus] = useState<string>("กำลังเตรียมระบบ...");
  const [results, setResults] = useState<FaceResult[]>([]);
  const [isStreaming, setIsStreaming] = useState<boolean>(false);

  // --- System Loaders (Logic เดิม) ---
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
      script.onerror = () => reject(new Error("Failed to load OpenCV"));
      document.body.appendChild(script);
    });
  }

  async function loadCascade() {
    const cv = cvRef.current;
    if (!cv) throw new Error("OpenCV not ready");
    const cascadeUrl = "/opencv/haarcascade_frontalface_default.xml";
    const res = await fetch(cascadeUrl);
    if (!res.ok) throw new Error("Failed to load cascade");
    const data = new Uint8Array(await res.arrayBuffer());
    const cascadePath = "haarcascade_frontalface_default.xml";
    try { cv.FS_unlink(cascadePath); } catch { }
    cv.FS_createDataFile("/", cascadePath, data, true, false, false);
    const faceCascade = new cv.CascadeClassifier();
    if (!faceCascade.load(cascadePath)) throw new Error("Cascade load failed");
    faceCascadeRef.current = faceCascade;
  }

  async function loadModel() {
    const session = await ort.InferenceSession.create("/models/emotion_yolo11n_cls.onnx", { executionProviders: ["wasm"] });
    sessionRef.current = session;
    const clsRes = await fetch("/models/classes.json");
    if (!clsRes.ok) throw new Error("Failed to load classes");
    classesRef.current = await clsRes.json();
  }

  // --- Camera Control ---
  async function startCamera() {
    if (isRunningRef.current) return;
    setStatus("ขอสิทธิ์กล้อง...");
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "user" }, audio: false });
      if (!videoRef.current) return;
      videoRef.current.srcObject = stream;
      await videoRef.current.play();
      setStatus("กำลังทำงาน...");
      isRunningRef.current = true;
      setIsStreaming(true);
      requestAnimationFrame(loop);
    } catch (error: any) {
      setStatus(`เปิดกล้องไม่สำเร็จ: ${error.message}`);
    }
  }

  function stopCamera() {
    if (!isRunningRef.current) return;
    isRunningRef.current = false;
    setIsStreaming(false);
    setStatus("หยุดกล้องแล้ว");
    setResults([]);
    if (videoRef.current && videoRef.current.srcObject) {
      const stream = videoRef.current.srcObject as MediaStream;
      stream.getTracks().forEach((track) => track.stop());
      videoRef.current.srcObject = null;
    }
    if (canvasRef.current) {
      const ctx = canvasRef.current.getContext("2d");
      if (ctx) ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
    }
  }

  // --- Logic ---
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

  async function loop() {
    if (!isRunningRef.current) return;
    try {
      const cv = cvRef.current;
      const faceCascade = faceCascadeRef.current;
      const session = sessionRef.current;
      const classes = classesRef.current;
      const video = videoRef.current;
      const canvas = canvasRef.current;

      if (!cv || !faceCascade || !session || !classes || !video || !canvas) {
        if (isRunningRef.current) requestAnimationFrame(loop);
        return;
      }
      if (video.paused || video.ended) {
         if (isRunningRef.current) requestAnimationFrame(loop);
         return;
      }

      const ctx = canvas.getContext("2d", { willReadFrequently: true })!;
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      ctx.drawImage(video, 0, 0);

      const src = cv.imread(canvas);
      const gray = new cv.Mat();
      cv.cvtColor(src, gray, cv.COLOR_RGBA2GRAY);

      const facesVec = new cv.RectVector();
      const msize = new cv.Size(0, 0);
      faceCascade.detectMultiScale(gray, facesVec, 1.1, 3, 0, msize, msize);

      let faceArray = [];
      for (let i = 0; i < facesVec.size(); i++) faceArray.push(facesVec.get(i));
      faceArray.sort((a, b) => (b.width * b.height) - (a.width * a.height));
      const facesToProcess = faceArray.slice(0, 2);
      const currentFrameResults: FaceResult[] = [];

      for (let i = 0; i < facesToProcess.length; i++) {
        const r = facesToProcess[i];
        
        // HUD Style
        ctx.strokeStyle = "#4ade80";
        ctx.lineWidth = 2;
        ctx.strokeRect(r.x, r.y, r.width, r.height);
        const len = Math.min(20, r.width / 4);
        ctx.strokeStyle = "#22c55e";
        ctx.lineWidth = 4;
        ctx.beginPath();
        ctx.moveTo(r.x, r.y + len); ctx.lineTo(r.x, r.y); ctx.lineTo(r.x + len, r.y);
        ctx.moveTo(r.x + r.width - len, r.y); ctx.lineTo(r.x + r.width, r.y); ctx.lineTo(r.x + r.width, r.y + len);
        ctx.moveTo(r.x, r.y + r.height - len); ctx.lineTo(r.x, r.y + r.height); ctx.lineTo(r.x + len, r.y + r.height);
        ctx.moveTo(r.x + r.width - len, r.y + r.height); ctx.lineTo(r.x + r.width, r.y + r.height); ctx.lineTo(r.x + r.width, r.y + r.height - len);
        ctx.stroke();

        const faceCanvas = document.createElement("canvas");
        faceCanvas.width = r.width;
        faceCanvas.height = r.height;
        const fctx = faceCanvas.getContext("2d")!;
        fctx.drawImage(canvas, r.x, r.y, r.width, r.height, 0, 0, r.width, r.height);

        const input = preprocessToTensor(faceCanvas);
        const feeds: Record<string, ort.Tensor> = {};
        feeds[session.inputNames[0]] = input;

        const out = await session.run(feeds);
        const probs = softmax(out[session.outputNames[0]].data as Float32Array);
        
        let maxIdx = 0;
        for (let j = 1; j < probs.length; j++) {
          if (probs[j] > probs[maxIdx]) maxIdx = j;
        }

        const detectedEmotion = classes[maxIdx] ?? `class_${maxIdx}`;
        const confidence = probs[maxIdx] ?? 0;

        currentFrameResults.push({ emotion: detectedEmotion, conf: confidence });

        ctx.fillStyle = "rgba(0, 0, 0, 0.7)";
        ctx.fillRect(r.x, Math.max(0, r.y - 40), 180, 40);
        ctx.fillStyle = "#ffffff";
        ctx.font = "bold 18px Kanit, sans-serif";
        const emoji = emotionMap[detectedEmotion.toLowerCase()] || "";
        ctx.fillText(`${emoji} ${detectedEmotion} ${(confidence * 100).toFixed(0)}%`, r.x + 10, r.y - 14);
      }

      setResults(currentFrameResults);
      src.delete(); gray.delete(); facesVec.delete();
      if (isRunningRef.current) requestAnimationFrame(loop);
    } catch (e) {
      console.error(e);
      isRunningRef.current = false;
      setIsStreaming(false);
    }
  }

  useEffect(() => {
    let mounted = true;
    (async () => {
      try {
        await loadOpenCV(); if (!mounted) return;
        await loadCascade(); if (!mounted) return;
        await loadModel(); if (!mounted) return;
        setStatus("พร้อมใช้งาน เริ่มกดปุ่ม Start");
      } catch (e: any) { if (mounted) setStatus(`Error: ${e.message}`); }
    })();
    return () => { mounted = false; };
  }, []);

  return (
    <main className="min-h-screen flex items-center justify-center p-4 relative overflow-hidden bg-transparent transition-colors duration-300">
      
      {/* 👇👇👇 จุดที่เพิ่มกรอบ (Frame) 👇👇👇
         - bg-white/40: พื้นหลังขาวโปร่งแสง (ปรับเลข 40 เพื่อเปลี่ยนความเข้ม)
         - backdrop-blur-md: เบลอฉากหลังเพื่อให้ข้อความชัดขึ้น
         - border-2 border-white/50: เส้นขอบสีขาวจางๆ
         - shadow-2xl: เงาให้ดูลอยออกมา
      */}
      <div className="relative z-10 w-full max-w-3xl bg-white/40 dark:bg-black/40 backdrop-blur-md rounded-3xl border-2 border-white/50 shadow-2xl p-6 md:p-8 flex flex-col items-center space-y-5">
        
        {/* Header Section */}
        <div className="text-center space-y-3 w-full flex flex-col items-center">
          <h1 className="text-2xl md:text-3xl font-extrabold text-gray-900 dark:text-white tracking-tight drop-shadow-sm">
            Face Emotion <span className="text-indigo-700 dark:text-indigo-400">(Multi-Face)</span>
          </h1>

          <div className="flex flex-col sm:flex-row justify-center items-center gap-2 sm:gap-4">
            <div className="text-sm font-medium text-gray-800 dark:text-gray-200 bg-white/60 dark:bg-white/10 px-4 py-1.5 rounded-full border border-white/30">
              สถานะ: <span className={`${isStreaming ? 'text-green-700 dark:text-green-400' : 'text-gray-600 dark:text-gray-400'} font-bold`}>{status}</span>
            </div>
            
            <div className="text-base font-medium bg-indigo-100/70 dark:bg-indigo-900/40 px-5 py-1.5 rounded-full border border-indigo-200/50 dark:border-indigo-500/30">
               Detected: <span className="text-indigo-800 dark:text-indigo-300 font-bold">{results.length} Faces</span>
            </div>
          </div>
        </div>

        {/* Buttons Section */}
        <div className="flex w-full justify-center gap-4">
          {!isStreaming ? (
            <button onClick={startCamera} className="px-8 py-3 rounded-xl font-bold shadow-lg bg-gradient-to-r from-green-500 to-emerald-600 text-white hover:from-green-400 hover:to-emerald-500 transform transition hover:-translate-y-1 border border-green-400/50">
              Start Camera
            </button>
          ) : (
            <button onClick={stopCamera} className="px-8 py-3 rounded-xl font-bold shadow-lg bg-gradient-to-r from-red-500 to-rose-600 text-white hover:from-red-400 hover:to-rose-500 transform transition hover:-translate-y-1 border border-red-400/50">
              Stop Camera
            </button>
          )}
        </div>

        {/* Emoji Section */}
        <div className="flex gap-6 min-h-[60px] items-center justify-center py-1">
            {results.length === 0 ? (
               <div className="text-4xl opacity-50 grayscale drop-shadow-sm">{emotionMap["-"]}</div>
            ) : (
               results.map((res, idx) => (
                 <div key={idx} className="flex flex-col items-center animate-bounce-short">
                    <div className="text-4xl md:text-5xl drop-shadow-lg transition-transform hover:scale-110">
                      {emotionMap[res.emotion.toLowerCase()] || emotionMap["-"]}
                    </div>
                    <span className="text-[10px] font-bold text-gray-800 dark:text-gray-300 mt-1 shadow-sm">Face {idx + 1}</span>
                 </div>
               ))
            )}
        </div>

        {/* Camera Frame */}
        <div className="relative w-full aspect-video bg-black/90 rounded-2xl overflow-hidden shadow-[0_0_20px_rgba(0,0,0,0.3)] border-[4px] border-white/20 dark:border-gray-700/80 group">
          <video ref={videoRef} className="hidden" playsInline muted />
          <canvas ref={canvasRef} className={`w-full h-full object-contain ${!isStreaming && 'hidden'}`} />
          
          {!isStreaming && (
             <div className="absolute inset-0 flex flex-col items-center justify-center text-gray-400 dark:text-gray-500">
                <span className="text-sm font-medium">Camera Offline</span>
             </div>
          )}

          {isStreaming && (
            <div className="absolute inset-0 z-20 pointer-events-none">
                <div className="w-full h-1 bg-green-400/60 shadow-[0_0_15px_rgba(74,222,128,0.8)] animate-scanline opacity-70"></div>
                <div className="absolute inset-0 bg-[linear-gradient(rgba(18,16,16,0)_50%,rgba(0,0,0,0.1)_50%),linear-gradient(90deg,rgba(255,0,0,0.03),rgba(0,255,0,0.01),rgba(0,0,255,0.03))] z-10 background-size-[100%_3px,3px_100%]"></div>
                
                <div className="absolute top-4 left-4 w-10 h-10 border-t-[3px] border-l-[3px] border-green-500/30 rounded-tl-xl"></div>
                <div className="absolute top-4 right-4 w-10 h-10 border-t-[3px] border-r-[3px] border-green-500/30 rounded-tr-xl"></div>
                <div className="absolute bottom-4 left-4 w-10 h-10 border-b-[3px] border-l-[3px] border-green-500/30 rounded-bl-xl"></div>
                <div className="absolute bottom-4 right-4 w-10 h-10 border-b-[3px] border-r-[3px] border-green-500/30 rounded-br-xl"></div>

                <div className="absolute top-5 right-7 flex items-center gap-2 bg-black/60 px-2 py-1 rounded backdrop-blur-sm border border-white/10">
                    <div className="w-2.5 h-2.5 bg-red-600 rounded-full animate-pulse shadow-[0_0_8px_rgba(220,38,38,0.8)]"></div>
                    <span className="text-white text-[10px] font-mono tracking-widest opacity-80">REC</span>
                </div>
            </div>
          )}
        </div>

        {/* Footer Text */}
        <p className="text-xs text-center text-gray-800 dark:text-gray-300 mt-2 font-bold drop-shadow-sm">
        </p>
      </div>
    </main>
  );
}