"""
Real-Time Image Classifier
===========================
Live image classification using ONNX Runtime with webcam support,
batch processing, and performance monitoring.

Supports: Webcam stream, image files, image directories, and URLs.

Author: SQ1111
License: MIT
"""

import numpy as np
import time
import json
import os
import sys
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass


# ============================================================
# Configuration
# ============================================================
@dataclass
class ClassifierConfig:
    model_path: str = "model_int8.onnx"
    image_size: int = 224
    confidence_threshold: float = 0.5
    top_k: int = 5
    mean: tuple = (0.485, 0.456, 0.406)
    std: tuple = (0.229, 0.224, 0.225)


# ImageNet class names (top 20 for demo)
IMAGENET_CLASSES = {
    0: "tench", 1: "goldfish", 2: "great_white_shark",
    3: "tiger_shark", 4: "hammerhead", 5: "electric_ray",
    6: "stingray", 7: "cock", 8: "hen", 9: "ostrich",
    10: "brambling", 11: "goldfinch", 12: "house_finch",
    13: "junco", 14: "indigo_bunting", 15: "robin",
    16: "bulbul", 17: "jay", 18: "magpie", 19: "chickadee",
}


# ============================================================
# Image Preprocessor
# ============================================================
class ImagePreprocessor:
    """Handles image loading and preprocessing for inference."""
    
    def __init__(self, config: ClassifierConfig):
        self.config = config
        self.mean = np.array(config.mean, dtype=np.float32)
        self.std = np.array(config.std, dtype=np.float32)
    
    def preprocess_numpy(self, image: np.ndarray) -> np.ndarray:
        """Preprocess a numpy array (H, W, C) BGR/RGB image."""
        from PIL import Image
        
        # Resize
        img = Image.fromarray(image)
        img = img.resize(
            (self.config.image_size, self.config.image_size),
            Image.BILINEAR
        )
        
        # Normalize
        arr = np.array(img, dtype=np.float32) / 255.0
        arr = (arr - self.mean) / self.std
        
        # HWC → CHW → NCHW
        arr = np.transpose(arr, (2, 0, 1))
        return np.expand_dims(arr, axis=0)
    
    def preprocess_file(self, file_path: str) -> np.ndarray:
        """Load and preprocess an image file."""
        from PIL import Image
        
        img = Image.open(file_path).convert("RGB")
        return self.preprocess_numpy(np.array(img))
    
    def preprocess_batch(self, file_paths: List[str]) -> np.ndarray:
        """Preprocess a batch of image files."""
        images = [self.preprocess_file(p) for p in file_paths]
        return np.concatenate(images, axis=0)


# ============================================================
# ONNX Inference Engine
# ============================================================
class InferenceEngine:
    """ONNX Runtime inference engine with performance tracking."""
    
    def __init__(self, model_path: str):
        import onnxruntime as ort
        
        # Select best available provider
        available = ort.get_available_providers()
        providers = []
        if "CoreMLExecutionProvider" in available:
            providers.append("CoreMLExecutionProvider")
        if "CUDAExecutionProvider" in available:
            providers.append("CUDAExecutionProvider")
        providers.append("CPUExecutionProvider")
        
        self.session = ort.InferenceSession(model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.active_provider = self.session.get_providers()[0]
        
        # Performance tracking
        self.latency_history: List[float] = []
        self.total_inferences = 0
    
    def predict(self, input_tensor: np.ndarray) -> np.ndarray:
        """Run inference and track latency."""
        start = time.perf_counter()
        outputs = self.session.run(None, {self.input_name: input_tensor})
        elapsed = (time.perf_counter() - start) * 1000
        
        self.latency_history.append(elapsed)
        self.total_inferences += 1
        
        # Keep last 1000 latencies for rolling stats
        if len(self.latency_history) > 1000:
            self.latency_history = self.latency_history[-1000:]
        
        return outputs[0]
    
    def get_stats(self) -> Dict:
        """Get rolling performance statistics."""
        if not self.latency_history:
            return {}
        
        latencies = sorted(self.latency_history)
        n = len(latencies)
        mean = sum(latencies) / n
        
        return {
            "total_inferences": self.total_inferences,
            "provider": self.active_provider,
            "p50_ms": round(latencies[n // 2], 2),
            "p95_ms": round(latencies[int(n * 0.95)], 2),
            "p99_ms": round(latencies[int(n * 0.99)], 2),
            "mean_ms": round(mean, 2),
            "fps": round(1000 / mean, 1),
            "last_ms": round(latencies[-1], 2),
        }


# ============================================================
# Classifier
# ============================================================
class RealTimeClassifier:
    """High-level classifier combining preprocessing and inference."""
    
    def __init__(self, config: ClassifierConfig):
        self.config = config
        self.preprocessor = ImagePreprocessor(config)
        self.engine = InferenceEngine(config.model_path)
        
        print(f"✓ Model loaded: {config.model_path}")
        print(f"  Provider: {self.engine.active_provider}")
    
    def classify_image(self, image_path: str) -> Dict:
        """Classify a single image file."""
        input_tensor = self.preprocessor.preprocess_file(image_path)
        logits = self.engine.predict(input_tensor)
        return self._postprocess(logits[0], image_path)
    
    def classify_numpy(self, image: np.ndarray) -> Dict:
        """Classify a numpy array image (for webcam frames)."""
        input_tensor = self.preprocessor.preprocess_numpy(image)
        logits = self.engine.predict(input_tensor)
        return self._postprocess(logits[0])
    
    def classify_directory(self, dir_path: str) -> List[Dict]:
        """Classify all images in a directory."""
        extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        image_files = [
            str(f) for f in Path(dir_path).iterdir()
            if f.suffix.lower() in extensions
        ]
        
        results = []
        for i, path in enumerate(sorted(image_files)):
            result = self.classify_image(path)
            results.append(result)
            
            if (i + 1) % 10 == 0:
                stats = self.engine.get_stats()
                print(f"  Processed {i+1}/{len(image_files)} | "
                      f"Avg: {stats['mean_ms']:.1f}ms | "
                      f"FPS: {stats['fps']:.0f}")
        
        return results
    
    def _postprocess(self, logits: np.ndarray, 
                     source: str = "frame") -> Dict:
        """Convert logits to predictions."""
        # Softmax
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / exp_logits.sum()
        
        # Top-K
        top_indices = np.argsort(probs)[::-1][:self.config.top_k]
        
        predictions = []
        for idx in top_indices:
            class_name = IMAGENET_CLASSES.get(idx, f"class_{idx}")
            conf = float(probs[idx])
            predictions.append({
                "class_id": int(idx),
                "class_name": class_name,
                "confidence": round(conf, 4),
            })
        
        return {
            "source": source,
            "predicted_class": predictions[0]["class_name"],
            "confidence": predictions[0]["confidence"],
            "top_k": predictions,
            "above_threshold": predictions[0]["confidence"] >= self.config.confidence_threshold,
        }
    
    def run_webcam(self):
        """Run real-time classification on webcam stream."""
        try:
            import cv2
        except ImportError:
            print("Error: opencv-python required for webcam mode")
            print("Install: pip install opencv-python")
            return
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open webcam")
            return
        
        print("\n🎥 Webcam classification started (press 'q' to quit)")
        print("─" * 50)
        
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Classify every 3rd frame for performance
            if frame_count % 3 == 0:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = self.classify_numpy(rgb_frame)
                
                # Draw result on frame
                label = f"{result['predicted_class']} ({result['confidence']:.1%})"
                stats = self.engine.get_stats()
                fps_text = f"FPS: {stats.get('fps', 0):.0f}"
                
                cv2.putText(frame, label, (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, fps_text, (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 0), 2)
            
            cv2.imshow("ML Classifier — SQ1111", frame)
            
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        stats = self.engine.get_stats()
        print(f"\n📊 Session Stats:")
        print(f"   Frames classified: {stats['total_inferences']}")
        print(f"   Average latency:   {stats['mean_ms']:.1f}ms")
        print(f"   P99 latency:       {stats['p99_ms']:.1f}ms")
        print(f"   Throughput:        {stats['fps']:.0f} FPS")


# ============================================================
# CLI
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Real-Time Image Classifier")
    parser.add_argument("--model", default="model_int8.onnx", help="ONNX model path")
    parser.add_argument("--image", help="Classify single image")
    parser.add_argument("--dir", help="Classify directory of images")
    parser.add_argument("--webcam", action="store_true", help="Live webcam mode")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--top-k", type=int, default=5)
    
    args = parser.parse_args()
    
    config = ClassifierConfig(
        model_path=args.model,
        confidence_threshold=args.threshold,
        top_k=args.top_k,
    )
    
    classifier = RealTimeClassifier(config)
    
    if args.webcam:
        classifier.run_webcam()
    elif args.image:
        result = classifier.classify_image(args.image)
        print(json.dumps(result, indent=2))
    elif args.dir:
        results = classifier.classify_directory(args.dir)
        print(f"\n✓ Classified {len(results)} images")
        stats = classifier.engine.get_stats()
        print(f"  Average: {stats['mean_ms']:.1f}ms | FPS: {stats['fps']:.0f}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
