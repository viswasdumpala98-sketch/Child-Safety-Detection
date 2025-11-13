from ultralytics import YOLO
import matplotlib.pyplot as plt
import pandas as pd
import torch  # <-- Add this import at the top

def train_yolov8():
    # --- Configuration ---
    config = {
        "model": "yolov8n.pt",       # Pretrained model
        "data": "data.yaml",          # Dataset config
        "epochs": 20,                # Stop after 20 epochs
        "imgsz": 640,                # Image size
        "batch": 16,                 # Batch size
        "name": "child_safety_v1",   # Experiment name
        "patience": 5,               # Early stopping if no improvement
        "device": "0" if torch.cuda.is_available() else "cpu"  # Use GPU if available
    }

    # --- Training ---
    print("🚀 Starting training...")
    model = YOLO(config["model"])
    results = model.train(
        data=config["data"],
        epochs=config["epochs"],
        imgsz=config["imgsz"],
        batch=config["batch"],
        name=config["name"],
        patience=config["patience"],
        device=config["device"]
    )

    # --- Save Results ---
    print("\n📊 Training complete! Saving results...")
    results_path = f"runs/detect/{config['name']}"
    
    # 1. Plot metrics
    plot_metrics(results_path)
    
    # 2. Export model to ONNX
    model.export(format="onnx")
    print(f"✅ Model exported to ONNX: {results_path}/weights/best.onnx")

def plot_metrics(results_path):
    """Plot training metrics from results.csv"""
    try:
        df = pd.read_csv(f"{results_path}/results.csv")
        plt.figure(figsize=(12, 4))
        
        # Plot losses
        plt.subplot(1, 2, 1)
        plt.plot(df["epoch"], df["train/box_loss"], label="Box Loss")
        plt.plot(df["epoch"], df["train/cls_loss"], label="Class Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        
        # Plot mAP
        plt.subplot(1, 2, 2)
        plt.plot(df["epoch"], df["metrics/mAP50"], label="mAP50")
        plt.plot(df["epoch"], df["metrics/mAP50-95"], label="mAP50-95")
        plt.xlabel("Epoch")
        plt.ylabel("mAP")
        plt.legend()
        
        plt.savefig(f"{results_path}/metrics.png")
        print(f"📈 Metrics plot saved: {results_path}/metrics.png")
    except Exception as e:
        print(f"⚠️ Could not plot metrics: {e}")

if __name__ == "__main__":
    train_yolov8()