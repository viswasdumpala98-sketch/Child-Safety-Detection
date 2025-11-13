# app.py

import gradio as gr
from predict import predict_image, predict_video
from realtime_webcam import run_live_detection

def trigger_webcam():
    return run_live_detection()

with gr.Blocks() as demo:
    gr.Markdown("## 🧒🔍 Child Safety Object Detection")

    with gr.Tab("Upload Image"):
        img_input = gr.Image(type="filepath", label="Upload Image")
        img_output = gr.Image(label="Detected Image")
        img_status = gr.Textbox(label="Detection Status")
        img_button = gr.Button("Detect Objects")
        img_button.click(fn=predict_image, inputs=img_input, outputs=[img_output, img_status])

    with gr.Tab("Upload Video"):
        vid_input = gr.Video(label="Upload Video")
        vid_output = gr.Video(label="Detected Video")
        vid_status = gr.Textbox(label="Detection Status")
        vid_button = gr.Button("Detect Objects")
        vid_button.click(fn=predict_video, inputs=vid_input, outputs=[vid_output, vid_status])

    with gr.Tab("🔴 Live Webcam (Real-time)"):
        webcam_button = gr.Button("Launch Live Webcam Detection")
        webcam_status = gr.Textbox(label="Live Detection Status")
        webcam_button.click(fn=trigger_webcam, outputs=webcam_status)

demo.launch()
