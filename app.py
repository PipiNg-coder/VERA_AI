# Import spaces BEFORE torch/transformers are imported by the backend.
import spaces
import gradio as gr

from backend.vera_ai_final_fixed import load_pipeline, predict_image

# Preload the trained bundle and CLIP vision encoder once at startup.
# ZeroGPU intercepts the CUDA placement and allocates the real GPU only
# while a function decorated with @spaces.GPU is running.
load_pipeline()


@spaces.GPU(duration=60)
def classify_image(image):
    if image is None:
        return {"REAL": 0.0, "AI": 0.0}, "Vui lòng tải ảnh lên trước khi phân tích."

    result = predict_image(image)
    ai_probability = float(result["ai_probability"])
    real_probability = 1.0 - ai_probability

    details = (
        f"KẾT LUẬN: {result['label']}\n"
        f"Độ tin cậy theo ngưỡng: {result['confidence'] * 100:.2f}%\n"
        f"Xác suất AI: {ai_probability * 100:.2f}%\n"
        f"Ngưỡng phân loại: {result['threshold']:.4f}\n"
        f"Mô hình: {result['model']}\n"
        f"Thiết bị: {result['device']}\n"
        "\nXác suất từng mô hình:\n"
        + "\n".join(
            f"- {name}: {probability * 100:.2f}% AI"
            for name, probability in result["component_probabilities"].items()
        )
    )
    return {"REAL": real_probability, "AI": ai_probability}, details


with gr.Blocks(title="VERA AI | Image Authenticity Detection", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # VERA AI
        ### Phân tích ảnh REAL / AI-generated
        Tải ảnh JPG, PNG hoặc WEBP để mô hình CLIP + FFT/SRM Ensemble đánh giá.
        """
    )
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(type="pil", label="Ảnh cần phân tích")
            analyze_button = gr.Button("PHÂN TÍCH ẢNH", variant="primary")
        with gr.Column():
            prediction = gr.Label(label="Xác suất phân loại", num_top_classes=2)
            details = gr.Textbox(label="Kết quả chi tiết", lines=12, interactive=False)

    analyze_button.click(
        fn=classify_image,
        inputs=input_image,
        outputs=[prediction, details],
    )
    gr.Markdown(
        "Lưu ý: kết quả là dự đoán của mô hình, không phải bằng chứng tuyệt đối về nguồn gốc ảnh."
    )

demo.queue().launch()
