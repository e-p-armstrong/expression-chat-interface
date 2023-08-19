from optimum.onnxruntime import ORTModelForSequenceClassification, ORTQuantizer
from optimum.onnxruntime.configuration import AutoQuantizationConfig
from optimum.pipelines import pipeline

model_id = "bhadresh-savani/distilbert-base-uncased-emotion"
tokenizer = AutoTokenizer.from_pretrained(model_id)
save_path = Path("optumum_model")
save_path.mkdir(exist_ok=True)

quantizer = ORTQuantizer.from_pretrained(model_id, feature="sequence-classification")
qconfig = AutoQuantizationConfig.avx512_vnni(is_static=False, per_channel=True)

quantizer.export(
    onnx_model_path=save_path / "model.onnx",
    onnx_quantized_model_output_path=save_path / "model-quantized.onnx",
    quantization_config=qconfig,
    )
quantizer.model.config.save_pretrained(save_path) # saves config.json 