# High-level integration utilities for routing multimodal inputs
def route_input(input_type, payload):
    if input_type == 'image':
        from src.image_module.cnn_predict import predict_image
        return predict_image(payload)
    if input_type == 'text':
        from src.text_module.transformer_predict import predict_sentiment
        return predict_sentiment(payload)
    if input_type == 'tabular':
        from src.ml_module.regression import predict_regression_sample
        return predict_regression_sample(payload)
    return None
