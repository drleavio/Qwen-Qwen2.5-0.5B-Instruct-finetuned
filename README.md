# EU AI Act LLM Classifier

**Hugging Face Model**: [drlevio/qwen-0.5b-eu-ai-act-classifier](https://huggingface.co/drlevio/qwen-0.5b-eu-ai-act-classifier)

This repository contains scripts to fine-tune and serve an LLM (Qwen 2.5) for classifying AI systems under the EU AI Act.

## Files
- `finetune.py`: Script to fine-tune the model using QLoRA.
- `inference.py`: Script to run local inference with the fine-tuned model in the terminal.
- `api.py`: FastAPI application to host the model as an HTTP endpoint.
- `push_to_hub.py`: Script to push your fine-tuned model to Hugging Face.

## Installation
```bash
pip install -r requirements.txt
```

## Running the API
```bash
python api.py
```
The API will be available on port 3001. Send a POST request to `/classify` with the JSON payload `{"use_case": "your use case text"}`.
