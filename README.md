# Qwen2-VL-7B-Instruct-vlm-finetuning

# Vision Language Model Fine-tuning with LoRA

Fine-tune the Qwen2-VL-7B-Instruct model on the ChartQA dataset using LoRA (Low-Rank Adaptation) and 4-bit quantization for efficient training.

## 📋 Overview

This project demonstrates how to fine-tune a Vision Language Model (VLM) to analyze and interpret visual data, specifically focusing on chart question-answering tasks. The implementation uses parameter-efficient fine-tuning techniques to reduce memory requirements while maintaining model performance.

## ✨ Features

- **4-bit Quantization**: Reduces memory footprint using BitsAndBytes
- **LoRA Adaptation**: Efficient fine-tuning with minimal trainable parameters
- **Gradient Checkpointing**: Trade computation time for memory efficiency
- **ChartQA Dataset**: Specialized dataset for chart understanding
- **Before/After Evaluation**: Compare model performance pre and post fine-tuning
- **Automatic Memory Management**: Clears GPU memory between training and inference

## 🔧 Requirements

```bash
pip install torch transformers datasets peft trl bitsandbytes accelerate
```





## 📊 Model Details

| Parameter | Value |
|-----------|-------|
| Base Model | Qwen2-VL-7B-Instruct |
| Dataset | HuggingFaceM4/ChartQA |
| Training Split | 1% of train/val/test |
| Training Method | Supervised Fine-tuning with LoRA |
| Quantization | 4-bit NF4 with double quantization |
| GPU Memory | ~12GB VRAM |

## ⚙️ Configuration

### Hyperparameters

```python
EPOCHS = 1
BATCH_SIZE = 1
LEARNING_RATE = 2e-5
MAX_SEQ_LEN = 128
GRADIENT_CHECKPOINTING = True
OPTIM = "paged_adamw_32bit"
```

### LoRA Configuration

```python
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.1
TARGET_MODULES = ["q_proj", "v_proj"]
```


## 💻 Usage

### Basic Training

The script performs the following steps:

1. **Load Dataset**: Downloads and formats ChartQA dataset (1% split)
2. **Initialize Model**: Loads Qwen2-VL with 4-bit quantization
3. **Pre-training Evaluation**: Tests model before fine-tuning
4. **Apply LoRA**: Adds trainable adapters to attention layers
5. **Train**: Fine-tunes for 1 epoch with evaluation checkpoints
6. **Save Best Model**: Stores adapter weights based on eval loss
7. **Post-training Evaluation**: Tests fine-tuned model



### Custom Dataset

To use your own dataset, modify the `format_data` function:

```python
def format_data(sample):
    return [
        {
            "role": "system",
            "content": [{"type": "text", "text": system_message}],
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": sample["your_image_field"]},
                {"type": "text", "text": sample["your_query_field"]},
            ],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": sample["your_answer_field"]}],
        },
    ]
```

### Inference Only

To use a fine-tuned model for inference:

```python
from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_ID = "Qwen/Qwen2-VL-7B-Instruct"

# Load base model with quantization
model = Qwen2VLForConditionalGeneration.from_pretrained(
    MODEL_ID, 
    device_map="auto",
    quantization_config=bnb_config,
    use_cache=True
)

# Load fine-tuned adapter
model.load_adapter("./output")

# Load processor
processor = Qwen2VLProcessor.from_pretrained(MODEL_ID)

# Generate predictions
# [Your inference code here]
```

## 🎯 Dataset Format

The ChartQA dataset is automatically formatted into a conversational structure:

```python
[
    {
        "role": "system",
        "content": [{"type": "text", "text": "System prompt..."}]
    },
    {
        "role": "user",
        "content": [
            {"type": "image", "image": <PIL.Image>},
            {"type": "text", "text": "What is the value of...?"}
        ]
    },
    {
        "role": "assistant",
        "content": [{"type": "text", "text": "The value is 42"}]
    }
]
```

## 🔍 Training Details

### Memory Optimization Techniques

1. **4-bit Quantization**: Reduces model size by ~75%
2. **Gradient Checkpointing**: Trades computation for memory
3. **LoRA**: Only trains 0.07% of parameters
4. **Paged Optimizer**: Efficient memory usage with `paged_adamw_32bit`

### Evaluation Strategy

- **Steps-based Evaluation**: Evaluates every 50 steps
- **Best Model Selection**: Saves model with lowest `eval_loss`
- **Automatic Loading**: Loads best checkpoint at training end

### Memory Clearing

The script includes automatic memory management:

```python
def clear_memory():
    # Clears variables, runs garbage collection
    # Empties CUDA cache and synchronizes
    # Prints GPU memory usage
```

## 📈 Results

The script outputs three key metrics:

1. **Pre-training Performance**: Model accuracy before fine-tuning
2. **Training Progress**: Loss curves during training
3. **Post-training Performance**: Model accuracy after fine-tuning

Example output:
```
Generated Answer: [Model's prediction]
Actual Answer: [Ground truth label]
```

## 🖥️ Hardware Requirements

| Requirement | Specification |
|-------------|---------------|
| **Minimum GPU** | 12GB VRAM (RTX 3060, T4) |
| **Recommended GPU** | 16GB+ VRAM (RTX 4080, A10G, V100) |
| **RAM** | 16GB system RAM |
| **Storage** | 20GB free space |

### GPU Memory Usage

- **Model Loading**: ~8GB
- **Training (batch=1)**: ~11GB
- **Inference**: ~8GB

## 🐛 Troubleshooting

### Out of Memory Errors

**Solution 1: Reduce Batch Size**
```python
BATCH_SIZE = 1  # Already at minimum
```

**Solution 2: Reduce Sequence Length**
```python
MAX_SEQ_LEN = 64  # Reduce from 128
```

**Solution 3: Use Gradient Accumulation**
```python
training_args = SFTConfig(
    gradient_accumulation_steps=4,
    # ... other args
)
```

### Slow Training

**Enable Mixed Precision** (if supported):
```python
training_args = SFTConfig(
    fp16=True,  # or bf16=True for newer GPUs
    # ... other args
)
```

### CUDA Out of Memory During Inference

Run the `clear_memory()` function before inference to free up GPU memory.

## 📝 Notes

- WandB logging is disabled by default (`WANDB_DISABLED=true`)
- The script automatically detects CUDA availability
- Model checkpoints are saved in `./output/` directory
- Only 1% of ChartQA is used for quick demonstration
- For production, increase dataset size and training epochs

## 🔄 Advanced Usage

### Training on Full Dataset

```python
train_dataset, eval_dataset, test_dataset = load_dataset(
    "HuggingFaceM4/ChartQA", 
    split=["train", "val", "test"]  # Remove [:1%]
)
```

### Multi-GPU Training

```python
training_args = SFTConfig(
    # ... other args
    ddp_find_unused_parameters=False,
)
```

Then run with:
```bash
torchrun --nproc_per_node=4 train.py
```

### Custom LoRA Targets

```python
peft_config = LoraConfig(
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],  # Add more layers
    r=16,  # Increase rank
)
```

## 📚 Resources

- [Qwen2-VL Model Card](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct)
- [ChartQA Dataset](https://huggingface.co/datasets/HuggingFaceM4/ChartQA)
- [PEFT Documentation](https://huggingface.co/docs/peft)
- [TRL Documentation](https://huggingface.co/docs/trl)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)

## 📄 Citation

If you use this code in your research, please cite:

```bibtex
@article{qwen2vl,
  title={Qwen2-VL: Enhancing Vision-Language Model's Perception of the World at Any Resolution},
  author={Qwen Team},
  year={2024}
}

@article{hu2021lora,
  title={LoRA: Low-Rank Adaptation of Large Language Models},
  author={Hu, Edward J and Shen, Yelong and Wallis, Phillip and Allen-Zhu, Zeyuan and Li, Yuanzhi and Wang, Shean and Wang, Lu and Chen, Weizhu},
  journal={arXiv preprint arXiv:2106.09685},
  year={2021}
}
```

## 📜 License

This project is provided for educational and research purposes. Please refer to the original model and dataset licenses:

- [Qwen2-VL License](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct)
- [ChartQA Dataset License](https://huggingface.co/datasets/HuggingFaceM4/ChartQA)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📧 Contact

For questions or issues, please open an issue on GitHub or contact [your-email@example.com]

## 🙏 Acknowledgments

- **Qwen Team** - For the amazing Qwen2-VL model
- **HuggingFace** - For the ChartQA dataset, transformers, PEFT, and TRL libraries
- **Community** - For valuable feedback and contributions

---

⭐ If you find this project helpful, please consider giving it a star!

**Made with ❤️ for the VLM community**
