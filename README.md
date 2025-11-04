# LLaMA-Token-Generation-Latency

This project benchmarks local large language model (LLM) performance using [Ollama](https://ollama.ai/) across different *batch sizes* and *prompt types*.  
It measures key metrics like token latency, throughput, and GPU utilization to analyze model efficiency and scalability.

---

## Requirements

Ensure the following are installed:

- **Python 3.9+**
- **Ollama Desktop** (or CLI) — [Download here](https://ollama.ai/download)
- **Jupyter Notebook**
- **NVIDIA GPU** (optional but recommended for GPU stats as that was the types of GPUs used)
- Python libraries:

```bash
pip install pandas numpy psutil pynvml tqdm
