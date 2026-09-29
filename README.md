# AutoReview - AI-Based Automotive Review & Customer Sentiment Analytics


> Aspect-Based Sentiment Analysis (ABSA) for automotive customer reviews,  
> powered by BERT and XLM-RoBERTa transformers.

---

## Features

| Feature | Description |
|---------|-------------|
| **Single review analysis** | Enter any review → detect aspects → classify sentiment per aspect |
| **Multi-aspect support** | One review can yield multiple aspect-sentiment pairs |
| **Confidence scores** | Softmax probability for each prediction |
| **Batch analysis** | Upload CSV/JSON or use the built-in synthetic dataset |
| **Aggregate analytics** | Sentiment distribution, aspect frequency, heatmaps |
| **Model training** | Fine-tune BERT or XLM-R on your own data |
| **Model evaluation** | Accuracy, Precision, Recall, F1, Confusion Matrix, Radar comparison |
| **Export** | Download results as CSV, JSON, or Excel |
| **Dark UI** | Streamlit dashboard with Plotly interactive charts |

---

## Supported Automotive Aspects

Vehicle · Engine · Battery · Mileage · Safety · Comfort · Service · Infotainment · Price

Configurable in `config.yaml` — add any aspect without touching code.

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the application

```bash
python run.py
# or
streamlit run app/main.py
```

### 3. Open in browser

```
http://localhost:8501
```

### 4. Use the app

1. Select **BERT** or **XLM-RoBERTa** in the sidebar → click **⚡ Load Model**
2. Go to **🏠 Single Review** → paste a review → **🚀 Analyse**
3. Go to **📊 Batch Analytics** → upload data or use synthetic → **🚀 Run**
4. Go to **🎓 Model Training** → train on your dataset
5. Go to **📈 Model Evaluation** → compare model performance

---

## Project Structure

```
├── app/
│   ├── main.py                  # Streamlit entry point + global CSS
│   ├── ui_components.py         # Reusable HTML/Streamlit components
│   └── pages/
│       ├── page_home.py         # Single review analysis page
│       ├── page_batch.py        # Batch analytics page
│       ├── page_training.py     # Model training page
│       ├── page_evaluation.py   # Model evaluation page
│       └── page_about.py        # About / documentation page
├── src/
│   ├── config_loader.py         # Centralised YAML config access
│   ├── data/
│   │   ├── dataset_generator.py # Synthetic ABSA dataset generator
│   │   ├── preprocessor.py      # Text preprocessing pipeline
│   │   └── validator.py         # Input & file validation
│   ├── models/
│   │   ├── absa_dataset.py      # PyTorch Dataset (sentence-pair encoding)
│   │   ├── model_factory.py     # Tokenizer/model loader + device selection
│   │   └── trainer.py           # HuggingFace Trainer fine-tuning pipeline
│   ├── inference/
│   │   ├── aspect_extractor.py  # Keyword-based aspect extraction
│   │   ├── sentiment_classifier.py  # Transformer sentiment inference
│   │   └── inference_engine.py  # Orchestrator (single + batch)
│   ├── analytics/
│   │   ├── analytics_engine.py  # Aggregate metrics computation
│   │   └── visualizer.py        # Plotly dark-themed charts
│   └── export/
│       └── export_service.py    # CSV / JSON / Excel export
├── config.yaml                  # Centralized configuration
├── requirements.txt
└── run.py                       # Convenience launcher
```

---

## Dataset Format

```csv
Review,Aspect,Sentiment
"The mileage is great but service is poor.",Mileage,Positive
"The mileage is great but service is poor.",Service,Negative
"Excellent engine performance.",Engine,Positive
```

Each row = one aspect-sentiment annotation for a review.

**Split:** 70% train / 15% validation / 15% test (configurable in `config.yaml`).

---

## Models

| Model | HuggingFace ID | Best For |
|-------|---------------|---------|
| BERT | `bert-base-uncased` | English reviews, fast |
| XLM-RoBERTa | `xlm-roberta-base` | Multilingual, better generalisation |

**Encoding:** Sentence-pair format — aspect as Sentence A, review as Sentence B.  
This gives the model aspect-conditioned context during classification.

---

## Hardware Requirements

| | Minimum | Recommended |
|-|---------|-------------|
| CPU | Modern multi-core | — |
| RAM | 8 GB | 16 GB |
| GPU | Not required (inference) | NVIDIA CUDA GPU (training) |

---

## Configuration

All system parameters are in [`config.yaml`](config.yaml):

- **Aspect taxonomy** — add/remove aspects and keywords
- **Model settings** — name, max token length
- **Training** — epochs, batch size, learning rate, dataset split
- **Export** — output directory
- **Dataset** — accepted file types, max size, required columns

---

## Constraints (as per SRS)

- Aspect extraction is keyword-based; **implicit aspects are not detected**.
- Sarcasm and figurative language may reduce accuracy.
- Confidence scores are model probability estimates, not guaranteed correctness.
- The system is an **analytical tool**, not a substitute for human decisions.

---

## License

This project is for academic and research purposes.
