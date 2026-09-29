# Model checkpoints directory
#
# After training, fine-tuned checkpoints will appear here in subdirectories:
#
#   models/checkpoints/bert/best_model/       ← BERT fine-tuned weights
#   models/checkpoints/xlmr/best_model/       ← XLM-R fine-tuned weights
#
# Place your downloaded checkpoint folders here and the app will auto-detect them.
# Expected structure for each checkpoint:
#
#   best_model/
#   ├── config.json
#   ├── model.safetensors  (or pytorch_model.bin)
#   ├── tokenizer_config.json
#   ├── vocab.txt / sentencepiece.bpe.model
#   └── special_tokens_map.json
