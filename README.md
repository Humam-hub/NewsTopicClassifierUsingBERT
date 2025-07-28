# BERT News Classification

A web-based news classification application that uses BERT (Bidirectional Encoder Representations from Transformers) to classify news headlines into four categories: World, Sports, Business, and Sci/Tech.

## 🚀 Features

- **Real-time Classification**: Classify news headlines instantly
- **Web Interface**: User-friendly Gradio web interface
- **Multi-category Support**: Classifies into 4 main news categories
- **GPU Support**: Automatically uses GPU if available, falls back to CPU
- **Confidence Scores**: Shows probability scores for each category

## 📋 Requirements

- Python 3.7+
- PyTorch
- Transformers (Hugging Face)
- Gradio
- NumPy
- Scikit-learn

## 🛠️ Installation

1. **Clone or download this repository**
   ```bash
   git clone <repository-url>
   cd "BERT News Classification"
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download the BERT model**
   
   You need to have a fine-tuned BERT model for AG News classification. The model should be in a directory called `ag_news_bert_model` in the project root, containing:
   - `config.json`
   - `pytorch_model.bin` (or similar model weights file)
   - `tokenizer.json` and other tokenizer files

   If you don't have a trained model, you can:
   - Train your own model on the AG News dataset
   - Use a pre-trained model and adapt the categories
   - Download a publicly available AG News BERT model

## 🚀 Usage

1. **Run the application**
   ```bash
   python app.py
   ```

2. **Access the web interface**
   - The application will start a local web server
   - Open your browser and go to the URL shown in the terminal (usually `http://127.0.0.1:7860`)

3. **Classify news headlines**
   - Enter a news headline in the text box
   - Click submit or press Enter
   - View the classification results with confidence scores

## 📊 Categories

The model classifies news into four main categories:

- **World**: International news, politics, global events
- **Sports**: Sports news, athletics, competitions
- **Business**: Business news, economics, finance, markets
- **Sci/Tech**: Science, technology, research, innovation

## 🔧 Technical Details

### Model Architecture
- **Base Model**: BERT (Bidirectional Encoder Representations from Transformers)
- **Task**: Sequence Classification
- **Input**: News headlines (max length: 128 tokens)
- **Output**: Probability distribution over 4 categories

### Key Components
- **Tokenization**: BERT tokenizer with padding and truncation
- **Model**: Fine-tuned BERT for sequence classification
- **Inference**: Softmax activation for probability scores
- **Interface**: Gradio web interface for easy interaction

## 📁 Project Structure

```
BERT News Classification/
├── app.py              # Main application file
├── requirements.txt    # Python dependencies
├── README.md          # This file
└── ag_news_bert_model/  # BERT model files (you need to add this)
    ├── config.json
    ├── pytorch_model.bin
    ├── tokenizer.json
    └── ...
```

## 🐛 Troubleshooting

### Common Issues

1. **Model not found error**
   - Ensure the `ag_news_bert_model` directory exists in the project root
   - Check that all required model files are present

2. **CUDA/GPU issues**
   - The app automatically detects and uses GPU if available
   - Falls back to CPU if GPU is not available

3. **Import errors**
   - Make sure all dependencies are installed: `pip install -r requirements.txt`

4. **Port already in use**
   - Gradio will automatically find an available port
   - Check the terminal output for the correct URL

## 🤝 Contributing

Feel free to contribute to this project by:
- Reporting bugs
- Suggesting new features
- Improving the documentation
- Optimizing the model performance

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 🙏 Acknowledgments

- Hugging Face for the Transformers library
- Google Research for BERT
- Gradio for the web interface framework
- AG News dataset for training data

---

**Note**: This application requires a pre-trained BERT model fine-tuned on the AG News dataset. Make sure you have the model files before running the application. 