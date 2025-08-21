# Neuron Embeddings Interactive Visualization

An interactive web application for exploring how neurons evolve during training across different checkpoints.

## Features

- **Interactive Neuron Selection**: Choose any layer (0-5) and neuron (0-2000, in steps of 20)
- **Checkpoint Comparison**: Compare neuron behavior at different training checkpoints
- **Dynamic Token Highlighting**: See the most common words/fragments that activate each neuron
- **Real-time Analysis**: Automatically analyzes text patterns for each cluster

## Local Development

### Prerequisites
- Python 3.8 or higher
- Neuron data files in `results/` and `results/pythia70m/` directories

### Installation
```bash
pip install -r requirements.txt
```

### Run Locally
```bash
streamlit run streamlit_app.py
```

## Online Deployment

### Option 1: Streamlit Community Cloud (Free & Recommended)

1. **Push to GitHub**:
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git remote add origin https://github.com/yourusername/neuron-embeddings.git
   git push -u origin main
   ```

2. **Deploy on Streamlit Cloud**:
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Click "New app"
   - Connect your GitHub repository
   - Set main file as `streamlit_app.py`
   - Click "Deploy"

3. **Your app will be live at**: `https://yourusername-neuron-embeddings-streamlit-app-hash.streamlit.app`

### Option 2: Alternative Platforms

**Heroku**:
- Add `Procfile`: `web: streamlit run streamlit_app.py --server.port=$PORT --server.address=0.0.0.0`
- Deploy via Git or GitHub integration

**Railway**:
- Connect GitHub repo
- Railway auto-detects Streamlit apps
- Automatic HTTPS and custom domains

**Render**:
- Connect GitHub repo  
- Build command: `pip install -r requirements.txt`
- Start command: `streamlit run streamlit_app.py --server.port=$PORT --server.address=0.0.0.0`

## Data Requirements

The app expects neuron data files in JSONL format:
- Location: `results/` and `results/pythia70m/` directories
- Format: `L{layer}N{neuron}_pythia70m_ckpt_series.jsonl`
- Content: One JSON object per line with `checkpoint_step`, `cluster_labels`, and `text_examples`

## Configuration

- **Highlighting**: Top 2 most frequent words/fragments per cluster
- **Fragment threshold**: Minimum 5 occurrences
- **Display limit**: First 5 examples per cluster (for performance)

## Performance Notes

- Data is cached using `@st.cache_data` for faster loading
- Large datasets may take time to initially load
- Consider data sampling for very large deployments
