# OpenROADM YANG CLI Chatbot

AI-powered command-line chatbot for understanding and querying OpenROADM YANG models.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the chatbot
python main.py --yang-dir ./openroadm-yang

# With custom config
python main.py --config config.yaml --log-level DEBUG
```

## Architecture

```
yang_chatbot/
├── models/           # YANG model processing with semantic chunking
├── llm/              # LLM integration with RAG pipeline
├── cli/              # Command-line interface
├── storage/          # Vector DB and keyword search
├── validation/       # Multi-stage validation pipeline
└── utils/            # Utilities and helpers
```

### Core Components

- **YANG Processor**: Parses `.yang` files and extracts semantic chunks (containers, groupings, identities, etc.)
- **RAG System**: Hybrid retrieval (vector + BM25 keyword search) for accurate context retrieval
- **Validation Pipeline**: Multi-stage validation (syntax, semantics, optical constraints, confidence threshold)
- **CLI Interface**: Rich-formatted interactive chat with session management

## Configuration

Set `OPENAI_API_KEY` environment variable for LLM-powered responses. Without it, the chatbot operates in context-retrieval mode using the indexed YANG models.

Edit `config.yaml` to customize behavior:

- `yang_models.directory`: Path to YANG model files
- `llm.provider`: LLM provider (default: openai)
- `rag.top_k`: Number of chunks to retrieve per query
- `validation.confidence_threshold`: Escalation threshold (default: 0.90)

## CLI Commands

| Command | Description |
|---------|-------------|
| `help` | Show available commands |
| `stats` | Display YANG model statistics |
| `modules` | List loaded YANG modules |
| `validate <text>` | Validate YANG content |
| `history` | Show conversation history |
| `clear` | Clear conversation history |
| `quit` | Exit the chatbot |

## Running Tests

```bash
pip install pytest
pytest tests/ -v
```
