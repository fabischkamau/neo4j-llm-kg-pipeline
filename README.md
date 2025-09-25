# Knowledge Graph Pipeline

![Marie Curie](https://commons.wikimedia.org/wiki/Special:FilePath/Marie-Curie-Nobel-portrait-600.jpg)

A powerful Python pipeline for extracting entities and relationships from PDF documents, enriching them with Google Knowledge Graph, and storing them in a Neo4j graph database.

## Features

- **PDF Document Processing**: Extract text from PDF documents with configurable chunking
- **Entity Extraction**: Identify entities using state-of-the-art LLMs (OpenAI or Anthropic)
- **Knowledge Graph Enrichment**: Disambiguate and enrich entities using Google Knowledge Graph API
- **Relationship Extraction**: Identify meaningful relationships between entities
- **Neo4j Integration**: Store and query the knowledge graph efficiently
- **Configurable Pipeline**: Customize processing parameters and LLM settings
- **Asynchronous Processing**: Efficiently handle large documents

## Prerequisites

- Python 3.13+ (as specified in `pyproject.toml`)
- Neo4j 4.4+ (local or remote instance)
- API keys for:
  - OpenAI or Anthropic (for LLM processing)
  - Google Knowledge Graph API

## Installation (uv)

1. Clone the repository:

   ```bash
   git clone <repository-url>
   cd knowlegdegraph-pipeline
   ```

2. Install dependencies with uv (creates an isolated environment automatically):

   ```bash
   uv sync
   ```

3. Copy the environment template and fill in your values (see Configuration below):

   ```bash
   cp .env.template .env
   ```

## Configuration

1. Copy the template environment file and update with your API keys:

   ```bash
   cp .env.template .env
   ```

2. Edit the `.env` file with your configuration:

   ```
   # Required API Keys
   OPENAI_API_KEY=your_openai_api_key
   GOOGLE_KG_API_KEY=your_google_kg_api_key
   ANTHROPIC_API_KEY=your_anthropic_api_key

   # LLM Provider (openai or anthropic)
   LLM_PROVIDER=openai

   # Neo4j Configuration
   NEO4J_URI=neo4j://localhost:7687
   NEO4J_USER=neo4j
   NEO4J_PASSWORD=your_secure_password

   # Model Configuration
   OPENAI_MODEL=gpt-4o-mini
   ANTHROPIC_MODEL=claude-3-5-sonnet-20241022

   # Text Processing
   CHUNK_SIZE=1500
   CHUNK_OVERLAP=200
   ```

## Usage

The app loads configuration from `.env` using `python-dotenv`. Set `PDF_PATH` in `.env` (or export it inline) to point to the PDF you want to process. The entrypoint is `main.py` and it uses the `PDF_PATH` environment variable; it does not accept `--input/--output` flags.

### Run with uv

```bash
# Option A: define the path in .env
echo "PDF_PATH=./path/to/your.pdf" >> .env
uv run python main.py

# Option B: set PDF_PATH inline for a one-off run
PDF_PATH=./path/to/your.pdf uv run python main.py
```

### Programmatic usage

```python
from main import KnowledgeGraphPipeline, Config
import asyncio

async def run():
    pipeline = KnowledgeGraphPipeline(Config())
    try:
        context = await pipeline.process_pdf_document("./path/to/your.pdf")
        print(len(context.processed_entities), len(context.processed_relationships))
        print(context.overall_summary)
    finally:
        pipeline.close()

asyncio.run(run())
```

## Project Structure

- `main.py`: Main pipeline implementation
- `pyproject.toml`: Project metadata and dependencies (managed by uv)
- `uv.lock`: Resolved dependency lockfile
- `.env.template`: Example environment template
- `.env`: Local configuration (not version controlled)
- `README.md`: This documentation

## Customization

### Entity Extraction

Modify the `extract_entities_from_text` method to customize how entities are extracted from text.

### Relationship Extraction

Adjust the `refine_relationships` method to define how relationships between entities are identified.

### Knowledge Graph Enrichment

Customize the `refine_with_kg` method to change how entities are enriched with knowledge graph data.

## Performance Considerations

- Adjust `CHUNK_SIZE` and `CHUNK_OVERLAP` in the `.env` file based on your document complexity
- For large documents, consider processing in smaller chunks
- Monitor API usage to avoid rate limits

## License

This project is licensed under the **MIT License**.

See the [LICENSE](./LICENSE) file for details.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## Support

For support, please open an issue in the repository.
