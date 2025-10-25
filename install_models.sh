#!/bin/bash
# Install script for models that cannot be added via uv sync
# Run this after: uv sync

set -e

echo "Installing spaCy and scispaCy models..."

# Try to install scispaCy BC5CDR model (for biomedical NER)
echo "Attempting to install en_ner_bc5cdr_md..."
uv pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_ner_bc5cdr_md-0.5.4.tar.gz || \
echo "Warning: Could not install en_ner_bc5cdr_md. You may need to install it manually."

echo "Model installation complete!"
echo ""
echo "To verify installation, run:"
echo "  uv run -- python -c 'import spacy; nlp = spacy.load(\"en_ner_bc5cdr_md\"); print(\"Model loaded successfully!\")'"
