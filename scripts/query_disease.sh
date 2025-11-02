#!/bin/bash
# Query drug predictions for a specific disease
# Usage: ./scripts/query_disease.sh "Castleman"

if [ -z "$1" ]; then
    echo "Usage: $0 <disease_name>"
    echo ""
    echo "Examples:"
    echo "  $0 Castleman"
    echo "  $0 Alzheimer"
    exit 1
fi

DISEASE="$1"
CSV="data/07_model_output/offlabel/10_all_predictions.csv"

if [ ! -f "$CSV" ]; then
    echo "Error: Predictions file not found at $CSV"
    echo ""
    echo "Please materialize the 'offlabel_novel_predictions' asset in Dagster first."
    exit 1
fi

echo "========================================================================"
echo "Top 10 Drug Candidates for: $DISEASE"
echo "========================================================================"

# Search for disease (case-insensitive), sort by score (column 6), take top 10
grep -i "$DISEASE" "$CSV" | \
  sort -t',' -k6 -rn | \
  head -10 | \
  awk -F',' '{printf "%5s. %-50s | Score: %s\n", $1, $3, $6}'

# Count total matches
TOTAL=$(grep -i "$DISEASE" "$CSV" | wc -l | tr -d ' ')
echo ""
echo "------------------------------------------------------------------------"
echo "Total candidate drugs found: $TOTAL"
