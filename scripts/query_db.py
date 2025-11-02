#!/usr/bin/env python
"""
Query the offlabel_predictions PostgreSQL database.

Usage:
    python scripts/query_db.py "Castleman"
    python scripts/query_db.py "Alzheimer" --top 20
    python scripts/query_db.py "cancer" --min-score 0.8
"""

import argparse
import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

# Load environment variables
load_dotenv()


def get_db_engine():
    """Create database engine from environment variables."""
    postgres_user = os.getenv("DAGSTER_POSTGRES_USER", "dagster_user")
    postgres_password = os.getenv("DAGSTER_POSTGRES_PASSWORD", "dagster_password_123")
    postgres_host = os.getenv("DAGSTER_POSTGRES_HOST", "localhost")
    postgres_port = os.getenv("DAGSTER_POSTGRES_PORT", "5432")
    postgres_db = os.getenv("DAGSTER_POSTGRES_DB", "clinical_drug_discovery_db")

    connection_string = f"postgresql://{postgres_user}:{postgres_password}@{postgres_host}:{postgres_port}/{postgres_db}"
    return create_engine(connection_string)


def main():
    parser = argparse.ArgumentParser(description='Query drug predictions from PostgreSQL database')
    parser.add_argument('disease', help='Disease name to search for')
    parser.add_argument('--top', type=int, default=10, help='Number of top drugs to show (default: 10)')
    parser.add_argument('--min-score', type=float, default=0.0, help='Minimum prediction score (default: 0.0)')

    args = parser.parse_args()

    engine = get_db_engine()

    # Query for disease predictions
    query = text("""
        SELECT rank, drug_name, prediction_score
        FROM offlabel_predictions
        WHERE disease_name ILIKE :disease_pattern
          AND prediction_score >= :min_score
        ORDER BY prediction_score DESC
        LIMIT :limit
    """)

    print(f"Querying PostgreSQL database for: {args.disease}")
    print("=" * 80)

    with engine.connect() as conn:
        result = conn.execute(
            query,
            {
                "disease_pattern": f"%{args.disease}%",
                "min_score": args.min_score,
                "limit": args.top
            }
        )

        rows = result.fetchall()

        if not rows:
            print(f"\nNo predictions found for '{args.disease}'")
            return

        # Get disease name from first row
        disease_query = text("""
            SELECT DISTINCT disease_name
            FROM offlabel_predictions
            WHERE disease_name ILIKE :disease_pattern
            LIMIT 1
        """)
        disease_result = conn.execute(disease_query, {"disease_pattern": f"%{args.disease}%"})
        disease_name = disease_result.scalar()

        print(f"\nTop {len(rows)} Drug Candidates for: {disease_name}")
        print("=" * 80)

        for rank, drug_name, score in rows:
            print(f"{rank:5d}. {drug_name[:50]:50s} | Score: {score:.4f}")

        # Get statistics
        stats_query = text("""
            SELECT
                COUNT(*) as total_candidates,
                COUNT(*) FILTER (WHERE prediction_score > 0.9) as high_confidence,
                COUNT(*) FILTER (WHERE prediction_score > 0.7 AND prediction_score <= 0.9) as moderate_confidence,
                AVG(prediction_score) as mean_score,
                PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY prediction_score) as median_score
            FROM offlabel_predictions
            WHERE disease_name = :disease_name
        """)

        stats_result = conn.execute(stats_query, {"disease_name": disease_name})
        stats = stats_result.fetchone()

        print("\n" + "-" * 80)
        print(f"Statistics for {disease_name}:")
        print(f"  Total candidate drugs: {stats[0]:,}")
        print(f"  High confidence (>0.9): {stats[1]:,}")
        print(f"  Moderate confidence (0.7-0.9): {stats[2]:,}")
        print(f"  Mean score: {stats[3]:.4f}")
        print(f"  Median score: {stats[4]:.4f}")


if __name__ == "__main__":
    main()
