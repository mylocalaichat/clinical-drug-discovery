# PostgreSQL Database Setup

This guide explains how to set up the PostgreSQL database for storing and querying off-label drug discovery predictions.

## Overview

The `offlabel_predictions_db` Dagster asset loads 46.5M drug-disease predictions into PostgreSQL for fast querying. The database stores:

- **Database:** `clinical_drug_discovery_db`
- **Schema:** `public`
- **Table:** `offlabel_predictions`
- **Rows:** ~46.5 million predictions
- **Indexes:** 4 indexes for fast queries

## Prerequisites

1. PostgreSQL installed and running
2. Python environment with dependencies installed

### Installing PostgreSQL

**macOS (Homebrew):**
```bash
brew install postgresql@14
brew services start postgresql@14
```

**macOS (Postgres.app):**
- Download from https://postgresapp.com/
- Open the app to start PostgreSQL

**Linux (Ubuntu/Debian):**
```bash
sudo apt-get install postgresql postgresql-contrib
sudo systemctl start postgresql
```

**Docker:**
```bash
docker run -d \
  --name postgres \
  -p 5432:5432 \
  -e POSTGRES_PASSWORD=postgres \
  postgres:14
```

## Setup Steps

### Option 1: Automated Setup (Recommended)

Run the Python setup script:

```bash
python scripts/setup_postgres.py
```

If you need to specify an admin password:

```bash
python scripts/setup_postgres.py --admin-user postgres --admin-password YOUR_PASSWORD
```

### Option 2: Bash Script (Linux/macOS)

```bash
./scripts/setup_postgres.sh
```

This requires the `postgres` superuser to be accessible without password (peer authentication).

### Option 3: Manual Setup

Connect to PostgreSQL as superuser:

```bash
psql -U postgres
```

Create user and database:

```sql
-- Create user
CREATE USER dagster_user WITH PASSWORD 'dagster_password_123';

-- Create database
CREATE DATABASE clinical_drug_discovery_db OWNER dagster_user;

-- Connect to the database
\c clinical_drug_discovery_db

-- Grant privileges
GRANT ALL PRIVILEGES ON DATABASE clinical_drug_discovery_db TO dagster_user;
GRANT ALL PRIVILEGES ON SCHEMA public TO dagster_user;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON TABLES TO dagster_user;
```

## Configuration

Database credentials are configured in `.env`:

```bash
# PostgreSQL Configuration
DAGSTER_POSTGRES_USER=dagster_user
DAGSTER_POSTGRES_PASSWORD=dagster_password_123
DAGSTER_POSTGRES_HOST=localhost
DAGSTER_POSTGRES_PORT=5432
DAGSTER_POSTGRES_DB=clinical_drug_discovery_db
DAGSTER_POSTGRES_SCHEMA=public
```

**Security Note:** For production, change the default password and use environment-specific credentials.

## Loading Predictions

After setup, load predictions via Dagster:

1. Open Dagster UI: http://localhost:3000
2. Navigate to Assets
3. Materialize: `offlabel_predictions_db`

This will:
- Load all 46.5M predictions from CSV
- Create the `offlabel_predictions` table
- Create 4 indexes for fast queries
- Takes ~5-10 minutes depending on hardware

## Querying Predictions

### Using the Python Script

```bash
# Top 10 drugs for Castleman disease
python scripts/query_db.py "Castleman"

# Top 20 drugs for Alzheimer's
python scripts/query_db.py "Alzheimer" --top 20

# Filter by minimum score
python scripts/query_db.py "cancer" --min-score 0.8
```

### Direct SQL Queries

```sql
-- Top 10 drugs for a disease
SELECT rank, drug_name, prediction_score
FROM public.offlabel_predictions
WHERE disease_name ILIKE '%Castleman%'
ORDER BY prediction_score DESC
LIMIT 10;

-- All high-confidence predictions
SELECT drug_name, prediction_score
FROM public.offlabel_predictions
WHERE disease_name = 'Castleman disease'
  AND prediction_score > 0.9
ORDER BY prediction_score DESC;

-- Find diseases for a specific drug
SELECT disease_name, prediction_score
FROM public.offlabel_predictions
WHERE drug_name ILIKE '%Triamterene%'
ORDER BY prediction_score DESC
LIMIT 20;

-- Summary statistics by disease
SELECT disease_name,
       COUNT(*) as candidate_count,
       MAX(prediction_score) as top_score,
       AVG(prediction_score) as avg_score
FROM public.offlabel_predictions
GROUP BY disease_name
ORDER BY top_score DESC
LIMIT 10;
```

## Table Schema

```sql
CREATE TABLE public.offlabel_predictions (
    rank INTEGER,
    drug_id TEXT,
    drug_name TEXT,
    disease_id TEXT,
    disease_name TEXT,
    prediction_score DOUBLE PRECISION
);

-- Indexes for fast queries
CREATE INDEX idx_disease_name ON public.offlabel_predictions (disease_name);
CREATE INDEX idx_drug_name ON public.offlabel_predictions (drug_name);
CREATE INDEX idx_prediction_score ON public.offlabel_predictions (prediction_score DESC);
CREATE INDEX idx_disease_score ON public.offlabel_predictions (disease_name, prediction_score DESC);
```

## Troubleshooting

### PostgreSQL not running

```bash
# Check if PostgreSQL is running
pg_isready -h localhost -p 5432

# Start PostgreSQL (macOS Homebrew)
brew services start postgresql@14

# Start PostgreSQL (Linux)
sudo systemctl start postgresql
```

### Connection refused

- Verify PostgreSQL is running on the correct port: `netstat -an | grep 5432`
- Check `pg_hba.conf` for authentication settings
- Ensure firewall allows connections to port 5432

### Permission denied

- Run setup script as PostgreSQL superuser
- Grant proper privileges: `GRANT ALL PRIVILEGES ON DATABASE clinical_drug_discovery_db TO dagster_user;`

### Table already exists

The asset automatically drops and recreates the table on each materialization. To preserve data, query it first or use a different approach.

## Performance

- **Table size:** ~5-6 GB
- **Load time:** 5-10 minutes
- **Query performance:** <100ms for filtered queries with indexes
- **Index size:** ~2-3 GB total

## Backup and Restore

```bash
# Backup
pg_dump -U dagster_user -h localhost clinical_drug_discovery_db > backup.sql

# Restore
psql -U dagster_user -h localhost clinical_drug_discovery_db < backup.sql
```
