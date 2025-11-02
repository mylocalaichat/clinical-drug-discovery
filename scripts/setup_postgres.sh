#!/bin/bash
# Setup PostgreSQL database for clinical drug discovery predictions
# This script creates the database, user, and grants necessary permissions

set -e  # Exit on error

# Load environment variables from .env
if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | grep -v '^$' | xargs)
else
    echo "Error: .env file not found"
    exit 1
fi

# Use environment variables or defaults
POSTGRES_USER="${DAGSTER_POSTGRES_USER:-dagster_user}"
POSTGRES_PASSWORD="${DAGSTER_POSTGRES_PASSWORD:-dagster_password_123}"
POSTGRES_HOST="${DAGSTER_POSTGRES_HOST:-localhost}"
POSTGRES_PORT="${DAGSTER_POSTGRES_PORT:-5432}"
POSTGRES_DB="${DAGSTER_POSTGRES_DB:-clinical_drug_discovery_db}"
POSTGRES_SCHEMA="${DAGSTER_POSTGRES_SCHEMA:-public}"

echo "=================================================="
echo "PostgreSQL Database Setup"
echo "=================================================="
echo "Host: $POSTGRES_HOST:$POSTGRES_PORT"
echo "Database: $POSTGRES_DB"
echo "User: $POSTGRES_USER"
echo "Schema: $POSTGRES_SCHEMA"
echo "=================================================="
echo ""

# Check if PostgreSQL is running
echo "Checking if PostgreSQL is running..."
if ! pg_isready -h "$POSTGRES_HOST" -p "$POSTGRES_PORT" > /dev/null 2>&1; then
    echo "Error: PostgreSQL is not running on $POSTGRES_HOST:$POSTGRES_PORT"
    echo ""
    echo "To start PostgreSQL:"
    echo "  - macOS (Homebrew): brew services start postgresql@14"
    echo "  - macOS (Postgres.app): Open Postgres.app"
    echo "  - Linux: sudo systemctl start postgresql"
    echo "  - Docker: docker run -d -p 5432:5432 -e POSTGRES_PASSWORD=postgres postgres:14"
    exit 1
fi
echo "✓ PostgreSQL is running"
echo ""

# Create user if not exists (requires superuser access)
echo "Creating user '$POSTGRES_USER' if not exists..."
psql -h "$POSTGRES_HOST" -p "$POSTGRES_PORT" -U postgres -tc \
    "SELECT 1 FROM pg_user WHERE usename = '$POSTGRES_USER'" | grep -q 1 || \
psql -h "$POSTGRES_HOST" -p "$POSTGRES_PORT" -U postgres -c \
    "CREATE USER $POSTGRES_USER WITH PASSWORD '$POSTGRES_PASSWORD';"
echo "✓ User '$POSTGRES_USER' ready"
echo ""

# Create database if not exists
echo "Creating database '$POSTGRES_DB' if not exists..."
psql -h "$POSTGRES_HOST" -p "$POSTGRES_PORT" -U postgres -tc \
    "SELECT 1 FROM pg_database WHERE datname = '$POSTGRES_DB'" | grep -q 1 || \
psql -h "$POSTGRES_HOST" -p "$POSTGRES_PORT" -U postgres -c \
    "CREATE DATABASE $POSTGRES_DB OWNER $POSTGRES_USER;"
echo "✓ Database '$POSTGRES_DB' ready"
echo ""

# Grant privileges
echo "Granting privileges to user '$POSTGRES_USER'..."
psql -h "$POSTGRES_HOST" -p "$POSTGRES_PORT" -U postgres -d "$POSTGRES_DB" -c \
    "GRANT ALL PRIVILEGES ON DATABASE $POSTGRES_DB TO $POSTGRES_USER;"
psql -h "$POSTGRES_HOST" -p "$POSTGRES_PORT" -U postgres -d "$POSTGRES_DB" -c \
    "GRANT ALL PRIVILEGES ON SCHEMA $POSTGRES_SCHEMA TO $POSTGRES_USER;"
psql -h "$POSTGRES_HOST" -p "$POSTGRES_PORT" -U postgres -d "$POSTGRES_DB" -c \
    "ALTER DEFAULT PRIVILEGES IN SCHEMA $POSTGRES_SCHEMA GRANT ALL ON TABLES TO $POSTGRES_USER;"
echo "✓ Privileges granted"
echo ""

# Test connection
echo "Testing connection as '$POSTGRES_USER'..."
if PGPASSWORD="$POSTGRES_PASSWORD" psql -h "$POSTGRES_HOST" -p "$POSTGRES_PORT" -U "$POSTGRES_USER" -d "$POSTGRES_DB" -c "SELECT 1" > /dev/null 2>&1; then
    echo "✓ Connection successful"
else
    echo "✗ Connection failed"
    exit 1
fi
echo ""

echo "=================================================="
echo "✓ PostgreSQL setup complete!"
echo "=================================================="
echo ""
echo "Connection details:"
echo "  Database: $POSTGRES_DB"
echo "  User: $POSTGRES_USER"
echo "  Host: $POSTGRES_HOST"
echo "  Port: $POSTGRES_PORT"
echo "  Schema: $POSTGRES_SCHEMA"
echo ""
echo "Connection string:"
echo "  postgresql://$POSTGRES_USER:****@$POSTGRES_HOST:$POSTGRES_PORT/$POSTGRES_DB"
echo ""
echo "You can now run the Dagster asset 'offlabel_predictions_db' to load predictions."
