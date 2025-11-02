#!/usr/bin/env python
"""
Setup PostgreSQL database for clinical drug discovery predictions.

This script creates the database, user, and grants necessary permissions.

Usage:
    python scripts/setup_postgres.py
    python scripts/setup_postgres.py --admin-user postgres --admin-password mypassword
"""

import argparse
import os
import sys
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.exc import OperationalError, ProgrammingError

# Load environment variables
load_dotenv()


def get_env_vars():
    """Get database configuration from environment variables."""
    return {
        'user': os.getenv('DAGSTER_POSTGRES_USER', 'dagster_user'),
        'password': os.getenv('DAGSTER_POSTGRES_PASSWORD', 'dagster_password_123'),
        'host': os.getenv('DAGSTER_POSTGRES_HOST', 'localhost'),
        'port': os.getenv('DAGSTER_POSTGRES_PORT', '5432'),
        'database': os.getenv('DAGSTER_POSTGRES_DB', 'clinical_drug_discovery_db'),
        'schema': os.getenv('DAGSTER_POSTGRES_SCHEMA', 'public'),
    }


def test_postgres_connection(host, port):
    """Test if PostgreSQL is accessible."""
    try:
        # Try to connect to default 'postgres' database
        engine = create_engine(f"postgresql://postgres@{host}:{port}/postgres",
                              connect_args={'connect_timeout': 5})
        with engine.connect():
            return True
    except Exception:
        return False


def create_database_and_user(admin_user, admin_password, config):
    """Create database and user with proper permissions."""
    host = config['host']
    port = config['port']
    database = config['database']
    user = config['user']
    password = config['password']
    schema = config['schema']

    print("=" * 60)
    print("PostgreSQL Database Setup")
    print("=" * 60)
    print(f"Host: {host}:{port}")
    print(f"Database: {database}")
    print(f"User: {user}")
    print(f"Schema: {schema}")
    print("=" * 60)
    print()

    # Check if PostgreSQL is running
    print("Checking if PostgreSQL is running...")
    if not test_postgres_connection(host, port):
        print(f"✗ Error: Cannot connect to PostgreSQL on {host}:{port}")
        print()
        print("To start PostgreSQL:")
        print("  - macOS (Homebrew): brew services start postgresql@14")
        print("  - macOS (Postgres.app): Open Postgres.app")
        print("  - Linux: sudo systemctl start postgresql")
        print("  - Docker: docker run -d -p 5432:5432 -e POSTGRES_PASSWORD=postgres postgres:14")
        return False
    print("✓ PostgreSQL is running")
    print()

    # Connect as admin to postgres database
    if admin_password:
        admin_conn_str = f"postgresql://{admin_user}:{admin_password}@{host}:{port}/postgres"
    else:
        admin_conn_str = f"postgresql://{admin_user}@{host}:{port}/postgres"

    try:
        admin_engine = create_engine(admin_conn_str, isolation_level="AUTOCOMMIT")
    except Exception as e:
        print(f"✗ Error: Cannot connect as admin user '{admin_user}'")
        print(f"   {e}")
        print()
        print("Try running with admin credentials:")
        print(f"  python {sys.argv[0]} --admin-user postgres --admin-password YOUR_PASSWORD")
        return False

    # Create user if not exists
    print(f"Creating user '{user}' if not exists...")
    try:
        with admin_engine.connect() as conn:
            # Check if user exists
            result = conn.execute(text(f"SELECT 1 FROM pg_user WHERE usename = '{user}'"))
            if not result.fetchone():
                conn.execute(text(f"CREATE USER {user} WITH PASSWORD '{password}'"))
                print(f"✓ Created user '{user}'")
            else:
                print(f"✓ User '{user}' already exists")
    except Exception as e:
        print(f"✗ Error creating user: {e}")
        return False
    print()

    # Create database if not exists
    print(f"Creating database '{database}' if not exists...")
    try:
        with admin_engine.connect() as conn:
            # Check if database exists
            result = conn.execute(text(f"SELECT 1 FROM pg_database WHERE datname = '{database}'"))
            if not result.fetchone():
                conn.execute(text(f"CREATE DATABASE {database} OWNER {user}"))
                print(f"✓ Created database '{database}'")
            else:
                print(f"✓ Database '{database}' already exists")
    except Exception as e:
        print(f"✗ Error creating database: {e}")
        return False
    print()

    # Create schema and grant privileges
    print(f"Creating schema '{schema}' if needed and granting privileges...")
    try:
        # Connect to the target database as admin
        if admin_password:
            db_admin_conn_str = f"postgresql://{admin_user}:{admin_password}@{host}:{port}/{database}"
        else:
            db_admin_conn_str = f"postgresql://{admin_user}@{host}:{port}/{database}"

        db_admin_engine = create_engine(db_admin_conn_str, isolation_level="AUTOCOMMIT")

        with db_admin_engine.connect() as conn:
            # Create schema if it doesn't exist (only for non-public schemas)
            if schema != 'public':
                conn.execute(text(f"CREATE SCHEMA IF NOT EXISTS {schema}"))
                print(f"✓ Schema '{schema}' created or already exists")

            # Grant privileges
            conn.execute(text(f"GRANT ALL PRIVILEGES ON DATABASE {database} TO {user}"))
            conn.execute(text(f"GRANT ALL PRIVILEGES ON SCHEMA {schema} TO {user}"))
            conn.execute(text(f"ALTER DEFAULT PRIVILEGES IN SCHEMA {schema} GRANT ALL ON TABLES TO {user}"))

            # Grant usage on schema to allow user to access it
            conn.execute(text(f"GRANT USAGE ON SCHEMA {schema} TO {user}"))
            conn.execute(text(f"GRANT CREATE ON SCHEMA {schema} TO {user}"))

            print(f"✓ Privileges granted to '{user}' on schema '{schema}'")
    except Exception as e:
        print(f"✗ Error creating schema or granting privileges: {e}")
        return False
    print()

    # Test connection as new user
    print(f"Testing connection as '{user}'...")
    try:
        user_conn_str = f"postgresql://{user}:{password}@{host}:{port}/{database}"
        user_engine = create_engine(user_conn_str)
        with user_engine.connect() as conn:
            conn.execute(text("SELECT 1"))
            print("✓ Connection successful")
    except Exception as e:
        print(f"✗ Connection failed: {e}")
        return False
    print()

    print("=" * 60)
    print("✓ PostgreSQL setup complete!")
    print("=" * 60)
    print()
    print("Connection details:")
    print(f"  Database: {database}")
    print(f"  User: {user}")
    print(f"  Host: {host}")
    print(f"  Port: {port}")
    print(f"  Schema: {schema}")
    print()
    print("Connection string:")
    print(f"  postgresql://{user}:****@{host}:{port}/{database}")
    print()
    print("You can now run the Dagster asset 'offlabel_predictions_db' to load predictions.")

    return True


def main():
    parser = argparse.ArgumentParser(description='Setup PostgreSQL database for clinical drug discovery')
    parser.add_argument('--admin-user', default='postgres', help='PostgreSQL admin username (default: postgres)')
    parser.add_argument('--admin-password', default=None, help='PostgreSQL admin password (default: none)')

    args = parser.parse_args()

    config = get_env_vars()

    success = create_database_and_user(args.admin_user, args.admin_password, config)

    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
