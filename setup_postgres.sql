-- Setup script for Dagster PostgreSQL database
-- Run this with: psql -U postgres -f setup_postgres.sql

-- Create user for Dagster
CREATE USER dagster_user WITH PASSWORD 'dagster_password_123';

-- Create database
CREATE DATABASE clinical_drug_discovery_db OWNER dagster_user;

-- Connect to the database
\c clinical_drug_discovery_db

-- Create schema
CREATE SCHEMA IF NOT EXISTS dagster AUTHORIZATION dagster_user;

-- Grant privileges
GRANT ALL PRIVILEGES ON SCHEMA dagster TO dagster_user;
GRANT ALL PRIVILEGES ON DATABASE clinical_drug_discovery_db TO dagster_user;

-- Set default search path for user
ALTER USER dagster_user SET search_path TO dagster, public;

-- Verify setup
\du dagster_user
\l clinical_drug_discovery_db
\dn+ dagster

-- Success message
SELECT 'Dagster PostgreSQL setup complete!' AS status;
