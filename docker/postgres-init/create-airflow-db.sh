#!/bin/bash
# Creates an additional 'airflow_db' database for Airflow
# so it doesn't conflict with MLflow's schema in 'mlflow_db'.
# This script runs automatically on first PostgreSQL startup.

set -e

psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-EOSQL
    CREATE DATABASE airflow_db;
    GRANT ALL PRIVILEGES ON DATABASE airflow_db TO $POSTGRES_USER;
EOSQL

echo "✅ Created airflow_db database"
