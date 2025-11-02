# Progress Tracking in Database Loading

The `offlabel_predictions_db` asset now includes comprehensive progress tracking for all major operations.

## Progress Bars and Logging

### 1. Database Connection
```
Connecting to PostgreSQL database 'clinical_drug_discovery_db' on localhost:5432
Target schema: public
Target table: offlabel_predictions
✓ Database connection successful
```

### 2. CSV Loading
```
Loading predictions from data/07_model_output/offlabel/10_all_predictions.csv...
Counting rows in CSV file...
Reading 46,557,830 predictions from CSV...
✓ Loaded 46,557,830 predictions into memory
```

### 3. Table Preparation
```
Loading data into PostgreSQL table 'offlabel_predictions'...
Dropped existing table 'public.offlabel_predictions' if it existed
Writing 46,557,830 rows to public.offlabel_predictions...
Inserting data in 466 chunks of 100,000 rows each...
Using PostgreSQL COPY for fast batch loading...
```

### 4. Data Insertion with Progress Bar (Using COPY)
```
Loading to PostgreSQL: 100%|████████████████| 466/466 [02:15<00:00, 3.4chunk/s]
✓ Data loaded successfully
```

**Expected time:** 2-5 minutes (significantly faster with COPY)
**Note:** Uses PostgreSQL's native COPY command for 5-10x faster bulk loading

### 5. Index Creation with Progress Bar
```
Creating indexes for fast queries (this may take a few minutes)...
Creating index idx_disease_name...
Creating index idx_drug_name...
Creating index idx_prediction_score...
Creating index idx_disease_score...
Creating indexes: 100%|████████████████| 4/4 [02:15<00:00, 33.8s/index]
✓ All indexes created successfully
```

**Expected time:** 2-5 minutes per index

### 6. Verification
```
✓ Successfully loaded 46,557,830 predictions to clinical_drug_discovery_db.public.offlabel_predictions
```

## Total Estimated Time

- **CSV Loading:** 30 seconds - 1 minute
- **Data Insertion (COPY):** 2-5 minutes (was 5-10 min with INSERT)
- **Index Creation:** 8-20 minutes (4 indexes × 2-5 min each)
- **Total:** ~10-25 minutes (was 15-30 min)

**Performance Improvement:** Using PostgreSQL COPY reduces data loading time by 50-80%

Performance varies based on:
- CPU speed
- Disk I/O (SSD vs HDD)
- PostgreSQL configuration
- Available RAM
- Network latency (if database is remote)

## Monitoring Progress in Dagster

1. Open the Dagster UI: http://localhost:3000
2. Navigate to the run page for `offlabel_predictions_db`
3. Watch the "Raw Compute Logs" tab for progress bars
4. The progress bars update in real-time showing:
   - Current chunk being inserted
   - Insertion rate (chunks/second)
   - Estimated time remaining
   - Current index being created

## Console Output Example

```bash
2025-01-15 10:30:15 - INFO - Connecting to PostgreSQL database...
2025-01-15 10:30:15 - INFO - ✓ Database connection successful
2025-01-15 10:30:15 - INFO - Loading predictions from CSV...
2025-01-15 10:30:45 - INFO - ✓ Loaded 46,557,830 predictions
2025-01-15 10:30:45 - INFO - Inserting data in 466 chunks...
2025-01-15 10:30:45 - INFO - Using PostgreSQL COPY for fast batch loading...

Loading to PostgreSQL:  52%|████████████▎           | 242/466 [01:14<01:09, 3.2chunk/s]

2025-01-15 10:34:23 - INFO - ✓ Data loaded successfully
2025-01-15 10:34:23 - INFO - Creating indexes...

Creating indexes:  25%|█████▌             | 1/4 [02:34<07:42, 154s/index]

2025-01-15 10:44:47 - INFO - ✓ All indexes created successfully
2025-01-15 10:44:47 - INFO - ✓ Successfully loaded 46,557,830 predictions
```

## Optimization Tips

### Speed Up Loading

1. **Already optimized with COPY** ✓
   - Using PostgreSQL's native COPY command (5-10x faster than INSERT)
   - Chunk size: 100,000 rows (optimized for COPY)

2. **Defer index creation** ✓
   - Already implemented - indexes created after data load

3. **Use faster storage**:
   - SSD significantly faster than HDD
   - NVMe SSD provides best performance

4. **Tune PostgreSQL** (in postgresql.conf):
   ```ini
   shared_buffers = 4GB
   maintenance_work_mem = 2GB
   work_mem = 256MB
   max_wal_size = 4GB
   checkpoint_timeout = 30min
   ```

5. **Further optimizations** (advanced):
   - Increase chunk size to 500,000 (requires more memory)
   - Disable WAL temporarily: `ALTER TABLE offlabel_predictions SET UNLOGGED;`
   - Use parallel index creation (PostgreSQL 11+)

### Monitor System Resources

```bash
# Monitor disk I/O
iostat -x 5

# Monitor PostgreSQL activity
psql -U dagster_user -d clinical_drug_discovery_db -c "
  SELECT
    pid,
    query,
    state,
    query_start
  FROM pg_stat_activity
  WHERE datname = 'clinical_drug_discovery_db';
"

# Monitor table size during load
psql -U dagster_user -d clinical_drug_discovery_db -c "
  SELECT
    pg_size_pretty(pg_total_relation_size('public.offlabel_predictions')) as size;
"
```

## Troubleshooting Slow Loading

### If loading is very slow:

1. **Check disk space:** Ensure at least 10GB free
2. **Check PostgreSQL logs:** Look for errors or warnings
3. **Increase work_mem:** For faster index creation
4. **Disable autovacuum temporarily:** Prevent vacuuming during load
5. **Use COPY instead of INSERT:** For even faster bulk loading (advanced)

### If progress bar doesn't show:

- Progress bars use `tqdm` which requires a TTY
- In Dagster logs, you'll see individual chunk numbers instead
- The operation is still progressing even without visual bar
