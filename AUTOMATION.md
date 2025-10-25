# Dagster Automation Guide

This document explains how to use the schedules, sensors, and jobs in the Clinical Drug Discovery pipeline.

## 📅 Schedules (Time-Based Automation)

Schedules automatically run jobs at specified times using cron syntax.

### Available Schedules

1. **`daily_clinical_extraction`**
   - **Runs**: Every day at 8:00 AM
   - **Cron**: `0 8 * * *`
   - **Purpose**: Extract drug-disease pairs from clinical notes
   - **Assets**: Clinical extraction group

2. **`weekly_data_refresh`**
   - **Runs**: Every Sunday at 2:00 AM
   - **Cron**: `0 2 * * 0`
   - **Purpose**: Refresh PrimeKG knowledge graph data
   - **Assets**: Data loading group

3. **`monthly_pipeline_run`**
   - **Runs**: 1st of every month at midnight
   - **Cron**: `0 0 1 * *`
   - **Purpose**: Complete pipeline execution
   - **Assets**: All assets

### Cron Schedule Format

```
┌───────────── minute (0 - 59)
│ ┌───────────── hour (0 - 23)
│ │ ┌───────────── day of month (1 - 31)
│ │ │ ┌───────────── month (1 - 12)
│ │ │ │ ┌───────────── day of week (0 - 6) (Sunday=0)
│ │ │ │ │
* * * * *
```

**Examples:**
- `0 8 * * *` - Daily at 8 AM
- `0 */6 * * *` - Every 6 hours
- `0 0 * * 0` - Every Sunday at midnight
- `0 0 1 * *` - First day of month at midnight

### How to Use Schedules

#### Option 1: Enable in Dagster UI

1. Restart Dagster to see new schedules:
   ```bash
   # Stop current dagster dev (Ctrl+C)
   dagster dev
   ```

2. Go to **Automation** tab in Dagster UI

3. Find your schedule and click **"Turn On"**

4. The schedule will now run automatically!

#### Option 2: Run Daemon (Required for Production)

For schedules to work, you need the Dagster daemon running:

```bash
# Terminal 1: Run the webserver
dagster dev

# Terminal 2: Run the daemon (in a separate terminal)
dagster-daemon run
```

**Note**: In development mode (`dagster dev`), the daemon is included. In production, you run them separately.

---

## 🔔 Sensors (Event-Based Automation)

Sensors trigger jobs based on events or conditions rather than time.

### Available Sensors

1. **`new_clinical_data_sensor`**
   - **Checks**: Every 5 minutes
   - **Trigger**: File `data/01_raw/clinical_notes_trigger.txt` exists
   - **Purpose**: Process new clinical notes when they arrive
   - **Assets**: Clinical extraction group

2. **`primekg_update_sensor`**
   - **Checks**: Every hour
   - **Trigger**: File `data/01_raw/primekg/update_trigger.flag` exists
   - **Purpose**: Refresh PrimeKG when update is available
   - **Assets**: Data loading group

### How to Use Sensors

#### Enable Sensor in UI

1. Go to **Automation** tab
2. Find your sensor
3. Click **"Turn On"**

#### Trigger a Sensor Manually

**Example: Trigger clinical data extraction**
```bash
# Create trigger file
echo "New data batch 2024-01-15" > data/01_raw/clinical_notes_trigger.txt

# Sensor will detect it within 5 minutes and trigger the pipeline
# The trigger file is automatically deleted after processing
```

**Example: Trigger PrimeKG refresh**
```bash
# Create trigger file
touch data/01_raw/primekg/update_trigger.flag

# Sensor will detect it within 1 hour and trigger data loading
```

---

## 🎯 Jobs (Reusable Asset Selections)

Jobs are named groups of assets that can be run together.

### Available Jobs

1. **`clinical_extraction_job`**
   - Runs all clinical extraction assets
   - Can be triggered manually or by schedules/sensors

2. **`weekly_data_refresh_job`**
   - Runs all data loading assets
   - Refreshes PrimeKG knowledge graph

3. **`complete_pipeline_job`**
   - Runs ALL assets in the pipeline
   - Full end-to-end execution

### How to Run Jobs

#### From UI

1. Go to **Overview** → **Jobs**
2. Click on a job (e.g., `clinical_extraction_job`)
3. Click **"Launch run"**

#### From CLI

```bash
# Run a specific job
dagster job execute -m dagster_definitions -j clinical_extraction_job

# Run with config (if needed)
dagster job execute -m dagster_definitions -j clinical_extraction_job -c config.yaml
```

---

## 🚀 Quick Start Guide

### 1. View Schedules and Sensors

```bash
# Start Dagster
dagster dev

# Open browser to: http://localhost:3000
# Click on "Automation" tab
```

### 2. Enable a Schedule

1. Navigate to **Automation** → **Schedules**
2. Click on `daily_clinical_extraction`
3. Click **"Turn On"**
4. ✅ It will now run daily at 8 AM!

### 3. Test a Sensor

```bash
# Trigger the clinical data sensor
echo "test run" > data/01_raw/clinical_notes_trigger.txt

# Check sensor status in UI
# Go to Automation → Sensors → new_clinical_data_sensor
# You'll see it trigger within 5 minutes
```

### 4. Run a Job Manually

1. Go to **Overview** → **Jobs**
2. Select `clinical_extraction_job`
3. Click **"Launch run"**

---

## 🔧 Development vs Production

### Development (Current Setup)

```bash
dagster dev
```

- Daemon is included automatically
- Schedules/sensors work out of the box
- Local SQLite database
- Single process

### Production

```bash
# Terminal 1: Webserver
dagster-webserver -h 0.0.0.0 -p 3000

# Terminal 2: Daemon (required for schedules/sensors)
dagster-daemon run

# Terminal 3: Monitor logs
tail -f /path/to/dagster/logs/*.log
```

- Separate daemon process
- PostgreSQL database (already configured in `.dagster/dagster.yaml`)
- Can scale horizontally
- Better monitoring

---

## 📊 Monitoring Automation

### Check Schedule Status

```bash
# List all schedules
dagster schedule list -m dagster_definitions

# Check specific schedule
dagster schedule debug -m dagster_definitions daily_clinical_extraction
```

### Check Sensor Status

```bash
# List all sensors
dagster sensor list -m dagster_definitions

# Check specific sensor
dagster sensor debug -m dagster_definitions new_clinical_data_sensor
```

---

## 💡 Tips and Best practices

1. **Start Small**: Enable one schedule at a time to test

2. **Monitor First Runs**: Check logs for the first few automated runs

3. **Use Sensors for Events**: Prefer sensors over frequent schedules when waiting for external events

4. **Test Locally**: Always test schedules/sensors in dev before production

5. **Cleanup Trigger Files**: Sensors automatically delete trigger files after processing

6. **Adjust Timing**: Modify cron schedules based on your data update frequency

7. **Use Job Tags**: Add tags to track why a run was triggered

---

## 🆘 Troubleshooting

### Schedule Not Running

**Problem**: Schedule shows as "On" but doesn't execute

**Solution**:
```bash
# Check if daemon is running
ps aux | grep dagster-daemon

# Restart daemon
pkill -f dagster-daemon
dagster-daemon run
```

### Sensor Not Detecting Files

**Problem**: Trigger file created but sensor doesn't fire

**Solution**:
- Check sensor is "On" in UI
- Verify file path is correct
- Wait for minimum interval (5 min for clinical, 1 hour for PrimeKG)
- Check sensor logs in Automation tab

### View Logs

```bash
# Daemon logs
tail -f $DAGSTER_HOME/logs/dagster-daemon.log

# Run logs (in Dagster UI)
Go to Runs → Select run → View logs
```

---

## 📚 Further Reading

- [Dagster Schedules Documentation](https://docs.dagster.io/concepts/automation/schedules)
- [Dagster Sensors Documentation](https://docs.dagster.io/concepts/automation/sensors)
- [Dagster Jobs Documentation](https://docs.dagster.io/concepts/ops-jobs-graphs/jobs)
- [Cron Expression Generator](https://crontab.guru/)
