# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "6"
# ///
# MAGIC %md
# MAGIC # 01 - Ingest PHS weekly A&E data
# MAGIC
# MAGIC Week 1, session 2 of the ED demand forecaster.
# MAGIC
# MAGIC - Reads the three Public Health Scotland CSVs from the `raw_files` volume
# MAGIC - Saves them unchanged as raw Delta tables, with source file name and load time
# MAGIC - Finds the two Glasgow hospitals' codes by name (no hard-coded codes)
# MAGIC - Builds a typed, filtered table `clean_ae_weekly`
# MAGIC
# MAGIC Contains public sector information licensed under the Open Government Licence v3.0.

# COMMAND ----------

# Cell 1 - Configuration
from pyspark.sql import functions as F

CATALOG = "workspace"
SCHEMA = "ed_forecaster"
VOLUME_PATH = f"/Volumes/{CATALOG}/{SCHEMA}/raw_files"

# File names as uploaded to the volume
FILES = {
    "ae_weekly": "weekly_ae_activity_20261115.csv",           # Weekly A&E Activity and Waiting Times
    "ae_sites": "1a4e3f48-3d9b-4769-80e9-3ef6d27852fe.csv",   # A&E Sites lookup
    "hospitals": "c698f450-eeed-41a0-88f7-c1e40a568acc.csv",  # Current NHS Hospitals in Scotland
}

# Hospitals to model, matched by name (case-insensitive) in the hospital codes file
TARGET_HOSPITALS = ["Queen Elizabeth University", "Glasgow Royal Infirmary"]

def table(name: str) -> str:
    return f"{CATALOG}.{SCHEMA}.{name}"

# COMMAND ----------

# Cell 2 - Read each CSV as text columns and add lineage columns
def read_raw(file_name: str):
    return (
        spark.read
        .option("header", True)
        .option("inferSchema", False)          # keep raw data exactly as published
        .csv(f"{VOLUME_PATH}/{file_name}")
        .withColumn("_source_file", F.col("_metadata.file_name"))
        .withColumn("_ingested_at", F.current_timestamp())
    )

raw = {key: read_raw(name) for key, name in FILES.items()}

for key, df in raw.items():
    print(f"--- {key}: {df.count()} rows")
    print(df.columns)

# COMMAND ----------

# Cell 3 - Save the raw tables (overwritten on each load; the source file name records which download)
for key, df in raw.items():
    df.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(table(f"raw_{key}"))
    print("Saved", table(f"raw_{key}"))

# COMMAND ----------

# Cell 4 - Look up the two hospitals' codes by name
hospitals = spark.table(table("raw_hospitals"))

name_filter = F.lit(False)
for name in TARGET_HOSPITALS:
    name_filter = name_filter | F.lower(F.col("HospitalName")).contains(name.lower())

targets = hospitals.filter(name_filter).select("HospitalCode", "HospitalName", "HealthBoard")
display(targets)

target_codes = [r["HospitalCode"] for r in targets.collect()]
print("Target codes:", target_codes)
assert len(target_codes) == len(TARGET_HOSPITALS), (
    "Expected one code per hospital - check the names above and adjust TARGET_HOSPITALS"
)

# COMMAND ----------

# Cell 5 - Check those codes appear as ED sites in the A&E Sites file
sites = spark.table(table("raw_ae_sites"))
print(sites.columns)
display(sites.filter(F.col("TreatmentLocationCode").isin(target_codes)))

# COMMAND ----------

# Cell 6 - Inspect categories before any totals are calculated
weekly = spark.table(table("raw_ae_weekly")).filter(F.col("TreatmentLocation").isin(target_codes))

display(weekly.groupBy("TreatmentLocation", "DepartmentType", "AttendanceCategory").count()
        .orderBy("TreatmentLocation", "DepartmentType", "AttendanceCategory"))

# Look at the categories above. If any value is itself a total (for example "All"),
# exclude it in Cell 8 so attendances are not double counted.

# COMMAND ----------

# Cell 7 - Build the clean table with proper types
clean = (
    weekly
    .withColumn("week_ending", F.to_date(F.col("WeekEndingDate").cast("string"), "yyyyMMdd"))
    .withColumn("attendances", F.col("NumberOfAttendancesEpisode").cast("int"))
    .withColumn("within_4h", F.col("NumberWithin4HoursEpisode").cast("int"))
    .withColumn("over_4h", F.col("NumberOver4HoursEpisode").cast("int"))
    .withColumn("over_8h", F.col("NumberOver8HoursEpisode").cast("int"))
    .withColumn("over_12h", F.col("NumberOver12HoursEpisode").cast("int"))
    .join(
        targets.select(F.col("HospitalCode").alias("TreatmentLocation"),
                       F.col("HospitalName").alias("hospital_name")),
        on="TreatmentLocation", how="left",
    )
    .select(
        "week_ending",
        F.col("TreatmentLocation").alias("location_code"),
        "hospital_name",
        F.col("HBT").alias("health_board"),
        F.col("DepartmentType").alias("department_type"),
        F.col("AttendanceCategory").alias("attendance_category"),
        "attendances", "within_4h", "over_4h", "over_8h", "over_12h",
        "_source_file", "_ingested_at",
    )
)

bad_dates = clean.filter(F.col("week_ending").isNull()).count()
assert bad_dates == 0, f"{bad_dates} rows have an unparsed WeekEndingDate - check its format"

clean.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(table("clean_ae_weekly"))
print("Saved", table("clean_ae_weekly"))

# COMMAND ----------

# Cell 8 - Weekly totals per hospital (the series the model will forecast)
EXCLUDE_CATEGORIES = []   # add any total-type values found in Cell 6, e.g. ["All"]

totals = (
    spark.table(table("clean_ae_weekly"))
    .filter(~F.col("attendance_category").isin(EXCLUDE_CATEGORIES))
    .groupBy("week_ending", "location_code", "hospital_name")
    .agg(F.sum("attendances").alias("total_attendances"))
    .orderBy("location_code", "week_ending")
)

totals.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(table("weekly_totals"))
display(totals)

# COMMAND ----------

# Cell 9 - Date range per hospital (copy into your run notes)
display(
    totals.groupBy("hospital_name")
    .agg(F.min("week_ending").alias("first_week"),
         F.max("week_ending").alias("last_week"),
         F.count("*").alias("weeks"))
)