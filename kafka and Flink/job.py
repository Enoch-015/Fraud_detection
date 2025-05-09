#!/usr/bin/env python3
"""
job.py: PyFlink streaming job for fraud detection.

1. Reads CDC events from Postgres table 'fraud_dec_test'.
2. Calls ZenML model server for classification.
3. Splits stream into 'fraud' and 'legit' branches.
4. Sinks fraud cases to Postgres and legit cases to Kafka.
"""

import json
import requests
from pyflink.datastream import StreamExecutionEnvironment
from pyflink.table import (
    EnvironmentSettings,
    TableEnvironment,
    DataTypes
)
from pyflink.table.udf import udf
from pyflink.datastream.connectors import KafkaSink
from pyflink.common.serialization import KafkaRecordSerializationSchema, SimpleStringSchema
from pyflink.table.descriptors import Schema

def main():
    # 1. Set up TableEnvironment in streaming mode
    env_settings = EnvironmentSettings.in_streaming_mode()
    t_env = TableEnvironment.create(env_settings)

    # 2. Register Postgres CDC source table
    t_env.execute_sql("""
    CREATE TABLE fraud_dec_cdc (
      V1            FLOAT(53),
      V2            FLOAT(53),
      V3            FLOAT(53),
      V4            FLOAT(53),
      V5            FLOAT(53),
      V6            FLOAT(53),
      V7            FLOAT(53),
      V8            FLOAT(53),
      V9            FLOAT(53),
      V10           FLOAT(53),
      V11           FLOAT(53),
      V12           FLOAT(53),
      V13           FLOAT(53),
      V14           FLOAT(53),
      V15           FLOAT(53),
      V16           FLOAT(53),
      V17           FLOAT(53),
      V18           FLOAT(53),
      V19           FLOAT(53),
      V20           FLOAT(53),
      V21           FLOAT(53),
      V22           FLOAT(53),
      V23           FLOAT(53),
      V24           FLOAT(53),
      V25           FLOAT(53),
      V26           FLOAT(53),
      V27           FLOAT(53),
      V28           FLOAT(53),
      ScaledAmount  FLOAT(53),
      Hour_Scaled   FLOAT(53),
      PRIMARY KEY (V1) NOT ENFORCED
    ) WITH (
      'connector'            = 'postgres-cdc',
      'hostname'             = 'floo3.postgres.database.azure.com',
      'port'                 = '5432',
      'database-name'        = 'postgres',
      'schema-name'          = 'public',
      'table-name'           = 'fraud_dec_test',
      'username'             = 'floo',
      'password'             = 'Agents1234',
      'decoding.plugin.name' = 'pgoutput',
      'scan.startup.mode'    = 'initial'
    );
    """)  # :contentReference[oaicite:4]{index=4}

    # 3. Define a UDF to call the ZenML model server
    @udf(result_type=DataTypes.INT())
    def predict_label(
        V1: float, V2: float, V3: float, V4: float, V5: float,
        V6: float, V7: float, V8: float, V9: float, V10: float,
        V11: float, V12: float, V13: float, V14: float, V15: float,
        V16: float, V17: float, V18: float, V19: float, V20: float,
        V21: float, V22: float, V23: float, V24: float, V25: float,
        V26: float, V27: float, V28: float, ScaledAmount: float,
        Hour_Scaled: float
    ) -> int:
        """
        Sends one record to ZenML inference endpoint and returns 0 or 1.
        """
        payload = {
            "instances": [[
                V1, V2, V3, V4, V5,
                V6, V7, V8, V9, V10,
                V11, V12, V13, V14, V15,
                V16, V17, V18, V19, V20,
                V21, V22, V23, V24, V25,
                V26, V27, V28, ScaledAmount,
                Hour_Scaled
            ]]
        }
        resp = requests.post(
            "http://127.0.0.1:8000/invocations",
            json=payload
        )
        return int(resp.json()[0])  # :contentReference[oaicite:5]{index=5}

    # 4. Apply UDF to the CDC table
    result = t_env.from_path("fraud_dec_cdc") \
        .add_columns("predict_label( " +
                     ",".join(f"V{i}" for i in range(1,29)) +
                     ",ScaledAmount,Hour_Scaled) AS label")

    # 5. Split into fraud (label=1) and legit (label=0)
    fraud_table = result.filter("label = 1")
    legit_table = result.filter("label = 0")

    # 6. JDBC Sink for fraud cases (create table if not exists)
    t_env.execute_sql("""
    CREATE TABLE IF NOT EXISTS fraud (
      V1            FLOAT(53),
      V2            FLOAT(53),
      V3            FLOAT(53),
      V4            FLOAT(53),
      V5            FLOAT(53),
      V6            FLOAT(53),
      V7            FLOAT(53),
      V8            FLOAT(53),
      V9            FLOAT(53),
      V10           FLOAT(53),
      V11           FLOAT(53),
      V12           FLOAT(53),
      V13           FLOAT(53),
      V14           FLOAT(53),
      V15           FLOAT(53),
      V16           FLOAT(53),
      V17           FLOAT(53),
      V18           FLOAT(53),
      V19           FLOAT(53),
      V20           FLOAT(53),
      V21           FLOAT(53),
      V22           FLOAT(53),
      V23           FLOAT(53),
      V24           FLOAT(53),
      V25           FLOAT(53),
      V26           FLOAT(53),
      V27           FLOAT(53),
      V28           FLOAT(53),
      ScaledAmount  FLOAT(53),
      Hour_Scaled   FLOAT(53)
    ) WITH (
      'connector' = 'jdbc',
      'url'       = 'jdbc:postgresql://floo3.postgres.database.azure.com:5432/postgres',
      'table-name' = 'fraud',
      'username'  = 'floo',
      'password'  = 'Agents1234'
    );
    """)  # :contentReference[oaicite:6]{index=6}

    # 7. Kafka Sink for legit cases
    ds_legit = t_env.to_append_stream(
        legit_table,
        DataTypes.ROW([
            DataTypes.FIELD(f"V{i}", DataTypes.FLOAT()) for i in range(1,29)
        ] + [
            DataTypes.FIELD("ScaledAmount", DataTypes.FLOAT()),
            DataTypes.FIELD("Hour_Scaled", DataTypes.FLOAT()),
            DataTypes.FIELD("label", DataTypes.INT())
        ])
    )

    kafka_sink = KafkaSink.builder() \
        .set_bootstrap_servers("kafka:9092") \
        .set_record_serializer(
            KafkaRecordSerializationSchema.builder()
                .set_topic("legit_transactions")
                .set_value_serialization_schema(SimpleStringSchema())
                .build()
        ) \
        .set_delivery_guarantee("AT_LEAST_ONCE") \
        .build()  # :contentReference[oaicite:7]{index=7}

    ds_legit.sink_to(kafka_sink)

    # 8. Execute the pipeline
    t_env.execute("fraud_detection_pipeline")


if __name__ == "__main__":
    main()
