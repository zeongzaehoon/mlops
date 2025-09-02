#!/bin/bash
set -e

if [[ -z "${AIRFLOW_UID}" ]]; then
  echo
  echo -e "\033[1;33mWARNING!!!: AIRFLOW_UID not set!\e[0m"
  export AIRFLOW_UID=$(id -u)
fi

echo "Creating missing opt dirs if missing:"
mkdir -v -p /opt/airflow/{logs,dags,plugins,config}

echo "Airflow version:"
/entrypoint airflow version

echo "Change ownership of files in /opt/airflow to ${AIRFLOW_UID}:0"
chown -R "${AIRFLOW_UID}:0" /opt/airflow/

echo "Files in shared volumes:"
ls -la /opt/airflow/{logs,dags,plugins,config}