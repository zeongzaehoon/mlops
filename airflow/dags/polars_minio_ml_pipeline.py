# MongoDB → Polars → MinIO(Parquet) → ML Training 파이프라인

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.mongo.hooks.mongo import MongoHook
import polars as pl
import numpy as np
import json
import os
from typing import List, Dict, Any
import boto3
from botocore.exceptions import ClientError

# DAG 기본 설정
default_args = {
    'owner': 'mlops-team',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=10),
}

dag = DAG(
    'polars_minio_ml_pipeline',
    default_args=default_args,
    description='Polars + MinIO 기반 MongoDB → ML Training 파이프라인',
    schedule_interval='@daily',
    catchup=False,
    max_active_runs=1,
    tags=['polars', 'minio', 'ml', 'transformer', 'mongodb']
)

# ===== MinIO 클라이언트 설정 =====

class MinIOManager:
    """MinIO 스토리지 매니저"""

    def __init__(self):
        self.client = boto3.client(
            's3',
            endpoint_url='http://minio:9000',
            aws_access_key_id='minioadmin',
            aws_secret_access_key='minioadmin',
            region_name='us-east-1'
        )
        self.bucket_name = 'mlops-data'
        self._ensure_bucket_exists()

    def _ensure_bucket_exists(self):
        """버킷 존재 확인 및 생성"""
        try:
            self.client.head_bucket(Bucket=self.bucket_name)
        except ClientError:
            self.client.create_bucket(Bucket=self.bucket_name)
            print(f"버킷 '{self.bucket_name}' 생성 완료")

    def upload_parquet(self, df: pl.DataFrame, key: str, metadata: dict = None):
        """Polars DataFrame을 Parquet으로 MinIO 업로드"""

        # 임시 파일로 저장
        temp_file = f"/tmp/{key.replace('/', '_')}.parquet"
        df.write_parquet(temp_file, compression='snappy')

        # MinIO 업로드
        extra_args = {}
        if metadata:
            extra_args['Metadata'] = {k: str(v) for k, v in metadata.items()}

        self.client.upload_file(temp_file, self.bucket_name, key, ExtraArgs=extra_args)

        # 임시 파일 삭제
        os.remove(temp_file)

        print(f"MinIO 업로드 완료: s3://{self.bucket_name}/{key}")
        return f"s3://{self.bucket_name}/{key}"

    def download_parquet(self, key: str) -> pl.DataFrame:
        """MinIO에서 Parquet 다운로드 후 Polars DataFrame 반환"""

        temp_file = f"/tmp/{key.replace('/', '_')}_download.parquet"
        self.client.download_file(self.bucket_name, key, temp_file)

        df = pl.read_parquet(temp_file)
        os.remove(temp_file)

        return df

    def list_objects(self, prefix: str) -> List[str]:
        """특정 prefix로 시작하는 객체 목록 반환"""
        response = self.client.list_objects_v2(Bucket=self.bucket_name, Prefix=prefix)
        return [obj['Key'] for obj in response.get('Contents', [])]

# ===== 1. MongoDB 데이터 추출 (Polars 최적화) =====

def extract_mongodb_to_polars(**context):
    """MongoDB에서 데이터 추출 후 Polars DataFrame으로 변환"""

    execution_date = context['ds']
    start_date = datetime.strptime(execution_date, '%Y-%m-%d')
    end_date = start_date + timedelta(days=1)

    print(f"📊 데이터 추출 시작: {start_date} ~ {end_date}")

    # MongoDB 연결
    mongo_hook = MongoHook(conn_id='mongodb_default')
    collection = mongo_hook.get_collection('user_events')

    # 쿼리 조건
    query = {
        'timestamp': {'$gte': start_date, '$lt': end_date},
        'event_type': {'$in': ['click', 'view', 'search', 'purchase', 'like', 'share']},
        'user_id': {'$exists': True, '$ne': None}
    }

    # 데이터 추출 (배치 처리)
    batch_size = 50000  # Polars는 대용량 처리에 강함
    cursor = collection.find(query).batch_size(batch_size)

    # 데이터를 리스트로 수집 (Polars 최적화)
    records = []
    for doc in cursor:
        # MongoDB ObjectId 처리
        doc['_id'] = str(doc['_id'])
        doc['timestamp'] = doc['timestamp'].isoformat()
        records.append(doc)

    print(f"📦 추출된 레코드: {len(records):,}개")

    if not records:
        raise ValueError("추출된 데이터가 없습니다.")

    # Polars DataFrame 생성 (스키마 추론 자동)
    df = pl.DataFrame(records)

    print(f"🔧 Polars DataFrame 생성: {df.shape}")
    print(f"📋 컬럼: {df.columns}")

    # MinIO에 Raw 데이터 저장
    minio = MinIOManager()
    raw_key = f"raw-data/year={start_date.year}/month={start_date.month:02d}/day={start_date.day:02d}/data.parquet"

    metadata = {
        'extraction_date': execution_date,
        'record_count': len(records),
        'source': 'mongodb_user_events'
    }

    minio.upload_parquet(df, raw_key, metadata)

    return {
        'raw_data_key': raw_key,
        'record_count': len(records),
        'execution_date': execution_date,
        'schema': df.schema
    }

# ===== 2. Polars 데이터 정제 및 변환 =====

def clean_data_with_polars(**context):
    """Polars를 사용한 고성능 데이터 정제"""

    ti = context['task_instance']
    extract_info = ti.xcom_pull(task_ids='extract_mongodb_to_polars')

    print(f"🧹 데이터 정제 시작: {extract_info['record_count']:,}개 레코드")

    # MinIO에서 데이터 로드
    minio = MinIOManager()
    df = minio.download_parquet(extract_info['raw_data_key'])

    print(f"📊 원본 데이터: {df.shape}")

    # Polars 체이닝을 활용한 고성능 데이터 정제
    cleaned_df = (
        df
        # 1. 기본 필터링
        .filter(
            pl.col('user_id').is_not_null() &
            pl.col('event_type').is_not_null() &
            pl.col('timestamp').is_not_null()
        )
        # 2. 데이터 타입 변환
        .with_columns([
            pl.col('timestamp').str.strptime(pl.Datetime, format='%Y-%m-%dT%H:%M:%S').alias('timestamp'),
            pl.col('user_id').cast(pl.Utf8),
            pl.col('event_type').cast(pl.Utf8)
        ])
        # 3. 파생 컬럼 생성
        .with_columns([
            pl.col('timestamp').dt.hour().alias('hour'),
            pl.col('timestamp').dt.weekday().alias('day_of_week'),
            pl.col('timestamp').dt.date().alias('date'),
            # 세션 ID 생성 (같은 사용자의 1시간 내 이벤트는 같은 세션)
            (pl.col('user_id') + '_' +
             pl.col('timestamp').dt.strftime('%Y%m%d%H')).alias('session_id')
        ])
        # 4. 이상값 제거
        .filter(
            # 너무 짧은 duration 제거
            pl.when(pl.col('duration_seconds').is_not_null())
            .then(pl.col('duration_seconds') >= 0)
            .otherwise(True)
        )
    )

    print(f"✨ 정제된 데이터: {cleaned_df.shape}")

    # 사용자별 이벤트 수 계산 (Polars 고성능 집계)
    user_stats = (
        cleaned_df
        .group_by('user_id')
        .agg([
            pl.count().alias('event_count'),
            pl.col('event_type').n_unique().alias('unique_events'),
            pl.col('timestamp').min().alias('first_event'),
            pl.col('timestamp').max().alias('last_event')
        ])
        .with_columns([
            (pl.col('last_event') - pl.col('first_event')).dt.total_seconds().alias('session_duration_sec')
        ])
    )

    # 최소 이벤트 수 필터링 (활성 사용자만)
    min_events = 5
    valid_users = user_stats.filter(pl.col('event_count') >= min_events)['user_id']

    # 유효 사용자만 필터링
    final_df = cleaned_df.filter(pl.col('user_id').is_in(valid_users))

    print(f"👥 유효 사용자: {valid_users.len():,}명")
    print(f"📈 최종 데이터: {final_df.shape}")

    # 데이터 품질 리포트 생성
    quality_report = {
        'original_count': extract_info['record_count'],
        'cleaned_count': final_df.height,
        'data_retention_rate': final_df.height / extract_info['record_count'],
        'unique_users': final_df['user_id'].n_unique(),
        'unique_events': final_df['event_type'].n_unique(),
        'event_distribution': final_df['event_type'].value_counts().to_dict(),
        'date_range': {
            'start': final_df['timestamp'].min().isoformat(),
            'end': final_df['timestamp'].max().isoformat()
        }
    }

    print(f"📊 데이터 보존율: {quality_report['data_retention_rate']:.2%}")

    # MinIO에 정제된 데이터 저장
    execution_date = extract_info['execution_date']
    start_date = datetime.strptime(execution_date, '%Y-%m-%d')

    clean_key = f"clean-data/year={start_date.year}/month={start_date.month:02d}/day={start_date.day:02d}/cleaned.parquet"
    stats_key = f"clean-data/year={start_date.year}/month={start_date.month:02d}/day={start_date.day:02d}/user_stats.parquet"

    # 정제된 데이터 업로드
    minio.upload_parquet(final_df, clean_key, {
        'stage': 'cleaned',
        'record_count': final_df.height,
        'unique_users': final_df['user_id'].n_unique()
    })

    # 사용자 통계 업로드
    minio.upload_parquet(user_stats, stats_key, {
        'stage': 'user_statistics',
        'user_count': user_stats.height
    })

    return {
        'clean_data_key': clean_key,
        'user_stats_key': stats_key,
        'quality_report': quality_report,
        'execution_date': execution_date
    }

# ===== 3. Transformer 시퀀스 생성 (Polars 최적화) =====

def create_sequences_with_polars(**context):
    """Polars를 활용한 고성능 시퀀스 생성"""

    ti = context['task_instance']
    clean_info = ti.xcom_pull(task_ids='clean_data_with_polars')

    print(f"🔗 시퀀스 생성 시작")

    # MinIO에서 정제된 데이터 로드
    minio = MinIOManager()
    df = minio.download_parquet(clean_info['clean_data_key'])

    # 이벤트 어휘집 생성
    event_vocab = {
        'click': 1, 'view': 2, 'search': 3,
        'purchase': 4, 'like': 5, 'share': 6,
        '[PAD]': 0, '[CLS]': 7, '[SEP]': 8, '[UNK]': 9
    }

    max_sequence_length = 128  # Transformer 최적 길이

    print(f"📚 이벤트 어휘집 크기: {len(event_vocab)}")

    # Polars를 활용한 고성능 시퀀스 생성
    sequence_df = (
        df
        # 1. 이벤트 토큰화
        .with_columns([
            pl.col('event_type').map_elements(
                lambda x: event_vocab.get(x, event_vocab['[UNK]']),
                return_dtype=pl.Int32
            ).alias('event_token')
        ])
        # 2. 사용자별 시간순 정렬
        .sort(['user_id', 'timestamp'])
        # 3. 사용자별 시퀀스 집계
        .group_by('user_id')
        .agg([
            pl.col('event_token').alias('event_sequence'),
            pl.col('hour').alias('hour_sequence'),
            pl.col('day_of_week').alias('day_sequence'),
            pl.col('timestamp').alias('timestamp_sequence'),
            pl.count().alias('sequence_length')
        ])
        # 4. 적절한 길이의 시퀀스만 선택
        .filter(
            (pl.col('sequence_length') >= 10) &  # 최소 길이
            (pl.col('sequence_length') <= max_sequence_length * 2)  # 최대 길이
        )
    )

    print(f"👤 시퀀스 생성된 사용자: {sequence_df.height:,}명")

    # 시퀀스 패딩 및 분할 함수 (Polars UDF)
    def process_sequence(events: List[int], hours: List[int], days: List[int]) -> Dict:
        """시퀀스 처리: 패딩, 분할, CLS/SEP 추가"""

        sequences = []

        if len(events) <= max_sequence_length - 2:  # CLS, SEP 토큰 공간
            # 짧은 시퀀스: 패딩
            padded_events = [event_vocab['[CLS]']] + events + [event_vocab['[SEP]']]
            padded_hours = [0] + hours + [0]
            padded_days = [0] + days + [0]

            # 패딩 추가
            while len(padded_events) < max_sequence_length:
                padded_events.append(event_vocab['[PAD]'])
                padded_hours.append(0)
                padded_days.append(0)

            sequences.append({
                'events': padded_events,
                'hours': padded_hours,
                'days': padded_days,
                'length': len(events) + 2,
                'attention_mask': [1 if x != event_vocab['[PAD]'] else 0 for x in padded_events]
            })
        else:
            # 긴 시퀀스: 슬라이딩 윈도우로 분할
            window_size = max_sequence_length - 2
            step_size = window_size // 2

            for i in range(0, len(events) - window_size + 1, step_size):
                window_events = events[i:i + window_size]
                window_hours = hours[i:i + window_size]
                window_days = days[i:i + window_size]

                # CLS, SEP 추가
                full_events = [event_vocab['[CLS]']] + window_events + [event_vocab['[SEP]']]
                full_hours = [0] + window_hours + [0]
                full_days = [0] + window_days + [0]

                sequences.append({
                    'events': full_events,
                    'hours': full_hours,
                    'days': full_days,
                    'length': len(full_events),
                    'attention_mask': [1] * len(full_events)
                })

        return sequences

    # 모든 시퀀스 처리 (Python으로 처리 후 Polars로 변환)
    all_sequences = []

    for row in sequence_df.iter_rows(named=True):
        user_sequences = process_sequence(
            row['event_sequence'],
            row['hour_sequence'],
            row['day_sequence']
        )

        for seq in user_sequences:
            all_sequences.append({
                'user_id': row['user_id'],
                'input_ids': seq['events'],
                'hours': seq['hours'],
                'days': seq['days'],
                'attention_mask': seq['attention_mask'],
                'length': seq['length']
            })

    print(f"🎯 생성된 시퀀스: {len(all_sequences):,}개")

    # Polars DataFrame으로 변환
    sequences_df = pl.DataFrame(all_sequences)

    # 훈련/검증 분할 (Polars 샘플링 활용)
    train_df = sequences_df.sample(fraction=0.8, seed=42)
    val_df = sequences_df.filter(~pl.col('user_id').is_in(train_df['user_id']))

    print(f"🏋️ 훈련 세트: {train_df.height:,}개")
    print(f"✅ 검증 세트: {val_df.height:,}개")

    # 메타데이터 생성
    metadata = {
        'vocab_size': len(event_vocab),
        'max_sequence_length': max_sequence_length,
        'event_vocab': event_vocab,
        'train_count': train_df.height,
        'val_count': val_df.height,
        'total_sequences': sequences_df.height,
        'avg_sequence_length': float(sequences_df['length'].mean()),
        'unique_users': sequences_df['user_id'].n_unique()
    }

    # MinIO에 저장
    execution_date = clean_info['execution_date']
    start_date = datetime.strptime(execution_date, '%Y-%m-%d')

    base_path = f"ml-sequences/year={start_date.year}/month={start_date.month:02d}/day={start_date.day:02d}"
    train_key = f"{base_path}/train_sequences.parquet"
    val_key = f"{base_path}/val_sequences.parquet"
    metadata_key = f"{base_path}/metadata.json"

    # 시퀀스 데이터 업로드
    minio.upload_parquet(train_df, train_key, {
        'stage': 'ml_sequences',
        'split': 'train',
        'sequence_count': train_df.height
    })

    minio.upload_parquet(val_df, val_key, {
        'stage': 'ml_sequences',
        'split': 'validation',
        'sequence_count': val_df.height
    })

    # 메타데이터 업로드
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(metadata, f, indent=2)
        temp_path = f.name

    minio.client.upload_file(temp_path, minio.bucket_name, metadata_key)
    os.unlink(temp_path)

    print(f"💾 MinIO 저장 완료:")
    print(f"  - 훈련: s3://{minio.bucket_name}/{train_key}")
    print(f"  - 검증: s3://{minio.bucket_name}/{val_key}")
    print(f"  - 메타: s3://{minio.bucket_name}/{metadata_key}")

    return {
        'train_key': train_key,
        'val_key': val_key,
        'metadata_key': metadata_key,
        'metadata': metadata,
        'execution_date': execution_date
    }

# ===== 4. 최종 ML 데이터 준비 =====

def prepare_final_ml_data(**context):
    """최종 ML 훈련 형식으로 데이터 준비"""

    ti = context['task_instance']
    sequence_info = ti.xcom_pull(task_ids='create_sequences_with_polars')

    print(f"🎯 최종 ML 데이터 준비")

    minio = MinIOManager()

    # 훈련/검증 데이터 로드
    train_df = minio.download_parquet(sequence_info['train_key'])
    val_df = minio.download_parquet(sequence_info['val_key'])

    print(f"📊 데이터 로드 완료:")
    print(f"  - 훈련: {train_df.shape}")
    print(f"  - 검증: {val_df.shape}")

    # HuggingFace Datasets 호환 형식으로 변환
    def convert_to_hf_format(df: pl.DataFrame) -> pl.DataFrame:
        """HuggingFace Datasets 호환 형식으로 변환"""
        return df.select([
            pl.col('input_ids'),
            pl.col('attention_mask'),
            pl.col('hours'),
            pl.col('days'),
            pl.col('user_id'),
            pl.col('length')
        ])

    hf_train_df = convert_to_hf_format(train_df)
    hf_val_df = convert_to_hf_format(val_df)

    # 최종 통계 계산
    final_stats = {
        'dataset_info': {
            'train_size': hf_train_df.height,
            'val_size': hf_val_df.height,
            'vocab_size': sequence_info['metadata']['vocab_size'],
            'max_length': sequence_info['metadata']['max_sequence_length']
        },
        'sequence_stats': {
            'avg_length_train': float(hf_train_df['length'].mean()),
            'avg_length_val': float(hf_val_df['length'].mean()),
            'min_length': int(min(hf_train_df['length'].min(), hf_val_df['length'].min())),
            'max_length': int(max(hf_train_df['length'].max(), hf_val_df['length'].max()))
        },
        'user_stats': {
            'unique_users_train': hf_train_df['user_id'].n_unique(),
            'unique_users_val': hf_val_df['user_id'].n_unique(),
            'total_unique_users': pl.concat([hf_train_df['user_id'], hf_val_df['user_id']]).n_unique()
        }
    }

    # 최종 데이터 저장
    execution_date = sequence_info['execution_date']
    start_date = datetime.strptime(execution_date, '%Y-%m-%d')

    final_path = f"ml-ready/year={start_date.year}/month={start_date.month:02d}/day={start_date.day:02d}"
    final_train_key = f"{final_path}/train_final.parquet"
    final_val_key = f"{final_path}/val_final.parquet"
    final_stats_key = f"{final_path}/final_stats.json"

    # 최종 데이터 업로드
    minio.upload_parquet(hf_train_df, final_train_key, {
        'stage': 'ml_ready',
        'split': 'train_final',
        'ready_for_training': 'true'
    })

    minio.upload_parquet(hf_val_df, final_val_key, {
        'stage': 'ml_ready',
        'split': 'val_final',
        'ready_for_training': 'true'
    })

    # 최종 통계 업로드
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(final_stats, f, indent=2)
        temp_path = f.name

    minio.client.upload_file(temp_path, minio.bucket_name, final_stats_key)
    os.unlink(temp_path)

    print(f"✅ 최종 ML 데이터 준비 완료!")
    print(f"📈 훈련 데이터: {final_stats['dataset_info']['train_size']:,}개")
    print(f"📊 검증 데이터: {final_stats['dataset_info']['val_size']:,}개")
    print(f"👥 총 사용자: {final_stats['user_stats']['total_unique_users']:,}명")

    return {
        'final_train_key': final_train_key,
        'final_val_key': final_val_key,
        'final_stats_key': final_stats_key,
        'final_stats': final_stats,
        'ready_for_training': True,
        's3_path': f"s3://{minio.bucket_name}/{final_path}"
    }

# ===== 5. 데이터 품질 검증 =====

def validate_final_data(**context):
    """최종 데이터 품질 검증"""

    ti = context['task_instance']
    final_info = ti.xcom_pull(task_ids='prepare_final_ml_data')

    print(f"🔍 최종 데이터 품질 검증")

    minio = MinIOManager()

    # 데이터 로드 테스트
    try:
        train_df = minio.download_parquet(final_info['final_train_key'])
        val_df = minio.download_parquet(final_info['final_val_key'])
        print("✅ 데이터 로드 성공")
    except Exception as e:
        raise ValueError(f"❌ 데이터 로드 실패: {e}")

    # 검증 항목들
    validations = []

    # 1. 데이터 형태 검증
    if train_df.height > 0 and val_df.height > 0:
        validations.append("✅ 데이터 존재")
    else:
        validations.append("❌ 빈 데이터셋")

    # 2. 필수 컬럼 검증
    required_cols = ['input_ids', 'attention_mask', 'hours', 'days']
    if all(col in train_df.columns for col in required_cols):
        validations.append("✅ 필수 컬럼 존재")
    else:
        validations.append("❌ 필수 컬럼 누락")

    # 3. 시퀀스 길이 검증
    max_len = final_info['final_stats']['dataset_info']['max_length']
    if train_df['length'].max() <= max_len:
        validations.append("✅ 시퀀스 길이 적절")
    else:
        validations.append("❌ 시퀀스 길이 초과")

    # 4. 최소 데이터 크기 검증
    min_samples = 1000
    if train_df.height >= min_samples:
        validations.append("✅ 충분한 훈련 데이터")
    else:
        validations.append("⚠️ 훈련 데이터 부족")

    # 검증 결과 출력
    for validation in validations:
        print(validation)

    # 실패 조건 체크
    if "❌" in "\n".join(validations):
        raise ValueError("데이터 품질 검증 실패")

    # 검증 리포트 생성
    validation_report = {
        'validation_timestamp': datetime.now().isoformat(),
        'validations': validations,
        'sample_stats': {
            'train_sample_shape': train_df.shape,
            'val_sample_shape': val_df.shape,
            'input_ids_sample': train_df['input_ids'].head(1).to_list()[0][:10],
            'attention_mask_sample': train_df['attention_mask'].head(1).to_list()[0][:10]
        },
        'data_quality_score': validations.count("✅") / len(validations),
        's3_location': final_info['s3_path']
    }

    print(f"📊 데이터 품질 점수: {validation_report['data_quality_score']:.2%}")
    print(f"🎯 S3 경로: {final_info['s3_path']}")
    print("✅ 모든 검증 통과! 훈련 준비 완료!")

    return validation_report

# ===== DAG Task 정의 =====

# Task 1: MongoDB → Polars 추출
extract_task = PythonOperator(
    task_id='extract_mongodb_to_polars',
    python_callable=extract_mongodb_to_polars,
    dag=dag
)

# Task 2: Polars 데이터 정제
clean_task = PythonOperator(
    task_id='clean_data_with_polars',
    python_callable=clean_data_with_polars,
    dag=dag
)

# Task 3: Polars 시퀀스 생성
sequence_task = PythonOperator(
    task_id='create_sequences_with_polars',
    python_callable=create_sequences_with_polars,
    dag=dag
)

# Task 4: 최종 ML 데이터 준비
ml_prep_task = PythonOperator(
    task_id='prepare_final_ml_data',
    python_callable=prepare_final_ml_data,
    dag=dag
)

# Task 5: 데이터 품질 검증
validation_task = PythonOperator(
    task_id='validate_final_data',
    python_callable=validate_final_data,
    dag=dag
)

# Task 의존성 설정
extract_task >> clean_task >> sequence_task >> ml_prep_task >> validation_task