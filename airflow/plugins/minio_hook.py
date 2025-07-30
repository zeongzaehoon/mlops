# plugins/hooks/minio_hook.py
import boto3
from botocore.exceptions import ClientError
import logging
import time

class MinIOManager:
    """MinIO 버킷 자동 생성 및 관리"""

    def __init__(self, max_retries=5, retry_delay=2):
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.client = None
        self._connect_with_retry()

    def _connect_with_retry(self):
        """재시도 로직이 있는 MinIO 연결"""
        for attempt in range(self.max_retries):
            try:
                self.client = boto3.client(
                    's3',
                    endpoint_url='http://minio:9000',
                    aws_access_key_id='minioadmin',
                    aws_secret_access_key='minioadmin',
                    region_name='us-east-1'
                )

                # 연결 테스트
                self.client.list_buckets()
                logging.info("✅ MinIO 연결 성공")
                break

            except Exception as e:
                logging.warning(f"MinIO 연결 실패 (시도 {attempt + 1}/{self.max_retries}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                else:
                    raise ConnectionError(f"MinIO 연결 실패: {e}")

    def ensure_buckets_exist(self):
        """필요한 버킷들 생성"""
        required_buckets = ['mlops-data', 'models', 'artifacts']

        for bucket_name in required_buckets:
            try:
                self.client.head_bucket(Bucket=bucket_name)
                logging.info(f"✅ 버킷 '{bucket_name}' 이미 존재")
            except ClientError:
                try:
                    self.client.create_bucket(Bucket=bucket_name)
                    logging.info(f"🆕 버킷 '{bucket_name}' 생성 완료")

                    # 개발용 공개 읽기 정책 설정 (선택사항)
                    self._set_public_read_policy(bucket_name)

                except Exception as e:
                    logging.error(f"❌ 버킷 '{bucket_name}' 생성 실패: {e}")

    def _set_public_read_policy(self, bucket_name):
        """개발용 공개 읽기 정책 설정"""
        try:
            policy = {
                "Version": "2012-10-17",
                "Statement": [
                    {
                        "Effect": "Allow",
                        "Principal": "*",
                        "Action": "s3:GetObject",
                        "Resource": f"arn:aws:s3:::{bucket_name}/*"
                    }
                ]
            }

            import json
            self.client.put_bucket_policy(
                Bucket=bucket_name,
                Policy=json.dumps(policy)
            )
            logging.info(f"📋 버킷 '{bucket_name}' 공개 읽기 정책 설정 완료")
        except Exception as e:
            logging.warning(f"⚠️ 정책 설정 실패 (무시 가능): {e}")

# 전역 인스턴스 (필요시 사용)
minio_manager = None

def get_minio_manager():
    """MinIO 매니저 싱글톤 인스턴스 반환"""
    global minio_manager
    if minio_manager is None:
        minio_manager = MinIOManager()
        minio_manager.ensure_buckets_exist()
    return minio_manager