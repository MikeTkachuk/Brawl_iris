import time
from pathlib import Path
import os
from typing import Iterable, Union
import zipfile

import boto3

from src.aws.compute_instance import InstanceContext
from src.aws.logger import LogListener


class JobRunner:
    def __init__(self,
                 bucket_name: str,
                 run_prefix: str,
                 instance_id: str,
                 region_name: str,
                 ssh_file_path: Union[Path, str],
                 commands: Iterable[str],
                 loggers: Iterable[LogListener] = None
                 ):
        """

        :param bucket_name:
        :param run_prefix:
        :param instance_id:
        :param region_name:
        :param ssh_file_path:
        :param commands: list of strings. Will be executed in the order provided.
        :param loggers: optional iterable of log listeners. starts every listener on job run.
        """
        self.bucket_name = bucket_name
        self.run_prefix = run_prefix

        self.instance_context = InstanceContext(instance_id, region_name)
        self.ssh_file_path = ssh_file_path
        self.commands = commands
        self.loggers = loggers

    def upload_code(self, storage_client=None):
        if storage_client is None:
            storage_client = boto3.client('s3')
        repo_root = Path(__file__).parents[2]  # ->Brawl_iris/src/aws/job_runner.py

        # upload code if not already
        if storage_client.list_objects_v2(Bucket=self.bucket_name,
                                          Prefix=f"{self.run_prefix}/{repo_root.name}.zip"
                                          )['KeyCount'] < 1:
            zip_path = Path(repo_root.name + '.zip')
            with zipfile.ZipFile(zip_path, mode='w') as zip_code:
                for file in repo_root.rglob("*"):
                    rel_file = file.relative_to(repo_root)
                    if Path(".git") in rel_file.parents or \
                            Path(".idea") in rel_file.parents or \
                            Path("results") in rel_file.parents or \
                            Path("assets") in rel_file.parents or \
                            Path("src/outputs") in rel_file.parents:
                        continue
                    zip_code.write(file, arcname=rel_file)

            print('JobRunner.upload_code: code upload started')
            # upload code if needed
            name_on_bucket = f"{self.run_prefix}/{zip_path}"
            os.system(f'aws s3 cp {zip_path} s3://{self.bucket_name}/{name_on_bucket} '
                      f'--quiet')
            zip_path.unlink()
            print('JobRunner.upload_code: code upload finished')
        else:
            print('JobRunner.upload_code: Skipped code upload')

    def init_job(self):
        s3_client = boto3.client('s3')
        for logger in self.loggers:
            logger.init(s3_client)
        self.upload_code(s3_client)

    def run(self):
        self.init_job()
        with self.instance_context:
            self.instance_context.connect(self.ssh_file_path)
            if self.loggers is not None:
                for logger in self.loggers:
                    logger.start()
            try:
                time.sleep(5)  # prevents unfinished initializations
                for command in self.commands:
                    self.instance_context.exec_command(command)
            finally:
                if self.loggers is not None:
                    for logger in self.loggers:
                        logger.stop()

        return self.instance_context.session_time
