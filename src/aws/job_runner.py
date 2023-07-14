import time
from pathlib import Path
import os

import boto3

from compute_instance import InstanceContext
from logger import LogListener


class JobRunner:
    def __init__(self,
                 bucket_name: str,
                 run_prefix: str,
                 instance_id: str,
                 region_name: str,
                 ssh_file_path,
                 commands,
                 logger: LogListener = None
                 ):
        """

        :param bucket_name:
        :param run_prefix:
        :param instance_id:
        :param region_name:
        :param ssh_file_path:
        :param commands: list of strings. Will be executed in the order provided.
        :param logger: optional instance of log listener.
        """
        self.bucket_name = bucket_name
        self.run_prefix = run_prefix

        self.instance_context = InstanceContext(instance_id, region_name)
        self.ssh_file_path = ssh_file_path
        self.commands = commands
        self.logger = logger

    def upload_code(self, storage_client=None):
        if storage_client is None:
            storage_client = boto3.client('s3')
        repo_root = Path(__file__).parents[2]  # ->Brawl_iris/src/aws/job_runner.py

        # upload code if not already
        if storage_client.list_objects_v2(Bucket=self.bucket_name,
                                     Prefix=f"{self.run_prefix}/{repo_root.name}"
                                     )['KeyCount'] < 3:
            print('JobRunner.upload_code: code upload started')
            # upload code if needed
            name_on_bucket = f"{self.run_prefix}/{repo_root.name}"
            os.system(f'aws s3 cp {repo_root} s3://{self.bucket_name}/{name_on_bucket} '
                      f'--exclude ".git/*" '
                      f'--exclude ".idea/*" '
                      f'--exclude "results/*" '
                      f'--exclude "assets/*" '
                      f'--recursive '
                      f'--quiet')
            print('JobRunner.upload_code: code upload finished')
        else:
            print('JobRunner.upload_code: Skipped code upload')

    def init_job(self):
        s3_client = boto3.client('s3')
        self.logger.init(s3_client)
        self.upload_code(s3_client)

    def run(self):

        self.init_job()
        with self.instance_context:
            self.instance_context.connect(self.ssh_file_path)
            if self.logger is not None:
                self.logger.init()
                self.logger.start()

            time.sleep(5)  # prevents unfinished initializations
            for command in self.commands:
                self.instance_context.exec_command(command)

            if self.logger is not None:
                self.logger.stop()

        return self.instance_context.session_time
