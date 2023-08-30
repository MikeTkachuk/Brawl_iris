import time

import boto3
import paramiko


class InstanceContext:
    def __init__(self, instance_id, region_name="us-east-1"):
        ec2 = boto3.resource('ec2', region_name=region_name)
        instance = ec2.Instance(instance_id)

        self.instance = instance
        self.command_client = None
        self._start_time = None
        self._stop_time = None

    @property
    def session_time(self):
        """
        Returns time elapsed since instance start till instance stop in seconds.
         If instance is not stopped time.time() is used
        """
        if self._start_time is None:
            raise RuntimeError("Session has never been started. Can not get session time")
        period_end = self._stop_time if self._stop_time is not None else time.time()
        return period_end - self._start_time

    def connect(self, key_file):
        key = paramiko.RSAKey.from_private_key_file(str(key_file))
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        client.connect(hostname=self.ip, username="ec2-user", pkey=key)
        self.command_client = client
        print('aws.InstanceContext: connected successfully')

    def exec_command(self, command, quiet=False):
        if self.command_client is None:
            raise RuntimeError(f"Not connected to instance. Please make sure to call {self}.connect beforehand")
        try:
            print(f'aws.InstanceContext: running {command}') if not quiet else None
            stdin, stdout, stderr = self.command_client.exec_command(command)
            stdin.close()
            for line in iter(stdout.readline, ""):  # https://docs.python.org/3/library/functions.html#iter
                print(line, end="")
            for line in iter(stderr.readline, ""):  # https://docs.python.org/3/library/functions.html#iter
                print(line, end="")
            print(f'aws.InstanceContext: finished execution') if not quiet else None
        except paramiko.ssh_exception.SSHException as e:
            print(e)
            print("Failed to execute command. Connection was closed")

    @property
    def ip(self):
        return self.instance.public_ip_address

    def __enter__(self):
        self.instance.start()
        print('aws.InstanceContext: instance start requested')
        self.instance.wait_until_running()
        print('aws.InstanceContext: instance started')
        self._start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.command_client is not None:
            self.command_client.close()
        self.instance.stop()
        print('aws.InstanceContext: instance stop requested')
        self.instance.wait_until_stopped()
        print('aws.InstanceContext: instance stopped')
        self._stop_time = time.time()

