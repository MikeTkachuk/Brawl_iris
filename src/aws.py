import boto3
import paramiko


class InstanceContext:
    def __init__(self, instance_id, region_name="us-east-1"):
        ec2 = boto3.resource('ec2', region_name=region_name)
        instance = ec2.Instance(instance_id)

        self.instance = instance
        self.command_client = None

    def connect(self, key_file):
        key = paramiko.RSAKey.from_private_key_file(str(key_file))
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        client.connect(hostname=self.ip, username="ec2-user", pkey=key)
        self.command_client = client
        print('aws.InstanceContext: connected successfully')

    def exec_command(self, command):
        if self.command_client is None:
            raise RuntimeError(f"Not connected to instance. Please make sure to call {self}.connect beforehand")
        print(f'aws.InstanceContext: running {command}')
        stdin, stdout, stderr = self.command_client.exec_command(command)
        stdin.close()
        for line in iter(stdout.readline, ""):  # https://docs.python.org/3/library/functions.html#iter
            print(line, end="")
        for line in iter(stderr.readline, ""):  # https://docs.python.org/3/library/functions.html#iter
            print(line, end="")
        print(f'aws.InstanceContext: finished execution')

    @property
    def ip(self):
        return self.instance.public_ip_address

    def __enter__(self):
        # TODO track running time
        self.instance.start()
        print('aws.InstanceContext: instance start requested')
        self.instance.wait_until_running()
        print('aws.InstanceContext: instance started')
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.command_client is not None:
            self.command_client.close()
        self.instance.stop()
        print('aws.InstanceContext: instance stop requested')
        self.instance.wait_until_stopped()
        print('aws.InstanceContext: instance stopped')

