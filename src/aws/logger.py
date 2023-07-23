import time
from io import BytesIO
from threading import Thread, Event

import botocore.exceptions


class LogListener:
    """
    Tracks the change in the specified json file on cloud storage
     Then invokes log_func as log_func(json.loads(file_contents))
    """

    def __init__(self, log_func, path_to_listen: str, bucket_name: str, storage_client, name=None):
        self.path_to_listen = path_to_listen
        self.bucket_name = bucket_name
        self.interval = 10
        self.log_func = log_func
        self.storage_client = storage_client
        self.name = name

        self._termination_key = None
        self._thread = None
        self._ref_hash = hash('hash')
        self._counter = 1

        self.storage_client.delete_object(Bucket=self.bucket_name,
                                          Key=self.path_to_listen)  # cleanup log files from prev run

    def init(self, storage_client=None, step=None):
        """
          Listener needs to be init before each start() call

        :param storage_client: optionally update s3 storage client
        :param step: optionally updates logger step
        :return:
        """
        self._termination_key = Event()
        self._thread = Thread(target=self._listen, args=())
        if storage_client is not None:
            self.storage_client = storage_client
        if step is not None:
            self._counter = step
        print(f'aws.logger.LogListener: listening {self.path_to_listen}. init at step {self._counter}')

    def start(self):
        self._thread.start()

    def stop(self):
        self._termination_key.set()
        self._thread.join()

    def _listen(self):
        while True:
            if self._termination_key.is_set():
                break
            try:
                data = BytesIO()
                self.storage_client.download_fileobj(
                    Bucket=self.bucket_name,
                    Key=self.path_to_listen,
                    Fileobj=data
                )
                data.seek(0)
                to_log = data.read()
                to_log_hash = hash(to_log)
                if self._ref_hash != to_log_hash:
                    self._ref_hash = to_log_hash
                    self.log_func(to_log, step=self._counter, name=self.name)
                    self._counter += 1
            except Exception as e:
                if not isinstance(e, botocore.exceptions.ClientError):
                    print(f"logger._listen encountered: {e}")
                if isinstance(e, KeyboardInterrupt):
                    break

            time.sleep(self.interval)
