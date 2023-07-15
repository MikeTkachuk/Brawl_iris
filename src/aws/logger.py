import json
import time
from io import BytesIO
from threading import Thread, Event


class LogListener:
    """
    Tracks the change in the specified json file on cloud storage
     Then invokes log_func as log_func(json.loads(file_contents))
    """

    def __init__(self, log_func, path_to_listen, bucket_name, storage_client):
        self.path_to_listen = path_to_listen
        self.bucket_name = bucket_name
        self.interval = 10
        self.log_func = log_func
        self.storage_client = storage_client

        self._termination_key = None
        self._thread = None
        self._ref_hash = hash('hash')

        self.storage_client.delete_object(Bucket=self.bucket_name,
                                          Key=self.path_to_listen)  # cleanup log files from prev run

    def init(self, storage_client=None):
        """
          Listener needs to be init before each start() call

        :param storage_client: s3 storage client. should be set here or in __init__
        :return:
        """
        self._termination_key = Event()
        self._thread = Thread(target=self._listen, args=())
        self.storage_client = storage_client if storage_client is not None else self.storage_client

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
                to_log = data.read().decode('utf-8')
                to_log_hash = hash(to_log)
                if self._ref_hash != to_log_hash:
                    self._ref_hash = to_log_hash
                    self.log_func(json.loads(to_log))
            except Exception:
                pass

            time.sleep(self.interval)
