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

        self._termination_key = Event()
        self._thread = Thread(target=self._listen, args=(self,))
        self._ref_hash = hash('hash')

    def start(self):
        self._thread.start()

    def stop(self):
        self._termination_key.set()
        self._thread.join()

    def _listen(self):
        while True:
            if self._termination_key.is_set():
                break
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

            time.sleep(self.interval)
