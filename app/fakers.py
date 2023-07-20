from datetime import datetime


class FakeClock:

    def set_start_time(self, start_time):
        self._fake_start_time = datetime.fromtimestamp(start_time)
    
    def start(self):
        self._true_start_time = datetime.now()

    def now(self):
        return self._fake_start_time + (datetime.now() - self._true_start_time)


