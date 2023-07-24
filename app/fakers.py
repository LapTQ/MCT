from datetime import datetime


class FakeClock:

    def set_start_time(self, start_time):
        self._fake_start_time = datetime.fromtimestamp(start_time)
    
    def start(self):
        self._true_start_time = datetime.now()

    def now(self):
        return self._fake_start_time + (datetime.now() - self._true_start_time)
    

fake_signin_time = {
    1: {
        4: 396,
        3: 745,
    },

    2: {
        4: 268,
        3: 570,
    },

    3: {
        4: 275,
        5: 716,
    },

    4: {
        4: 148,
        5: 626,
    },

    5: {
         4: 334,
         3: 763,
         5: 1128,
    },

    6: {
         4: 245,
         3: 517,
         5: 735,
    },

    7: {
         4: 145,
         3: 482,
         5: 948,
    },

    8: {
         4: 225,
         3: 588,
         5: 905,
    },

    9: {
         4: 158,
         3: 559,
         5: 763,
    },

    10: {
         4: 227,
         3: 598,
         5: 912,
    },

    11: {
         4: 216,
         3: 472,
         5: 864,
    },

    12: {
         4: 235,
         3: 613,
         5: 861,
    }
}


