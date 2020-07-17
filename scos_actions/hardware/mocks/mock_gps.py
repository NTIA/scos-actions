from datetime import datetime

from scos_actions.hardware.gps_iface import GPSInterface


class MockGPS(GPSInterface):
    def get_lat_long(timeout_s=1):
        return 39.995118, -105.261572

    def get_gps_time(self):
        return datetime.now()
