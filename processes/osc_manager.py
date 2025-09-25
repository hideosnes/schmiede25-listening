import time
import threading
import queue
import socket

# change these settings ig needed
OSC_HOST = "127.0.0.1"
OSC_PORT = 9000
DEFAULT_UPDATE_INTERVAL = 1.0 # seconds

class SimpleOSCServer:
    def __init__(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.running = False
        self.latest_predictions = [("---", 0.0), ("---", 0.0), ("---", 0.0)]
        self.update_interval = DEFAULT_UPDATE_INTERVAL
        self.message_count = 0
        self.last_send_time = "Never"

    def create_osc_message(self, address, values):
        message = address.encode('utf-8')
        message += b'\x00' * (4 - len(message) % 4)

        types = ","
        for val in values:
            if isinstance(val, str):
                types += "s"
            elif isinstance(val, float):
                types += "f"
            elif isinstance(val, int):
                types += "i"

        message += types.encode('utf-8')
        message += b'\x00' * (4 - len(types) % 4)

        for val in values:
            if isinstance(val, str):
                val_bytes = val.encode('utf-8')
                message += val_bytes
                message += b'\x00' * (4 - len(val_bytes) % 4)
                
            elif isinstance(val, (int, float)):
                import struct
                message += struct.pack('>f', float(val))

        return message

    def send_predictions(self):
        try:
            for i, (label, score) in enumerate(self.latest_predictions):
                address = f"/instrument/top{i+1}"
                osc_msg = self.create_osc_message(address, [label, score])
                self.sock.sendto(osc_msg, (OSC_HOST, OSC_PORT))

            # send as one (pewpew <3)
            all_labels = [pred[0] for pred in self.latest_predictions]
            all_scores = [pred[1] for pred in self.latest_predictions]

            labels_msg = self.create_osc_message("/instrument/labels", all_labels)
            scores_msg = self.create_osc_message("/instrument/scores", all_scores)

            self.sock.sendto(labels_msg, (OSC_HOST, OSC_PORT))
            self.sock.sendto(scores_msg, (OSC_HOST, OSC_PORT))

            self.message_count += 5
            import datetime
            self.last_send_time = datetime.datetime.now().strftime("%H:%M:%S")
        except Exception as e:
            print(f"OSC send error: {e}")
            self.last_send_time = f"ERROR: {e}"
    
    def update_predictions(self, predictions):
        # having 3 predictions is pretty sexy!
        self.latest_predictions = predictions[:3]
        while len(self.latest_predictions) < 3:
            self.latest_predictions.append(("---", 0.0))

    def spam_loop(self):
        print(f"OSC server blindly blasting to {OSC_HOST}:{OSC_PORT} every {self.update_interval}s")
        print("OSC Addresses:")
        print(" - /instrument/top1, /instrument/top2, /instrument/top3 -> [label, score]")
        print(" - /instrument/labels -> [label1, label2, label3]")
        print(" - /instrument/scores -> [score1, score2, score3]")

        while self.running:
            self.send_predictions()
            time.sleep(self.update_interval)

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self.spam_loop)
        self.thread.daemon = True
        self.thread.start()
        return self.thread

    def stop(self):
        self.running = False
        self.sock.close()

# global OSC instance
_osc_server = None

def start_osc_server(update_interval=DEFAULT_UPDATE_INTERVAL):
    global _osc_server
    if _osc_server is None:
        _osc_server = SimpleOSCServer()
        _osc_server.update_interval = update_interval
        _osc_server.start()
    return _osc_server

def update_osc_predictions(predictions):
    global _osc_server
    if _osc_server:
        _osc_server.update_predictions(predictions)

def stop_osc_server():
    global _osc_server
    if _osc_server:
        _osc_server.stop()
        _osc_server = None

def get_osc_status():
    global _osc_server
    if _osc_server and _osc_server.running:
        return {
            'host': OSC_HOST,
            'port': OSC_PORT,
            'message_count': _osc_server.message_count,
            'last_send': _osc_server.last_send_time,
            'running': True
        }
    return None


if __name__ == "__main__":
    print("Standalone OSC Test Server")
    print("Spamming the shit out of test data for the music-moopsies")

    osc = SimpleOSCServer()
    osc.start()

    try:
        test_predictions = [
            ("guitar", 0.85),
            ("piano", 0.12),
            ("drums", 0.03)
        ]
        osc.update_predictions(test_predictions)

        print("Press Strg+C to stop...")
        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        print("\nStopping OSC Server...")
        osc.stop()