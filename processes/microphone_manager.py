import argparse
from pathlib import Path

NAME = "microphone_manager"
DESCRIPTION = "List and test available microphones / audio input devices"

def add_arguments(parser: argparse.ArgumentParser):
    parser.add_argument("--list", action="store_true", help="List all available microphones")
    parser.add_argument("--test", type=int, default=None, help="Test microphone by device ID (get ID from --list)")
    parser.add_argument("--duration", type=int, default=5, help="Test duration in seconds (default: 5)")

def list_microphones():
    try:
        import pyaudio
    except ImportError:
        raise RuntimeError("Brrrrrrrr, 'pyaudio' is required and should already be installed with python! Go, c&p the error message into a LLM!")

    p = pyaudio.PyAudio()

    print("Available Microphones/Input Devices:")
    print("=" * 50)

    devices = []
    for i in range(p.get_device_count()):
        device_info = p.get_device_info_by_index(i)

        if device_info['maxInputChannels'] > 0:
            is_default = "(DEFAULT)" if i == p.get_default_input_device_info()['index'] else ""

            print(f"ID: {i:2d} - {device_info['name']} {is_default}")
            print(f" - Channels: {device_info['maxInputChannels']}")
            print(f" - Sample Rate: {int(device_info['defaultSampleRate'])} Hz")
            print(f" - API: {p.get_host_api_info_by_index(device_info['hostApi'])['name']}")
            print()

            devices.append({
                'id': i,
                'name': device_info['name'],
                'channels': device_info['maxInputChannels'],
                'sample_rate': int(device_info['defaultSampleRate']),
                'is_default': is_default != ""
            })


    p.terminate()
    return devices

def test_microphone(device_id, duration=5):
    try:
        import pyaudio
        import numpy as np
        import time
    except ImportError:
        raise RuntimeError("Brrrrrrrr, 'pyaudio' is required and should already be installed with python! Go, c&p the error message into a LLM!")

    p = pyaudio.PyAudio()

    try: 
        device_info = p.get_device_info_by_index(device_id)
        print(f"Testing microphone: {device_info['name']}")
        print(f"Recording for {duration} seconds...")
        print(f"Make some noise to test the microphone!")

        sample_rate = int(device_info['defaultSampleRate'])
        chunk_size = 1024

        stream = p.open(
            format=pyaudio.paFloat32,
            channels=1,
            rate=sample_rate,
            input=True,
            input_device_index=device_id,
            frames_per_buffer=chunk_size
        )

        print(f"Volume levels (make noise!):")
        print("=" * 40)

        start_time = time.time()
        max_volume = 0

        while time.time() - start_time < duration:
            try:
                data = stream.read(chunk_size, exception_on_overflow=False)
                audio_data = np.frombuffer(data, dtype=np.float32)

                # RMS volume
                rms = np.sqrt(np.mean(audio_data**2))
                volume_percent = min(rms * 100, 100)
                max_volume = max(max_volume, volume_percent)

                # Visual volume bar
                bar_length = int(volume_percent / 2)
                bar = "█" * bar_length + "░" * (50 - bar_length)
                print(f"\r [{bar}] {volume_percent:5.1f}%", end="", flush=True)

                time.sleep(0.05)

            except Exception as e:
                print(f"\nError reading audio: {e}")
                break

        stream.stop_stream()
        stream.close()

        print(f"\n\n Test completed!")
        print(f"Maximum volume detected: {max_volume:.1f}%")

        if max_volume < 1:
            print("Brrrrrr, check microphone connection or permissions")
        elif max_volume > 50:
            print("Good volume levels detected!")
        else:
            print("Moderate volume detected, but microphone is working.")

    except Exception as e:
        print(f"Unknown error testing microphone {device_id}: {e}")
    finally:
        p.terminate()

def run(args):
    if args.list:
        devices = list_microphones()

        if devices:
            print("\n Usage:")
            print(f" - Test microphone: python app.py {NAME} --test <ID>")
            print(f" - Example: python app.py {NAME} --test 0")

            default_devices = [d for d in devices if d['is_default']]
            if default_devices:
                print(f"\n Recommended: Use device ID {default_devices[0]['id']} (default)")
        else:
            print("No microphones found!")
    elif args.test is not None:
        test_microphone(args.test, args.duration)

    else:
        print("Use --list to see available microphones or --test <ID> to test one")
        print(f"Example: python app.py {NAME} --list")