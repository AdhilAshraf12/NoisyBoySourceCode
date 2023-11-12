import pyaudio
import numpy as np

class AdaptiveNoiseCancellation:
    def __init__(self, adaptation_step=0.01):
        self.adaptation_step = adaptation_step
        self.reference_signal = None

    def update_reference(self, mic_signal):
        if self.reference_signal is None:
            self.reference_signal = mic_signal.copy()
        else:
            self.reference_signal += self.adaptation_step * (mic_signal - self.reference_signal)

    def cancel_noise(self, mic_signal):
        if self.reference_signal is None:
            return mic_signal  # No reference signal yet, return original signal

        error_signal = mic_signal - self.reference_signal
        canceled_signal = -error_signal  # Basic inversion

        return canceled_signal

def main():
    p = pyaudio.PyAudio()

    # Open a stream for microphone input
    mic_stream = p.open(format=pyaudio.paFloat32,
                        channels=1,
                        rate=44100,
                        input=True,
                        frames_per_buffer=1024)

    # Open a stream for speaker output
    speaker_stream = p.open(format=pyaudio.paFloat32,
                            channels=1,
                            rate=44100,
                            output=True,
                            frames_per_buffer=1024)

    noise_cancellation_system = AdaptiveNoiseCancellation()

    try:
        print("Press Ctrl+C to exit.")
        while True:
            # Read input from the microphone
            mic_signal = np.frombuffer(mic_stream.read(1024), dtype=np.float32)

            # Update the reference signal for adaptive filtering
            noise_cancellation_system.update_reference(mic_signal)

            # Apply noise cancellation
            canceled_signal = noise_cancellation_system.cancel_noise(mic_signal)

            # Play the canceled signal through the speaker
            speaker_stream.write(canceled_signal.tobytes())
    except KeyboardInterrupt:
        pass
    finally:
        mic_stream.stop_stream()
        mic_stream.close()
        speaker_stream.stop_stream()
        speaker_stream.close()
        p.terminate()

if __name__ == "__main__":
    main()
