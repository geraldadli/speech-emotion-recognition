class MicrophoneCapture extends AudioWorkletProcessor {
  constructor() {
    super();
    this.buffer = new Float32Array(2048);
    this.used = 0;
    this.active = true;
    this.port.onmessage = ({ data }) => {
      if (data === 'flush') {
        this.active = false;
        this.flush();
        this.port.postMessage({ flushed: true });
      }
    };
  }
  flush() {
    if (!this.used) return;
    const packet = this.buffer.slice(0, this.used);
    this.port.postMessage(packet, [packet.buffer]);
    this.used = 0;
  }
  process(inputs) {
    const channels = inputs[0];
    if (this.active && channels?.length) {
      for (let i = 0; i < channels[0].length; i++) {
        let sample = 0;
        for (const channel of channels) sample += channel[i];
        this.buffer[this.used++] = sample / channels.length;
        if (this.used === this.buffer.length) this.flush();
      }
    }
    return true;
  }
}
registerProcessor('microphone-capture', MicrophoneCapture);
