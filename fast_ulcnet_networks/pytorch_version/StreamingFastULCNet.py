import copy

import torch
import torch.nn.functional as F
import yaml

from FastULCNet import FastULCNet


class StreamingFastULCNet(FastULCNet):
    """
    Frame-by-frame version of Fast-ULCNet.

    It processes a single STFT frame per call and exposes the whole temporal
    memory of the network as inputs and outputs, so the module itself is
    stateless (handy for ONNX export / real-time inference).

    The layers and their names are the ones of FastULCNet, so the weights of a
    trained FastULCNet can be loaded directly with load_state_dict.

    Note on the states: every convolution of Fast-ULCNet has a (1, k) kernel
    on [B, C, T, F] tensors, i.e. it only spans the frequency axis, and the
    bidirectional frequency RNN runs along frequency inside a frame. None of
    them carries memory from one frame to the next. The only temporal memory
    of the network is the hidden state of the two sub-band RNNs.

    The STFT / iSTFT are expected to be done outside of the model.
    """

    def __init__(self, config: dict):
        config = copy.deepcopy(config)
        config["model_parameters"]["temporal_input"] = False
        super().__init__(config)

    def init_states(self, batch_size=1, device=None, dtype=torch.float32):
        """
        Returns the zero states to use for the first frame, each of shape
        [num_layers, B, sub_band_rnn_units].
        """
        return tuple(
            torch.zeros(
                rnn.num_layers, batch_size, rnn.hidden_size, device=device, dtype=dtype
            )
            for rnn in (self.sub_band_rnn1, self.sub_band_rnn2)
        )

    def forward(self, x, h1, h2):
        """
        x:  [B, F, 2] real and imaginary parts of one noisy STFT frame
        h1: [num_layers, B, sub_band_rnn_units] state of sub_band_rnn1
        h2: [num_layers, B, sub_band_rnn_units] state of sub_band_rnn2

        Returns:
          estimated_speech: [B, F, 2] real and imaginary parts of the
                            enhanced STFT frame
          h1, h2:           updated states, to feed back at the next frame
        """
        # 1. Preprocessing, on a [B, 1, F] frame
        x = x.unsqueeze(1)
        c = self.compression_factor
        real = torch.sign(x[..., 0]) * torch.pow(torch.abs(x[..., 0]), c)
        imag = torch.sign(x[..., 1]) * torch.pow(torch.abs(x[..., 1]), c)
        mag = torch.sqrt(real**2 + imag**2)
        phase = torch.atan2(imag, real)

        # 2. Reorientation: [B, 1, F] -> [B, n_bands, 1, window_size]
        features = self.reorientation(mag).permute(0, 2, 1, 3)

        # 3. Conv Block with MaxPools
        x = self.conv_block(features)

        # 4. Frequency RNN (runs along frequency: no state to carry over)
        B, C, _, F_red = x.shape
        x_rnn = x.permute(0, 2, 3, 1).reshape(B, F_red, C)
        frnn_out, _ = self.freq_rnn(x_rnn)
        frnn_out = frnn_out.view(B, 1, F_red, -1).permute(0, 3, 1, 2)

        # 5. Temporal Sub-band RNNs, one time step
        x = F.relu(self.pointwise_conv(frnn_out))  # [B, 64, 1, F_red]
        x = x.permute(0, 2, 3, 1).reshape(B, 1, -1)  # Flatten F and C

        sub1, sub2 = torch.chunk(x, 2, dim=-1)
        r1, h1 = self.sub_band_rnn1(sub1, h1)
        r2, h2 = self.sub_band_rnn2(sub2, h2)
        concatenated = torch.cat([r1, r2], dim=-1)

        # 6. Mask Computation
        mask = F.relu(self.fc1(concatenated))
        mask = F.relu(self.fc2(mask))

        # 7. Intermediate Features for Stage 2
        inter_feat = self.intermediate_feature_computation(mask, phase)

        # 8. CNN block for final mask
        cnn_out = self.cnn_block(inter_feat)
        c_mask = F.relu(self.complex_mask_conv(cnn_out))

        # 9. CRM and decompression
        m_real, m_imag = c_mask[:, 0, :, :], c_mask[:, 1, :, :]
        est_speech_comp = self.crm_layer(real, imag, m_real, m_imag)
        estimated_speech = self.power_law_decompression(est_speech_comp)

        return torch.view_as_real(estimated_speech)[:, 0], h1, h2


if __name__ == "__main__":
    # Check that frame-by-frame processing matches the offline model
    with open("fast_ulcnet_networks/config.yml", "r") as f:
        config = yaml.load(f, yaml.FullLoader)
    config["model_parameters"]["temporal_input"] = False

    torch.manual_seed(0)
    offline = FastULCNet(config).eval()
    streaming = StreamingFastULCNet(config).eval()
    streaming.load_state_dict(offline.state_dict())

    n_frames = 20
    freq_dim = config["data_parameters"]["n_fft"] // 2 + 1
    B = 8
    noisy = torch.randn(B, n_frames, freq_dim, dtype=torch.complex64)
    output_shape = None
    with torch.no_grad():
        expected = offline(noisy)
        h1, h2 = streaming.init_states(batch_size=noisy.shape[0])
        frames = []
        for t in range(n_frames):
            frame, h1, h2 = streaming(torch.view_as_real(noisy[:, t]), h1, h2)
            output_shape = frame.shape
            frames.append(torch.view_as_complex(frame))
        output = torch.stack(frames, dim=1)

    print(f"Single frame shape: {torch.view_as_real(noisy[:, 0]).shape}")
    print(f"Single output frame shape: {output_shape}")
    print(f"Input shape: {noisy.shape}")
    print(f"Output shape: {output.shape}")
    print("States:", tuple(h1.shape), tuple(h2.shape))
    print(
        "Max abs difference streaming vs offline:",
        (output - expected).abs().max().item(),
    )
