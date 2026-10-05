from FastULCNet import FastULCNet, STFTLayer, ISTFT
from StreamingFastULCNet import StreamingFastULCNet
from argparse import ArgumentParser, BooleanOptionalAction
import yaml
import torch
import onnx
import onnxruntime as ort
import libsegmenter


def annotate_model(model_proto: onnx.ModelProto, prefix: str, annot: str):
    for node in model_proto.graph.node:
        # Assign a layer annotation based on your own logic
        if node.name.startswith(prefix):
            entry = next(
                (prop for prop in node.metadata_props if prop.key == "layer_ann"), None
            )
            if entry is None:
                entry = node.metadata_props.add()
                entry.key = "layer_ann"
            entry.value = annot  # your annotation string

    return model_proto


class AudioToModel(torch.nn.Module):

    def __init__(
        self,
        n_fft: int = 512,
        hop_size: int = 256,
        win_size: int = 512,
        window: torch.Tensor = None,
    ):
        super().__init__()
        self.n_fft = n_fft
        self.hop_size = hop_size
        self.win_size = win_size
        self.register_buffer("window", window)

        self.stft = STFTLayer(self.n_fft, self.hop_size, self.win_size, self.window)

    def forward(self, x: torch.Tensor):
        """Makes temporal input into streamable model input

        Args:
            x (torch.Tensor): [B, T] audio frame

        Returns:
            _type_: [B, T, F, 2] expected streaming model input format
        """
        stft_data = self.stft(x)
        return torch.view_as_real(stft_data)


class ModelOutToAudio(torch.nn.Module):
    def __init__(self, n_fft, hop_size, win_size, window=None):
        super().__init__()
        self.n_fft = n_fft
        self.hop_size = hop_size
        self.win_size = win_size

        self.istft = ISTFT(n_fft, hop_size, win_size, window)

    def forward(self, x: torch.Tensor):
        """From streaming model output to audio domain

        Args:
            x (torch.Tensor): [B, F, 2] Streaming model output tensor

        Returns:
            _type_: [B, T]
        """
        return self.istft(x)


class StreamNFrames(torch.nn.Module):
    def __init__(self, streaming_fulcnet):
        super().__init__()
        self.streaming_fulcnet = streaming_fulcnet
        window = torch.from_numpy(
            libsegmenter.WindowSelector(
                "hann75", "wola", streaming_fulcnet.win_size
            ).analysis_window,
        ).float()
        self.preprocess, self.postprocess = (
            AudioToModel(
                self.streaming_fulcnet.n_fft,
                self.streaming_fulcnet.hop_size,
                self.streaming_fulcnet.win_size,
                window,
            ),
            ModelOutToAudio(
                self.streaming_fulcnet.n_fft,
                self.streaming_fulcnet.hop_size,
                self.streaming_fulcnet.win_size,
                window,
            ),
        )

    def forward(self, x: torch.Tensor, h1: torch.Tensor, h2: torch.Tensor):
        noisy = self.preprocess(x)  # [B, T, F, 2]
        audio_frames = noisy.shape[1]
        frames = []
        for frame in range(audio_frames):
            out, h1, h2 = self.streaming_fulcnet(noisy[:, frame], h1, h2)
            frames.append(out)  # [B, F, 2]
        output = torch.stack(frames, dim=2)  # [B, F, T, 2]
        return self.postprocess(output), h1, h2


parser = ArgumentParser()
parser.add_argument("config", type=str)
parser.add_argument("checkpoint", type=str)
parser.add_argument("--input_shape", nargs="+", type=int, default=[1, 257])
parser.add_argument("--output", type=str, default="streaming_fast_ulcnet.onnx")
parser.add_argument("--wrapped", action=BooleanOptionalAction, default=False)
parser.add_argument("--offline", action=BooleanOptionalAction, default=False)
args = parser.parse_args()

checkpoint = None
try:
    checkpoint = torch.load(args.checkpoint)
except RuntimeError:
    checkpoint = torch.load(args.checkpoint, map_location=torch.device("cpu"))
except FileNotFoundError:
    pass
with open(args.config, "r") as f:
    config = yaml.load(f, Loader=yaml.SafeLoader)

temporal_input = config["model_parameters"]["temporal_input"]
offline = FastULCNet(config)
offline.load_state_dict(checkpoint)
offline.eval()
streaming = StreamingFastULCNet(config).eval()
streaming.load_state_dict(offline.state_dict())

if args.wrapped:
    model = StreamNFrames(streaming)
    h1, h2 = model.streaming_fulcnet.init_states(batch_size=args.input_shape[0])

    dummy_input = torch.randn(args.input_shape, dtype=torch.float32)
    example_inputs = dummy_input, h1, h2
else:
    model = streaming
    dummy_input = torch.randn(
        args.input_shape[0], model.n_fft // 2 + 1, 2, dtype=torch.float32
    )
    h1, h2 = model.init_states(batch_size=args.input_shape[0])
    example_inputs = dummy_input, h1, h2

# Export streaming model
onnx_program = torch.onnx.export(model, example_inputs, dynamo=True)

onnx.checker.check_model(onnx_program.model_proto)
onnx_model_path = args.output
onnx.save(onnx_program.model_proto, onnx_model_path)
print(f"Saved to {onnx_model_path}")

ort_sess = ort.InferenceSession(onnx_model_path)
