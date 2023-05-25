import argparse

from pathlib import Path
import numpy as np
from tqdm import tqdm

import torch

from acoustic.model import AcousticModel
from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present


def repeat_expand_2d(content: torch.Tensor, target_len: int) -> torch.Tensor:
    # content : [h, t]
    src_len = content.shape[-1]
    if target_len < src_len:
        return content[:, :target_len]
    else:
        return torch.nn.functional.interpolate(
            content.unsqueeze(0), size=target_len, mode="nearest"
        ).squeeze(0)


def generate(args):
    print("Loading acoustic model checkpoint")
    #acoustic = torch.hub.load("bshall/acoustic-model:main", f"hubert_{args.model}").cuda()
    acoustic = AcousticModel().cuda().eval()
    checkpoint = torch.load("exp/rue/model-2000.pt", map_location="cuda")["acoustic-model"]
    consume_prefix_in_state_dict_if_present(checkpoint, "module.")
    acoustic.load_state_dict(checkpoint)

    print(f"Generating from {args.in_dir} -> {args.out_dir}")
    for path in tqdm(list(args.in_dir.rglob("*.npy"))):
        units = np.load(path)
        units_dtype = torch.long if args.model == "discrete" else torch.float
        units = torch.tensor(units, dtype=units_dtype).cuda()
        units = units.T
        print(units.shape)
        target_len = units.shape[1]*3
        units = units.squeeze(0)
        units = repeat_expand_2d(units, target_len).T.unsqueeze(0)

        with torch.inference_mode():
            mel_ = acoustic.generate(units)
            mel_ = mel_.transpose(1, 2)

        out_path = args.out_dir / path.relative_to(args.in_dir)
        out_path.parent.mkdir(exist_ok=True, parents=True)
        np.save(out_path.with_suffix(".npy"), mel_.squeeze().cpu().numpy())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate spectrograms from input speech units (discrete or soft)."
    )
    parser.add_argument(
        "model",
        help="available models (HuBERT-Soft or HuBERT-Discrete)",
        choices=["soft", "discrete"],
    )
    parser.add_argument(
        "in_dir",
        metavar="in-dir",
        help="path to the dataset directory.",
        type=Path,
    )
    parser.add_argument(
        "out_dir",
        metavar="out-dir",
        help="path to the output directory.",
        type=Path,
    )
    args = parser.parse_args()
    generate(args)

