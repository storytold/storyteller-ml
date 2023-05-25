#@title Load the models 
#@markdown leave hifigan blank to use defaults
import sys
import torch, torchaudio
from typing import Dict, Any
from collections import OrderedDict
from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present

# Inputs 
check_point_path = './acoustic-model/exp/glados/model-35000.pt' #@param {type:"string"}
hifigan_checkpoint_path = "./hifigan/exp/vctk/model-156000.pt" #@param {type: "string"}
file_name = "/mnt/8terra/tts/voice-conversion-sources/ieatmeat.wav" #@param {type: "string"}

def modify_model(state_dict:Dict[str,Any]):
    new_state_dict = OrderedDict()
    for k in state_dict['acoustic-model'].keys():
        value = state_dict['acoustic-model'][k]
        name = k[7:] # remove `module.` because they used ddp training to save
        new_state_dict[name] = value
    return new_state_dict

def repeat_expand_2d(content: torch.Tensor, target_len: int) -> torch.Tensor:
    # content : [h, t]
    src_len = content.shape[1]
    if target_len < src_len:
        return content[:, target_len]
    else:
        return torch.nn.functional.interpolate(
            content.unsqueeze(0), size=target_len, mode="nearest"
        )


sys.path.append("./acoustic-model/")
from acoustic import AcousticModel

hubert = torch.hub.load("bshall/hubert:main", "hubert_soft").cuda()


# Load the acoustic model (either hubert_soft or hubert_discrete) 
acoustic_dict = torch.load(check_point_path) # this used ddp
acoustic = AcousticModel().cuda()
acoustic.eval()

import importlib.util
import sys

spec = importlib.util.spec_from_file_location("hifigan.utils", "./hifigan/hifigan/utils.py")
u = importlib.util.module_from_spec(spec)
sys.modules["hifigan.utils"] = u
spec.loader.exec_module(u)

spec = importlib.util.spec_from_file_location("hifigan.generator", "./hifigan/hifigan/generator.py")
g = importlib.util.module_from_spec(spec)
sys.modules["hifigan.generator"] = g
spec.loader.exec_module(g)

new_state_dict = modify_model(state_dict=acoustic_dict)
acoustic.load_state_dict(new_state_dict)
if hifigan_checkpoint_path:
  checkpoint = torch.load(hifigan_checkpoint_path)["generator"]["model"]
  hifigan = g.HifiganGenerator()
  consume_prefix_in_state_dict_if_present(checkpoint, "module.")
  hifigan.load_state_dict(checkpoint)
  hifigan.remove_weight_norm()
  hifigan.eval()
else:
  hifigan = torch.hub.load("bshall/hifigan:main", "hifigan_hubert_soft")
hifigan.cuda()

#@title This code selects the source file
#import IPython.display as display
sys.path.insert(0, './acoustic-model/acoustic/')
from model import AcousticModel
import soundfile as sf
#from google.colab import files
from time import time
sys.path.append("./hifigan/")
from hifigan.generator import HifiganGenerator
from hifigan.generator import consume_prefix_in_state_dict_if_present
#uploaded = files.upload()

#for name in uploaded.keys():
    #file_name = name    
source, sr = torchaudio.load(f"{file_name}")
source = torchaudio.functional.resample(source, sr, 16000)
source = torch.mean(source, dim=0).unsqueeze(0)
source = source.unsqueeze(0).cuda()
#display.Audio(source.squeeze().cpu(), rate=16000)

#@title Run the inference and display the result
# Convert to the target speaker
with torch.inference_mode():
    # Extract speech units
    start = time()
    units = hubert.units(source)
    print(units.shape)
    #units = repeat_expand_2d(units, target_len)
    units = torch.nn.functional.interpolate(
            units.transpose(1,2), size=units.shape[1]*3, mode="nearest"
    ).transpose(1,2)

    print(units.shape)
    #units = units.unsqueeze(0)
    #units = units.transpose(1,2)

    end = time()
    # Generate target spectrogram
    print(f"time for feature embedding:{end-start}")

    start = time()
    mel = acoustic.generate(units).transpose(1,2)
    end = time()
    print(f"time for inference:{end-start}")
    # Generate audio waveform
    start = time()
    target = hifigan(mel)
    end = time()
    print(f"time for wavform {end-start}")
    print(target.squeeze().cpu().unsqueeze(0).shape)

torchaudio.save("output.wav", target.squeeze().cpu().unsqueeze(0), 48000)
