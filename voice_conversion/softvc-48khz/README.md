# 48KHz SoftVC

Herein lies 48KHz softvc, a failed experiment to produce a high samplerate version of https://github.com/bshall/soft-vc
The goal was to leverage the strong character quality that softvc has, ie it's ability to work well with weird voices
such as Evanescence, GladOS, Darth Vader, and Yoda, while providing a higher samplerate than the 16KHz of the original
version. We used the same technique as so-vits-svc, to upscale the 16KHz HuBERT "speech units" to 48KHz equivalents,
and trained on 48KHz and 44.1 KHz source data. Unfortunately, the results have a robotic, metallic sound to them,
and sound basically the same as if it was 16KHz (with weird robotic noises added for flavour).


Additionally, warm-starting didn't seem to work, maybe there's some trick to remove the speaker embedding of the base
model, but it's not part of the code provided by bshall.


To hear the results for yourself, visit https://drive.google.com/drive/folders/1byrr6R70qr4kUVj39KXi9NCXmbJVJgNU?usp=sharing
