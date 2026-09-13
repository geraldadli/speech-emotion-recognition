# Whisper Large V3 speech emotion model

Model: openai/whisper-large-v3 @ 06f233fe06e710322aca913c1bc4249a0d71fce1. Frozen full audio encoder with a supervised mlp head.
Eight labels: angry, calm, disgust, fear, happy, neutral, sad, surprise. Test accuracy from this run: 0.7309.
This is an acted-speech course/research prototype, not a validated assessment of a person's
internal state. Read evaluation.json for per-source results and class support. Confidence
is approximate; unknown emotions, noise, or non-speech may still receive confident labels.

Install requirements.txt, selecting the appropriate CPU/CUDA PyTorch wheel for your machine.
Keep all files together. Example:
    import sys
    sys.path.insert(0, 'whisper_ser_deployment')
    from predict import EmotionPredictor
    predictor = EmotionPredictor('whisper_ser_deployment', device='cpu')
    result = predictor.predict('recording.wav')
Or: python whisper_ser_deployment/predict.py --bundle whisper_ser_deployment --audio recording.wav --device cpu
Reuse one predictor instance. No Hugging Face network access is needed after extraction.
The raw input contract is the same decoder/resampler/boundary trim used in training.
CPU uses FP32 and GPU uses FP16 for Whisper; minor numeric differences are possible.

The eight labels preserve calm, which occurs only in RAVDESS. TESS pleasant surprise maps
to surprise. CREMA-D filename labels describe intended expressions, not necessarily audio
listener consensus. Source IDs are excluded from features. Speaker IDs are namespaced;
cross-corpus aliases and pretrained-data overlap were not independently verified.
TESS has only one training and one test speaker and no validation speaker. Scripts can recur
across speakers. This evaluates speakers from known corpora, not deployment microphones.
No test audio was used to fit normalization, head weights, epochs, or temperature.

Whisper/model terms: https://huggingface.co/openai/whisper-large-v3 (Apache-2.0 listing).
RAVDESS: https://zenodo.org/records/1188976
CREMA-D: https://github.com/CheyneyComputerScience/CREMA-D
TESS: https://doi.org/10.5683/SP2/E8H2MF
SAVEE: https://kahlan.eps.surrey.ac.uk/savee/
This bundle grants no additional rights to the source datasets or dependencies; their
individual terms still apply, including for commercial deployment.
