from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from datasets import load_dataset
import torch
import soundfile as sf
from datasets import load_dataset

processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")

app = Flask(__name__)
cors = CORS(app)
app.config["CORS_HEADERS"] = "Content-Type"


@app.route("/respond/tesxt", methods=["POST"])
@cross_origin()
def respond():

    test = request.json["text"]
    inputs = processor(text=test, return_tensors="pt")

    # load xvector containing speaker's voice characteristics from a dataset
    embeddings_dataset = load_dataset(
        "Matthijs/cmu-arctic-xvectors", split="validation")
    speaker_embeddings = torch.tensor(
        embeddings_dataset[7306]["xvector"]).unsqueeze(0)

    speech = model.generate_speech(
        inputs["input_ids"], speaker_embeddings, vocoder=vocoder)

    sf.write("speech4.wav", speech.numpy(), samplerate=16000)

    return jsonify({"success": True})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
