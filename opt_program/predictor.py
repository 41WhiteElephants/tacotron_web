# This is the file that implements a flask server to do inferences. It's the file that you will modify to
# implement the scoring for your own algorithm.

import os
import boto3
from s3fs.core import S3FileSystem
from sagemaker import get_execution_role
import sys
sys.path.append('waveglow/')
import flask

import numpy as np
import torch

from hparams import create_hparams
from train import load_model
from text import text_to_sequence
from denoiser import Denoiser
from pydub import AudioSegment
from datetime import date
import IPython.display as ipd

prefix = '/opt/ml/'
model_path = os.path.join(prefix, 'model')


def predict(sequence):
    hparams = create_hparams()
    hparams.sampling_rate = 22050
    my_bucket = "aws-linux-academy-2k10-ml-sagemaker"
    my_file = "checkpoint_32224"
    s3 = S3FileSystem(anon=False)
    # import pdb;pdb.set_trace()
    model = load_model(hparams)
    model.load_state_dict(
        torch.load(s3.open('{}/{}'.format(my_bucket, my_file), mode='rb'))['state_dict'])
    _ = model.eval().half()

    waveglow_path = 'waveglow_256channels_ljs_v3.pt'
    waveglow = torch.load(s3.open('{}/{}'.format(my_bucket, waveglow_path), mode='rb'))['model']
    waveglow.cuda().eval().half()
    for layer in waveglow.convinv:
        layer.float()
    denoiser = Denoiser(waveglow)
    mel_outputs, mel_outputs_postnet, _, alignments = model.inference(sequence)
    with torch.no_grad():
        audio = waveglow.infer(mel_outputs_postnet, sigma=0.666)
    audio_denoised = denoiser(audio, strength=0.01)[:, 0]
    return audio_denoised


def preprocess(text):
    sequence = np.array(text_to_sequence(text, ['basic_cleaners']))[None, :]
    return torch.autograd.Variable(
        torch.from_numpy(sequence)).cuda().long()


def write_to_s3(filename, bucket, key):
    with open(filename, 'rb') as f: # Read in binary mode
        return boto3.Session().resource('s3').Bucket(bucket).Object(key).upload_fileobj(f)


# The flask app for serving predictions
app = flask.Flask(__name__)


@app.route('/invocations', methods=['POST'])
def inference():
    """Do an inference on a single batch of data. In this sample server, we take data
    as txt, generate speech & put it on s3
    """
    try:
        data = None
        if flask.request.content_type == 'application/json':
            data = flask.request.get_json()
        else:
            return flask.Response(response='This predictor only supports json data',
                                  status=415, mimetype='text/plain')

        # Preprocess
        print("Preprocessing text!")
        sequence = preprocess(data["text"])

        # Do the prediction
        print("Predict - getting audio!")
        audio = predict(sequence)
        print("Denoising audio!")
        audio_denoised = denoiser(audio, strength=0.01)[:, 0]

        audio_file = ipd.Audio(audio_denoised.cpu().numpy(), rate=22050)
        filename = f"{date.today()}.wav"
        audio_file = AudioSegment(audio_file.data, frame_rate=22050, sample_width=2,
                                  channels=1)
        print("Saving audio file!")
        audio_file.export(filename, format="wav")
        my_bucket = "aws-linux-academy-2k10-ml-sagemaker"
        print("Writing file to s3")
        write_to_s3(filename, my_bucket, filename)
        return flask.Response(response="Audio file saved on S3 bucket", status=200,
                              mimetype='text/plain')
    except:
        import traceback

        return flask.Response(response=f"{traceback.format_exc()}", status=400,
                              mimetype='text/plain')




@app.route('/ping', methods=['GET'])
def ping():
    return flask.Response(response='Pong!', status=200, mimetype='application/json')

