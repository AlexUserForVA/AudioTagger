import cv2
import json
import numpy as np
import matplotlib.pyplot as plt

from pydoc import locate
from flask import Flask, Response, request

from server.producer.signal_provider import SignalProvider
from server.consumer.visualizers.spectrogram.madmom_spectrogram_provider import MadmomSpectrogramProvider
from server.audio_tagger_model import AudioTaggerModel
from server.config.load_config import loadPredictors, loadSources

############### construct audio tagger model ####################

### load configs ###
predList = loadPredictors()
sourceList = loadSources()

# create signal provider
signalProvider = SignalProvider()

specsProvider = MadmomSpectrogramProvider()

predProviderClass = locate('server.consumer.predictors.{}'.format(predList[0]['predictorClassPath']))
predProvider = predProviderClass()

model = AudioTaggerModel(signalProvider, specsProvider, predProvider, predList, sourceList)


###### startup web server to provide audio tagger REST API ######
app = Flask(__name__)

@app.route('/live_spec', methods=['GET'])
def live_spec():
    content = model.getLiveSpectrogram()
    content = convertSpecToJPG(content)
    return Response(content,
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/live_spec_browser', methods=['GET'])
def live_spec_browser():
    content = model.getLiveSpectrogram()
    content = convertSpecToJPG(content)
    content = (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + content + b'\r\n\r\n')
    return Response(content,
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/live_pred', methods=['GET'])
def live_pred():
    content = model.getLivePrediction()
    response = app.response_class(
        response=json.dumps(content),
        status=200,
        mimetype='application/json'
    )
    return response

@app.route('/pred_list', methods=['GET'])
def pred_list():
    content = [ {'id' : elem['id'], 'displayname': elem['displayname'], 'classes': elem['classes'], 'description': elem['description']} for elem in model.getPredList()]
    response = app.response_class(
        response=json.dumps(content),
        status=200,
        mimetype='application/json'
    )
    return response

@app.route('/source_list', methods=['GET'])
def source_list():
    content = [{'id' : elem['id'],'displayname': elem['displayname']} for elem in model.getSourceList()]
    response = app.response_class(
        response=json.dumps(content),
        status=200,
        mimetype='application/json'
    )
    return response

@app.route('/settings', methods=['POST'])
def add_message():
    print(request.json)
    content = request.json
    model.refreshAudioTagger(content)
    return 'OK'

###### Helper functions ######

def convertSpecToJPG(spec):
    spec = spec / 3.0
    resz_spec = 2
    spec = cv2.resize(spec, (spec.shape[1] * resz_spec, spec.shape[0] * resz_spec))
    spec = plt.cm.viridis(spec)[:, :, 0:3]
    spec_bgr = (spec * 255).astype(np.uint8)
    if spec_bgr.shape[1] < 512:
        p = (512 - spec_bgr.shape[1]) // 2
        spec_bgr = np.pad(spec_bgr, ((0, 0), (p, p), (0, 0)), mode="constant")
    spec_bgr = cv2.flip(spec_bgr, 0)
    _, curImage = cv2.imencode('.jpg', spec_bgr)
    return curImage.tobytes()

if __name__ == '__main__':
    app.run(host='127.0.0.1', debug=False)


