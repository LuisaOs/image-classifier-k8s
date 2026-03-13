from gluoncv.model_zoo import get_model
from mxnet import nd
from mxnet.gluon.data.vision import transforms
from PIL import Image
import io
import flask

app = flask.Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():

    if flask.request.files.get("img"):
        img = Image.open(io.BytesIO(flask.request.files["img"].read()))

        transform_fn = transforms.Compose([
            transforms.Resize(32),
            transforms.CenterCrop(32),
            transforms.ToTensor()
        ])

        img = transform_fn(nd.array(img))

        net = get_model('cifar_resnet20_v1', classes=10, pretrained=True)

        pred = net(img.expand_dims(axis=0))

        class_names = [
        'airplane','automobile','bird','cat','deer',
        'dog','frog','horse','ship','truck'
        ]

        ind = nd.argmax(pred, axis=1).astype('int')

        prediction = ('The input picture is classified as [%s], with probability %.3f.' %
                      (class_names[ind.asscalar()],
                       nd.softmax(pred)[0][ind].asscalar()))

        return prediction

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)