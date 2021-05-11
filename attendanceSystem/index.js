let mobilenet;
let model;
const webcam = new Webcam(document.getElementById('wc'));
const dataset = new Dataset();
var class1Samples=0, class2Samples=0;
let isPredicting = false;

async function loadMobilenet() {
  const mobilenet = await tf.loadLayersModel('https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_0.25_224/model.json');
  const layer = mobilenet.getLayer('conv_pw_13_relu');
  return tf.model({inputs: mobilenet.inputs, outputs: layer.output});
}

async function train() {
  dataset.ys = null;
  dataset.encodeLabels(2);
  model = tf.sequential({
    layers: [
      tf.layers.flatten({inputShape: mobilenet.outputs[0].shape.slice(1)}),
      tf.layers.dense({ units: 100, activation: 'relu'}),
      tf.layers.dense({ units: 2, activation: 'sigmoid'})
    ]
  });
  const optimizer = tf.train.sgd(0.0001);
  model.compile({optimizer: optimizer, loss: 'binaryCrossentropy'});
  let loss = 0;
  model.fit(dataset.xs, dataset.ys, {
    epochs: 10,
    callbacks: {
      onBatchEnd: async (batch, logs) => {
        loss = logs.loss.toFixed(5);
        console.log('LOSS: ' + loss);
        }
      }
   });
}


function handleButton(elem){
	switch(elem.id){
		case "0":
			class1Samples++;
			document.getElementById("class1samples").innerText = "Your Samples:" + class1Samples;
			break;
		case "1":
			class2Samples++;
			document.getElementById("class2samples").innerText = "Background Samples:" + class2Samples;
			break;
	}
	label = parseInt(elem.id);
	const img = webcam.capture();
	dataset.addExample(mobilenet.predict(img), label);
}

async function predict() {
  while (isPredicting) {
    const predictedClass = tf.tidy(() => {
      const img = webcam.capture();
      const activation = mobilenet.predict(img);
      const predictions = model.predict(activation);
      return predictions.as1D().argMax();
    });
    const classId = (await predictedClass.data())[0];
    var predictionText = "";
    switch(classId){
		case 0:
			predictionText = "Yes it's YOU  (❁´◡`❁), Come in!";
			break;
		case 1:
			predictionText = "Oops! I don't see you (⊙_⊙;) ";
			break;
	}
      predictionText.fontcolor("green");
      predictionText.bold();
	document.getElementById("prediction").innerText = predictionText;
			
    
    predictedClass.dispose();
    await tf.nextFrame();
  }
}


function doTraining(){
	train();
}

function startPredicting(){
	isPredicting = true;
	predict();
}

function stopPredicting(){
	isPredicting = false;
	predict();
}
function saveModel(){
    model.save('downloads://my_model');
}
async function init(){
	await webcam.setup();
	mobilenet = await loadMobilenet();
	tf.tidy(() => mobilenet.predict(webcam.capture()));		
}

init();
