// variables declaration
let mobilenet;
let model;
const webcam = new Webcam(document.getElementById('wc'));
const dataset = new Dataset();
var class1Samples=0, class2Samples=0, class3Samples=0, class4Samples=0, class5Samples=0;
let isPredicting = false;

// loading the architecture of MobileNet
async function loadMobilenet() {
    // downloading the mobileNet topology and weights
  const mobilenet = await tf.loadLayersModel('https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_0.25_224/model.json');
    // picking our layer from the topology over which our on-the top training will take place
  const layer = mobilenet.getLayer('conv_pw_13_relu');
  return tf.model({inputs: mobilenet.inputs, outputs: layer.output});
}

// training on the top of the topology of the MobileNet
async function train() {
  dataset.ys = null;
  dataset.encodeLabels(5);
  model = tf.sequential({
    layers: [
        // flattening the layer over which we training is to be done
      tf.layers.flatten({inputShape: mobilenet.outputs[0].shape.slice(1)}),
        // 120 neurons are added with activation function "ReLU"
      tf.layers.dense({ units: 120, activation: 'relu'}),
        // output is governed by the multi-class classification function "SIGMOID"
      tf.layers.dense({ units: 5, activation: 'sigmoid'})
    ]
  });
    // used optimiser "Stochastic Gradient Descent" with learning-rate, alpha (α) = 0.002
  const optimizer = tf.train.sgd(0.002);
    // loss-function: measuring distances between target labels and predicted labels
    // optimizer: reduces loss and updates weights and biases regularly
  model.compile({optimizer: optimizer, loss: 'binaryCrossentropy'});
  let loss = 0;
    // training the model with above stated loss function and optimizer with epochs(no of times trained)
  model.fit(dataset.xs, dataset.ys, {
    epochs: 30,
    callbacks: {
      onBatchEnd: async (batch, logs) => {
        loss = logs.loss.toFixed(5);
        console.log('LOSS: ' + loss);
        }
      }
   });
}

// taking samples of multiple people
function handleButton(elem){
	switch(elem.id){
		case "0":
			class1Samples++;
			document.getElementById("class1samples").innerText = "Person 1 Samples:" + class1Samples;
			break;
		case "1":
			class2Samples++;
			document.getElementById("class2samples").innerText = "Person 2 Samples:" + class2Samples;
			break;
        case "2":
			class3Samples++;
			document.getElementById("class3samples").innerText = "Person 3 Samples:" + class3Samples;
			break;
        case "3":
			class4Samples++;
			document.getElementById("class4samples").innerText = "Person 4 Samples:" + class4Samples;
			break;
        case "4":
			class5Samples++;
			document.getElementById("class5samples").innerText = "Person 5 Samples:" + class5Samples;
			break;
	}
	label = parseInt(elem.id);
	const img = webcam.capture();
	dataset.addExample(mobilenet.predict(img), label);
}


// predicting the subject in the frame of camera
async function predict() {
  while (isPredicting) {
    const predictedClass = tf.tidy(() => {
      const img = webcam.capture();
      const activation = mobilenet.predict(img);
      const predictions = model.predict(activation);
      // returning the class value with maximum probability
      return predictions.as1D().argMax();
    });
    const classId = (await predictedClass.data())[0];
    var predictionText = "";
      if(classId==0 && class1Samples>0){
			predictionText = "Yes it's person 1  (❁´◡`❁), Come in!";}
      else if(classId==1 && class2Samples>0){
			predictionText = "Yes it's person 2  (❁´◡`❁), Come in!";}
      else if(classId==2 && class3Samples>0){
            predictionText = "Yes it's person 3  (❁´◡`❁), Come in!";}
      else if(classId==3 && class4Samples>0){
            predictionText = "Yes it's person 4  (❁´◡`❁), Come in!";}
      else if(classId==4 && class5Samples>0){
            predictionText = "Yes it's person 5  (❁´◡`❁), Come in!";}
      else{
            predictionText = "I'm sorry, I see no one ¯\_(ツ)_/¯";}
	
      predictionText.fontcolor("green");
      predictionText.bold();
	document.getElementById("prediction").innerText = predictionText;
			
    
    predictedClass.dispose();
    await tf.nextFrame();
  }
}

// to start training 
function doTraining(){
	train();
}

// to start prediction
function startPredicting(){
	isPredicting = true;
	predict();
}

// to stop predicting
function stopPredicting(){
	isPredicting = false;
	predict();
}

// for downloading the model to use the learnt weights of the samples in other system
function saveModel(){
    model.save('downloads://my_model');
}

// initial function when the program starts
async function init(){
	await webcam.setup();
	mobilenet = await loadMobilenet();
	tf.tidy(() => mobilenet.predict(webcam.capture()));		
}

init();
