const trainUrl = document.getElementById('trainUrl');
const testUrl = document.getElementById('testUrl');
const backend = document.getElementById('backend');
const log = document.getElementById('log');
const loss = document.getElementById('loss');
const accuracy = document.getElementById('accuracy');
const inferenceResult = document.getElementById('inferenceResult');
const gtTable = document.getElementById('gtTable');
const inferenceTable = document.getElementById('inferenceTable');
const gtChart = document.getElementById('gtChart');
const inferenceChart = document.getElementById('inferenceChart');
const elapsed = document.getElementById('elapsed');
const lossType = document.getElementById('lossType');
const lossFunction = document.getElementById('lossFunction');
const hiddenLayers = document.getElementById('hiddenLayers');

let isTraining = false;
let startTime;
const TRAIN_SET_URL =
  'https://raw.githubusercontent.com/ajaxlab/tfjs-study/main/csv/iris_train.csv';
const TEST_SET_URL =
  'https://raw.githubusercontent.com/ajaxlab/tfjs-study/main/csv/iris_test.csv';

trainUrl.innerText = TRAIN_SET_URL;
trainUrl.href = TRAIN_SET_URL;
testUrl.innerText = TEST_SET_URL;
testUrl.href = TEST_SET_URL;

tf.setBackend('cpu');

function prepareData() {
  return dfd.read_csv(TRAIN_SET_URL).then((df) => {
    df.head(3).print();
    console.log(df.shape);

    const X_train = df.loc({
      columns: ['petal_h', 'petal_w', 'sepal_h', 'sepal_w'],
    });
    X_train.head(3).print();
    console.log(X_train.shape);

    const encoder = new dfd.OneHotEncoder();
    const Y_train = encoder.fit(df['type']);
    Y_train.head(3).print();
    console.log(Y_train.shape);

    return {
      X_train,
      Y_train,
    };
  });
}

function createHiddenLayer(inputs) {
  let nums = parseInt(hiddenLayers.value, 10);
  let hidden;
  while (nums) {
    nums--;
    hidden = tf.layers
      .dense({ units: 8, activation: 'swish' })
      .apply(hidden || inputs);
  }
  return hidden;
}

function createModel() {
  const inputs = tf.input({ shape: [4] });
  const hidden = createHiddenLayer(inputs);
  const outputs = tf.layers
    .dense({ units: 3, activation: 'softmax' })
    .apply(hidden);
  const model = tf.model({ inputs, outputs });

  model.compile({
    optimizer: tf.train.adam(),
    loss: lossType.value, // 'categoricalCrossentropy' || 'meanSquaredError'
    metrics: ['accuracy'],
  });
  tfvis.show.modelSummary({ name: 'Summary', tab: 'Model' }, model);

  const surface = { name: 'Layer Summary', tab: 'Model Inspection' };
  tfvis.show.layer(surface, model.getLayer(undefined, 1));

  lossFunction.innerText = `Loss Function: ${model.loss}`;
  return model;
}

async function train(model, X_train, Y_train) {
  const surface = { name: 'Values Distribution', tab: 'Model Inspection' };
  await tfvis.show.valuesDistribution(surface, X_train.tensor);
  const history = [];
  const EPOCH = parseInt(epoch.value, 10);
  startTime = Date.now();
  return model
    .fit(X_train.tensor, Y_train.tensor, {
      epochs: EPOCH,
      callbacks: {
        onEpochEnd(epoch, logs) {
          log.innerText = `Epoch: ${epoch}/${EPOCH}`;
          loss.innerText = `Loss: ${logs.loss}`;
          accuracy.innerText = `Accuracy: ${logs.acc}`;
          if (epoch % 100 === 0) {
            console.log(epoch, logs);
          }
          history.push(logs);
          tfvis.show.history(
            {
              name: 'Loss',
              tab: 'History',
            },
            history,
            ['loss'],
          );
          tfvis.show.history(
            {
              name: 'accuracy',
              tab: 'History',
            },
            history,
            ['acc'],
          );
          elapsed.innerText = `Elapsed: ${(Date.now() - startTime) / 1000}s`;
        },
      },
    })
    .then((result) => {
      btnStart.disabled = false;
      isTraining = false;
      inferenceResult.className = 'ready';
      return result;
    });
}

function infer(model, result) {
  console.log(result);
  dfd.read_csv(TEST_SET_URL).then((df) => {
    console.log('df', df);
    const X_test = df.loc({
      columns: ['petal_h', 'petal_w', 'sepal_h', 'sepal_w'],
    });
    const encoder = new dfd.OneHotEncoder();
    const gt = encoder.fit(df['type']);
    const inference = new dfd.DataFrame(model.predict(X_test.tensor));
    console.log('inference', inference);
    gt.plot('gtTable').table();
    inference.plot('inferenceTable').table();
    gt.plot('gtChart').line();
    inference.plot('inferenceChart').line();
  });
}

async function main() {
  const { X_train, Y_train } = await prepareData();
  const model = createModel();
  const result = await train(model, X_train, Y_train);
  infer(model, result);
}

btnStart.addEventListener('click', () => {
  if (!isTraining) {
    isTraining = true;
    btnStart.disabled = true;
    inferenceResult.className = 'training';
    gtTable.innerHTML = '';
    inferenceTable.innerHTML = '';
    gtChart.innerHTML = '';
    inferenceChart.innerHTML = '';
    main();
  }
});

backend.addEventListener('change', () => {
  tf.setBackend(backend.value);
});
