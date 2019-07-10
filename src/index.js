import "./styles.css";

import * as tf from "@tensorflow/tfjs";
import * as tfvis from "@tensorflow/tfjs-vis";
import * as Papa from "papaparse";
import * as Plotly from "plotly.js-dist";
import _ from "lodash";

Papa.parsePromise = function(file) {
  return new Promise(function(complete, error) {
    Papa.parse(file, {
      header: true,
      download: true,
      dynamicTyping: true,
      complete,
      error
    });
  });
};

const prepareData = async () => {
  const csv = await Papa.parsePromise(
    "https://raw.githubusercontent.com/curiousily/Simple-Neural-Network-with-TensorFlow-js/master/src/data/housing.csv"
  );

  return csv.data;
};

const renderHistogram = (container, data, column, config) => {
  const columnData = data.map(r => r[column]);

  const columnTrace = {
    name: column,
    x: columnData,
    type: "histogram",
    opacity: 0.7,
    marker: {
      color: "dodgerblue"
    }
  };

  Plotly.newPlot(container, [columnTrace], {
    xaxis: {
      title: config.xLabel,
      range: config.range
    },
    yaxis: { title: "Count" },
    title: config.title
  });
};

const renderScatter = (container, data, columns, config) => {
  var trace = {
    x: data.map(r => r[columns[0]]),
    y: data.map(r => r[columns[1]]),
    mode: "markers",
    type: "scatter",
    opacity: 0.7,
    marker: {
      color: "dodgerblue"
    }
  };

  var chartData = [trace];

  Plotly.newPlot(container, chartData, {
    title: config.title,
    xaxis: {
      title: config.xLabel
    },
    yaxis: { title: config.yLabel }
  });
};

const renderPredictions = (trueValues, slmPredictions, lmPredictions) => {
  var trace = {
    x: [...Array(trueValues.length).keys()],
    y: trueValues,
    mode: "lines+markers",
    type: "scatter",
    name: "true",
    opacity: 0.5,
    marker: {
      color: "dodgerblue"
    }
  };

  var slmTrace = {
    x: [...Array(trueValues.length).keys()],
    y: slmPredictions,
    name: "pred",
    mode: "lines+markers",
    type: "scatter",
    opacity: 0.5,
    marker: {
      color: "forestgreen"
    }
  };

  var lmTrace = {
    x: [...Array(trueValues.length).keys()],
    y: lmPredictions,
    name: "pred",
    mode: "lines+markers",
    type: "scatter",
    opacity: 0.5,
    marker: {
      color: "forestgreen"
    }
  };

  Plotly.newPlot("slm-predictions-cont", [trace, slmTrace], {
    title: "Simple Linear Regression predictions",
    yaxis: { title: "Price" }
  });

  Plotly.newPlot("lm-predictions-cont", [trace, lmTrace], {
    title: "Linear Regression predictions",
    yaxis: { title: "Price" }
  });
};

const VARIABLE_CATEGORY_COUNT = {
  OverallQual: 10,
  GarageCars: 5,
  FullBath: 4
};

// normalized = (value − min_value) / (max_value − min_value)
const normalize = tensor =>
  tf.div(
    tf.sub(tensor, tf.min(tensor)),
    tf.sub(tf.max(tensor), tf.min(tensor))
  );

const oneHot = (val, categoryCount) =>
  Array.from(tf.oneHot(val, categoryCount).dataSync());

const createDataSets = (data, features, categoricalFeatures, testSize) => {
  const X = data.map(r =>
    features.flatMap(f => {
      if (categoricalFeatures.has(f)) {
        return oneHot(!r[f] ? 0 : r[f], VARIABLE_CATEGORY_COUNT[f]);
      }
      return !r[f] ? 0 : r[f];
    })
  );

  const X_t = normalize(tf.tensor2d(X));

  const y = tf.tensor(data.map(r => (!r.SalePrice ? 0 : r.SalePrice)));

  const splitIdx = parseInt((1 - testSize) * data.length, 10);

  const [xTrain, xTest] = tf.split(X_t, [splitIdx, data.length - splitIdx]);
  const [yTrain, yTest] = tf.split(y, [splitIdx, data.length - splitIdx]);

  return [xTrain, xTest, yTrain, yTest];
};

const trainLinearModel = async (xTrain, yTrain) => {
  const model = tf.sequential();

  model.add(
    tf.layers.dense({
      inputShape: [xTrain.shape[1]],
      units: xTrain.shape[1],
      activation: "sigmoid"
    })
  );
  model.add(tf.layers.dense({ units: 1 }));

  model.compile({
    optimizer: tf.train.sgd(0.001),
    loss: "meanSquaredError",
    metrics: [tf.metrics.meanAbsoluteError]
  });

  const trainLogs = [];
  const lossContainer = document.getElementById("loss-cont");
  const accContainer = document.getElementById("acc-cont");

  await model.fit(xTrain, yTrain, {
    batchSize: 32,
    epochs: 100,
    shuffle: true,
    validationSplit: 0.1,
    callbacks: {
      onEpochEnd: async (epoch, logs) => {
        trainLogs.push({
          rmse: Math.sqrt(logs.loss),
          val_rmse: Math.sqrt(logs.val_loss),
          mae: logs.meanAbsoluteError,
          val_mae: logs.val_meanAbsoluteError
        });
        tfvis.show.history(lossContainer, trainLogs, ["rmse", "val_rmse"]);
        tfvis.show.history(accContainer, trainLogs, ["mae", "val_mae"]);
      }
    }
  });

  return model;
};

const run = async () => {
  const data = await prepareData();

  renderHistogram("qual-cont", data, "OverallQual", {
    title: "Overall material and finish quality (0-10)",
    xLabel: "Score"
  });

  renderHistogram("liv-area-cont", data, "GrLivArea", {
    title: "Above grade (ground) living area square feet",
    xLabel: "Area (sq. ft)"
  });

  renderHistogram("year-cont", data, "YearBuilt", {
    title: "Original construction date",
    xLabel: "Year"
  });

  renderScatter("year-price-cont", data, ["YearBuilt", "SalePrice"], {
    title: "Year Built vs Price",
    xLabel: "Year",
    yLabel: "Price"
  });

  renderScatter("qual-price-cont", data, ["OverallQual", "SalePrice"], {
    title: "Quality vs Price",
    xLabel: "Quality",
    yLabel: "Price"
  });

  renderScatter("livarea-price-cont", data, ["GrLivArea", "SalePrice"], {
    title: "Living Area vs Price",
    xLabel: "Living Area",
    yLabel: "Price"
  });

  const [
    xTrainSimple,
    xTestSimple,
    yTrainSimple,
    yTestIgnored
  ] = createDataSets(data, ["GrLivArea"], new Set(), 0.1);
  const simpleLinearModel = await trainLinearModel(xTrainSimple, yTrainSimple);

  const features = [
    "OverallQual",
    "GrLivArea",
    "GarageCars",
    "TotalBsmtSF",
    "FullBath",
    "YearBuilt"
  ];
  const categoricalFeatures = new Set([
    "OverallQual",
    "GarageCars",
    "FullBath"
  ]);
  const [xTrain, xTest, yTrain, yTest] = createDataSets(
    data,
    features,
    categoricalFeatures,
    0.1
  );
  const linearModel = await trainLinearModel(xTrain, yTrain);

  const trueValues = yTest.dataSync();
  const slmPreds = simpleLinearModel.predict(xTestSimple).dataSync();
  const lmPreds = linearModel.predict(xTest).dataSync();

  renderPredictions(trueValues, slmPreds, lmPreds);
};

if (document.readyState !== "loading") {
  run();
} else {
  document.addEventListener("DOMContentLoaded", run);
}
