"use strict";
/*
  Parsing data
*/
const matchData = require('./matches.json');
const matches = matchData.matches;
function getTeamGoldEarned(teamMembers) {
  let totalGold = 0;
  for (let i = 0; i < teamMembers.length; i++) {
    totalGold += teamMembers[i].stats.goldEarned;
  }
  return totalGold;
}

function getTeamKDA(teamMembers) {
  let kills = 0, deaths = 0, assists = 0;
  for (let i = 0; i < teamMembers.length; i++) {
    kills += teamMembers[i].stats.kills;
    deaths += teamMembers[i].stats.deaths;
    assists += teamMembers[i].stats.assists;
  }
  return (kills + assists)/deaths;
}

function getTeamCSRate(teamMembers) {
  let csPerMin = 0;
  for (let i = 0; i < teamMembers.length; i++) {
    let csDeltas = teamMembers[i].timeline.creepsPerMinDeltas;
    csPerMin += (csDeltas.zeroToTen + csDeltas.tenToTwenty)/2;
  }
  return csPerMin;
}

function parseData() {
  let parsedData = [];
  for (let i = 0; i < matches.length; i++) {
    let t1DataPoint = {x: []}, t2DataPoint = {x: []};
    let matchDuration = matches[i].matchDuration;

    let team1 = matches[i].participants.slice(0, 5);
    t1DataPoint.x.push(getTeamGoldEarned(team1)/matchDuration);
    t1DataPoint.x.push(getTeamKDA(team1));
    t1DataPoint.x.push(getTeamCSRate(team1));

    let team2 = matches[i].participants.slice(5, 10);
    t2DataPoint.x.push(getTeamGoldEarned(team2)/matchDuration);
    t2DataPoint.x.push(getTeamKDA(team2));
    t2DataPoint.x.push(getTeamCSRate(team2));

    if (matches[i].teams[0].winner) {
      t1DataPoint.y = 1;
      t2DataPoint.y = 0;
    } else {
      t1DataPoint.y = 0;
      t2DataPoint.y = 1;
    }
    parsedData.push(t1DataPoint);
    parsedData.push(t2DataPoint);
  }
  return parsedData;
}

function getXYVectors() {
  let x = [], y = [];
  let parsedData = parseData();
  for (let i = 0; i < parsedData.length; i++) {
    x.push(parsedData[i].x);
    y.push(parsedData[i].y);
  }
  return [x, y];
}

/*
  Logistic Regression
*/

// This is equivalent to dot product of two vector
// Theta0 + Theta1 * X1 + Theta2 * X2 + ...
function thetaTransposeX(theta, xVector) {
  let sum = theta[0];
  for (let i = 0; i < xVector.length; i++) {
    sum += xVector[i]*theta[i + 1];
  }
  return sum;
}

// Hypothesis function
function sigmoid(theta, xVector) {
  return 1/(1 + Math.exp(-1 * thetaTransposeX(theta, xVector)));
}

function costFunction(theta, x, y, lambda) {
  // m denotes # of data points, n denotes # of features.
  let m = y.length, n = theta.length;
  let regularizedFactor = theta.slice(1).reduce((accum, el) => {return accum + el;});
  regularizedFactor *= lambda/(2*m);
  let sum = 0;
  for (let i = 0; i < m; i++) {
    let temp = y[i]*Math.log(sigmoid(theta, x[i]));
    temp += (1 - y[i])*Math.log(1 - sigmoid(theta, x[i]));
    sum += temp;
  }
  return regularizedFactor - (sum/m);
}

// Derivative of cost function with respect to theta 0
function djTheta0(theta, x, y) {
  let m = y.length;
  let sum = 0;
  for (let i = 0; i < m; i++) {
    sum += (sigmoid(theta, x[i]) - y[i])*x[i][0];
  }
  return (sum/m);
}

// Derivative of cost function with respect to theta j
function djThetaJ(theta, x, y, j, lambda) {
  let m = y.length;
  let sum = 0;
  for (let i = 0; i < m; i++) {
    sum += (sigmoid(theta, x[i]) - y[i])*x[i][j - 1];
  }
  return (sum/m) + lambda*theta[j]/m;
}

function vectorNorm(vector) {
  let sum = 0;
  for (let i = 0; i < vector.length; i++) {
    sum += vector[i]*vector[i];
  }
  return sum;
}

function isConverged(gradientVector) {
  return (vectorNorm(gradientVector) < 0.0000001);
}

function gradientDescent(x, y) {
  // a: learning rate, lambda: regularization
  let a = 0.01, lambda = 0.001, thetas = [0, 0, 0, 0];
  let n = thetas.length, gradient;
  do {
    let thetasTemp = thetas.slice();
    gradient = [djTheta0(thetasTemp, x, y)];
    for (let j = 1; j < n; j++) {
      gradient.push(djThetaJ(thetasTemp, x, y, j, lambda));
    }

    for (let j = 0; j < n; j++) {
      thetas[j] = thetas[j] - (a*gradient[j]);
    }
    console.log(`Cost function: ${costFunction(thetas, x, y, lambda)}`);
    console.log(`Vector norm of gradient vector: ${vectorNorm(gradient)}`);
  } while (!isConverged(gradient));

  return thetas;
}

let data = getXYVectors();
let trainingX = data[0].slice(0, 180), testX = data[0].slice(180,200);
let trainingY = data[1].slice(0, 180), testY = data[1].slice(180,200);

let params = gradientDescent(trainingX, trainingY);

console.log(params);

// for (let i = 0; i < testY.length; i++) {
//   let prob = sigmoid(params, testX[i]);
//   console.log(`${i}: ${testX[i]}, Prob: ${Math.round(prob*10000)/100}, y: ${testY[i]}`);
// }

// for (let i = 0; i < testY.length; i++) {
//   let val = sigmoid(params, testX[i]), prediction;
//   if (val > 0.50) {
//     prediction = 1;
//   } else {
//     prediction = 0;
//   }
//   if (prediction === testY[i]) {
//     console.log(`Probability: ${Math.round(val*10000)/100}%, Result: PASSED`);
//   } else {
//     console.log(`Probability: ${Math.round(val*10000)/100}%, Result: FAILED`);
//   }
// }

// Writing data to file
// const jsonfile = require('jsonfile');
// let file = './parsed-data.json';
// let obj = {
//   win: {
//     x1: [],
//     x2: [],
//     x3: []
//   },
//   loss: {
//     x1: [],
//     x2: [],
//     x3: []
//   }
// };
//
// for (let i = 0; i < trainingX.length; i++) {
//   if (trainingY[i] === 1) {
//     obj.win.x1.push(Math.round(trainingX[i][0]*1000)/1000);
//     obj.win.x2.push(Math.round(trainingX[i][1]*1000)/1000);
//     obj.win.x3.push(Math.round(trainingX[i][2]*1000)/1000);
//   } else {
//     obj.loss.x1.push(Math.round(trainingX[i][0]*1000)/1000);
//     obj.loss.x2.push(Math.round(trainingX[i][1]*1000)/1000);
//     obj.loss.x3.push(Math.round(trainingX[i][2]*1000)/1000);
//   }
// }
//
// jsonfile.writeFile(file, obj, function (err) {
//   console.error(err);
// });
