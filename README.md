# Logistic Regression
The complete write-up can be found on my github [page]
[page]: https://calvinfeng.github.io/logistic-regression.html

This is a lazy write-up because the details are on my github page.

## Hypothesis
The following is a sigmoid function

![sigmoid]
[sigmoid]: ./img/sigmoid.png
Sigmoid function is a S-shaped function with a range from 0 to 1 as output. This is the basis for
our hypothesis in this regression model.

![prob]
[prob]: ./img/prob.png
Our hypothesis function describes the probability of `y = 1` given a list of parameters and features.

What does `y = 1` mean though? The output `denoted as y` takes on two values, either 0 or 1. It is an answer to the yes-or-no question. For this particular project, 1 represents a win while 0 represents a loss in League of Legend matches. But of course, this regression model can generalize to many other real life applications.

![hypo]
[hypo]: ./img/hypo.png
![theta-t-x]
[theta-t-x]: ./img/theta-transpose-x.png
``` javascript
function thetaTransposeX(theta, xVector) {
  let sum = theta[0];
  for (let i = 0; i < xVector.length; i++) {
    sum += xVector[i]*theta[i + 1];
  }
  return sum;
}
```
This is the hypothesis function. `x = [x1, x2, x3]` which is called the feature vector. Features are the inputs. For example, if we were to predict housing price, x1 can be the size of a house, x2 can be the number of bedrooms in a house, and x3 can be the size of the garage in a house. The theta(s) are the parameters we are trying to find through gradient descent.

``` javascript
// Hypothesis function
function sigmoid(theta, xVector) {
  return 1/(1 + Math.exp(-1 * thetaTransposeX(theta, xVector)));
}
```

## Minimizing cost
![cost]
[cost]: ./img/cost-function.png
Gradient descent is an optimization technique for maximizing likelihood, or in other words, it minimizes the cost function. What does cost function do though? The cost function J is describing how well the hypothesis fits with the training data.
``` javascript
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
```
## Gradient Descent
We will repeat the following code until convergence.

![theta-0]
[theta-0]: ./img/theta-0.png
![theta-j]
[theta-j]: ./img/theta-j.png

``` javascript
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
  } while (!isConverged(gradient));

  return thetas;
}
```
