require_relative './match_loader.rb'

# MatchLoader.parse_data

# dot product of theta vector and x vector
def theta_t_x(theta, x_vec)
  sum = theta.first
  (0...x_vec.length).each do |idx|
    sum += theta[idx + 1]*x_vec[idx]
  end
  sum
end

# This is the hypothesis
def sigmoid(theta, x_vec)
  1.0/(1 + Math.exp(-1*theta_t_x(theta, x_vec)))
end

#=======================================================================

# This is the cost function for minimization
def cost_function(theta, x, y, lambda_factor)
  m, n = y.length, theta.length

  theta_sum = (1...n).inject(0) {|sum, i| sum += theta[i]**2}
  theta_sum *= lambda_factor/(2*m)

  sum = (0...m).inject(0) do |accum, i|
    temp = y[i]*Math.log(sigmoid(theta, x[i]))
    temp += (1 - y[i])*Math.log(1 - sigmoid(theta, x[i]))
    accum += temp
  end

  if (sum.abs > 1000)
    debugger
  end
  theta_sum - (sum/m)
end

# This is the derivative of cost function for parameter 0
def dj_theta_0(theta, x, y)
  m = y.length
  sum = (0...m).inject(0) do |accum, i|
    accum += (sigmoid(theta, x[i]) - y[i])*x[i][0]
  end
  sum.to_f/m
end

# This is the dervative of cost function for parameter j's
def dj_theta_j(theta, x, y, j, lambda_factor)
  m = y.length
  sum = (0...m).inject(0) do |accum, i|
    accum += (sigmoid(theta, x[i]) - y[i])*x[i][j-1]
  end
  (sum.to_f/m) + lambda_factor*theta[j]/m
end

# Gradient Descent for Logistic Regression
def gradient_descent(x, y)
  # Specify learning rate, and lambda factor
  a, l = 0.05, 0.001
  # Initialize theta
  thetas = [0, 0, 0, 0]
  n = thetas.length # number of features
  loop do
    thetas_temp = thetas.dup
    dj_thetas = [dj_theta_0(thetas_temp, x, y)]
    (1...n).each do |j|
      dj_thetas << dj_theta_j(thetas_temp, x, y, j, l)
    end
    convergence = is_converging(dj_thetas)

    (0...n).each do |j|
      thetas[j] = thetas[j] - a*dj_thetas[j]
    end
    cost = cost_function(thetas, x, y, l)
    puts "Cost function: #{cost}, convergence: #{convergence}"
    puts dj_thetas.to_s
    break if is_converging(dj_thetas)
  end
  thetas
end

def is_converging(dj_thetas)
  sum = dj_thetas.inject(0) do |accum, dj_theta|
    accum += dj_theta*dj_theta
  end
  sum <= 0.0000001
end

# Happiness, Availability, Makeup Frequency
x = [[8.0, 9.0, 1.0],[7.0, 2.0, 3.0],[9.0, 7.0, 5.0], [8.5, 1.0, 3.0]]
y = [0, 1, 0, 1]
params = gradient_descent(x, y)
p sigmoid(params, [7.5, 1.0, 4.0])
