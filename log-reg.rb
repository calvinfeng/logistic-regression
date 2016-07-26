require_relative './match-loader.rb'

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
  #
  # if (sum.abs > 1000)
  #   debugger
  # end
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
  a, l = 0.01, 0.001
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
    puts "Vector norm of gradient vector: #{vector_norm(dj_thetas)}"
    break if is_converging(dj_thetas)
  end
  thetas
end

def vector_norm(vector)
  vector.inject(0) do |accum, el|
    accum += el*el
  end
end

def is_converging(dj_thetas)
  vector_norm(dj_thetas) <= 0.0000001
end

match_data = MatchLoader.parse_data
x, y = [], []
match_data.take(180).each do |data_point|
  # x = [gold_earned, team_kda, team_cs_rate]
  x << data_point[:x]
  y << data_point[:y]
end
params = gradient_descent(x, y)
puts "Gradient descent has been completed=============================="
match_data.drop(180).each do |data_point|
  val = sigmoid(params, data_point[:x])
  if  val > 0.50
    prediction = 1
  else
    prediction = 0
  end

  if prediction == data_point[:y]
    puts "PASSED"
  else
    puts "FAILED"
  end
end
