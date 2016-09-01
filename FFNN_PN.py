

import tensorflow as tf
import numpy as np
sess = tf.Session()

"""This code 's structure
1.load data/ parameters
2. NN construction
3.centering and scaling
4.Utility
5.Objective function using utility and L1 regularizer
6.Train function definition and Roop of training
7. PREDICTION
"""


"""
1.load data / parameters.

"""

# convert 1x1 arrays to scalar values 
hourly_volume = np.asscalar(hourly_volume)
iterations = int(np.asscalar(iteration_count))
learning_rate = np.asscalar(learning_rate)
learning_rate_decay = float(learning_rate_decay)
beta_1 = float(beta_1)
beta_2 = float(beta_2)
eta = float(eta)
hidden_units = hidden_units.astype(np.int32)
hidden_units = hidden_units[0]
L1_parameter = np.asscalar(L1_parameter)
L2_parameter = np.asscalar(L2_parameter)
dropout_keep_prob = float(dropout_keep_prob) # 1.0 is usual
decision_pi_rate = np.asscalar(decision_pi_rate)
utility_parameter = np.asscalar(utility_parameter)
evaluation_price_impact_rate = np.asscalar(evaluation_pi_rate)
eps_multiplier = float(eps_multiplier) # multiplier used for last layer initialization

num_features = train_features.shape[1] # is this the same? np.size(train_features, 1)
num_nodes = train_actuals.shape[1]




""" 2. NN construction
"""

layer_sizes = [num_features]
layer_sizes.extend([hidden_units])
layer_sizes.extend([num_nodes])
num_layers = len(layer_sizes)


def init_layer_weight(fan_in, fan_out, multiplier):
    epsilon = np.sqrt(6) / np.sqrt(fan_in + fan_out)
    ini = multiplier*tf.random_uniform( [fan_in,fan_out] ,minval= -epsilon, maxval= epsilon) 
    return tf.Variable(ini)


"""PlaceHolders"""
keep_prob = tf.placeholder(tf.float32)
features = tf.placeholder(tf.float32,shape=[None,num_features]) #48 hours , three features. 
actuals = tf.placeholder(tf.float32,shape=[None,num_nodes])  #acctual prices of two nodes.
# (KYO) SEE THE LINE OF "node_hour_returns = tf.mul(new_volumes, actuals)  # new_volumes:24x228, prediction_actuals"
#  (KYO) in this line, actuals is feeded to the tensors and then we can calculate the hour return.


class WeightBias: # H0 -(W0 B0)-> H1 -(W1 B1)-> H2
    def __init__(self,number,unitsize_previous,unitsize_next, multiplier):
        self.number = number
        self.name = 'layer_'+str(number)
        self.weight = init_layer_weight(unitsize_previous, unitsize_next, multiplier)
        self.bias = tf.Variable(tf.zeros([unitsize_next]))


TensorWeightDict ={}

for i in range(num_layers-1): # The number of weights is layer num - 1.
    if i == 0: 
        TensorWeightDict[i] = WeightBias(i,num_features,np.asscalar(hidden_units[i]), 1)
        continue
    elif i == (num_layers - 2) : # for last set of weights, include possible multiplier 
        TensorWeightDict[i] = WeightBias(i,np.asscalar(hidden_units[i-1]),num_nodes, eps_multiplier)
        continue
    else:
        TensorWeightDict[i] = WeightBias(i,hidden_units[ i - 1 ],hidden_units[i], 1)
        continue


# at the start lets use tanh activation since this is what we are using now.
# later we can experiment with relu!
#keep_prob = tf.constant(dropout_keep_prob) # THIS IS FOR DROPOUT, THE PARAMETER SHOULD BE CREATED see line 49

hiddenValues={} # H0 -(W0 B0)-> H1 -(W1 B1)-> H2
hiddenV_withDO = {}
for i in range(num_layers-1):
    if i ==0:
        hiddenValues[i] = tf.nn.tanh(tf.matmul(features, TensorWeightDict[i].weight) + TensorWeightDict[i].bias)
        #h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
        hiddenV_withDO[i] = tf.nn.dropout(hiddenValues[i], keep_prob) 
    elif i == num_layers - 2:
        raw_volumes = (tf.matmul(hiddenV_withDO[i-1],TensorWeightDict[i].weight) + TensorWeightDict[i].bias)
    else:
        hiddenValues[i] = tf.nn.tanh(tf.matmul(hiddenV_withDO[i-1], TensorWeightDict[i].weight) + TensorWeightDict[i].bias)
        hiddenV_withDO[i] = tf.nn.dropout(hiddenValues[i], keep_prob) 


""" 3.centering and scaling"""
transposed_output = tf.transpose(tf.transpose(raw_volumes) - tf.transpose(tf.reduce_mean(raw_volumes, 1)))
total_volumes = tf.reduce_sum(tf.abs(transposed_output), 1)
volume_fraction = hourly_volume/total_volumes
too_much_volume = tf.to_float(total_volumes >= hourly_volume)
values_needing_scaling = tf.mul(volume_fraction, too_much_volume)
not_too_much_volume = tf.to_float(total_volumes < hourly_volume)
factors = not_too_much_volume + values_needing_scaling
new_volumes = tf.transpose(tf.mul(tf.transpose(transposed_output), factors))


"""4. Utility calc"""
node_hour_returns = tf.mul(new_volumes, actuals)  # new_volumes:24x228, prediction_actuals 
return_with_price_impact = tf.sub( node_hour_returns, tf.scalar_mul(decision_pi_rate, tf.mul(new_volumes,new_volumes)) )
# node_returns_pi = node_returns - pi_rate * tf.square(predicted_volumes) same?

hourly_returns = tf.reduce_sum(return_with_price_impact, 1) #over node
# rev fee estimated at $0.06 / MW
fees = tf.scalar_mul( 0.06 , tf.reduce_sum(tf.abs(new_volumes),  1) )
# returns include fees
price_impacted_profit =tf.sub(hourly_returns , fees) # Tensor 24x1 # UNTIL HERE GRAD IS CALCLATED WELL
# IF Utility is log: minus is gonna be null.


"""This is utility function"""
if utility_function == 'exponential':
    utility_parameter = -1.0/utility_parameter
    utility_temp = -1*tf.exp(tf.scalar_mul(utility_parameter,price_impacted_profit))
    utility_mean = tf.reduce_mean(utility_temp)
elif utility_function == 'profit':
    utility_mean = tf.reduce_mean(price_impacted_profit)
elif utility_function == 'sharpe':
    #hourly_utilities = [np.mean(hourly_returns) / np.std(hourly_returns)]
    tempDiff = tf.sub(price_impacted_profit,tf.reduce_mean(price_impacted_profit))
    squaredTempDiff = tf.mul(tempDiff,tempDiff)
    variance = tf.reduce_mean(squaredTempDiff)
    utility_mean = tf.div(tf.reduce_mean(price_impacted_profit),tf.sqrt(variance))
elif utility_function == 'ali_sharpe': ## mean() - (constant * np.std() )
    tempDiff = tf.sub(price_impacted_profit,tf.reduce_mean(price_impacted_profit))
    squaredTempDiff = tf.mul(tempDiff,tempDiff)
    variance = tf.reduce_mean(squaredTempDiff)
    para_std = tf.mul(utility_parameter,tf.sqrt(variance))
    utility_mean = tf.sub(tf.reduce_mean(price_impacted_profit),para_std)
elif utility_function == 'Kyo_original1':
    price_positives_bool = tf.to_float(tf.to_int32(tf.less_equal(0.0,price_impacted_profit)) )# 0 or positive values.
    price_positives_val = tf.mul(price_positives_bool,price_impacted_profit)
    #test = tf.ones(tf.Tensor.get_shape(price_positives_val)[0] )
    #price_positives_val2 = tf.add( test,price_positives_val)
    price_positives_val2 = tf.add( 1.0000001, price_positives_val)
    modified_positives  = tf.log(price_positives_val2 )
    price_negatives_bool = tf.to_float(tf.to_int32(tf.greater(0.0,price_impacted_profit))) # 0 or positive values.
    price_negatives_val1 = tf.mul(price_negatives_bool,price_impacted_profit)
    #price_negatives_val2 = tf.add(tf.scalar_mul(-1.0,price_negatives_val1) , price_negatives_bool)
    #modified_negatives = tf.exp(price_negatives_val2)  # it was a problem since the negative val goes infinite minus
    utility_mean = tf.add(price_negatives_val1,modified_positives) # Tensor 24x1
    utility_mean = tf.reduce_mean(utility_mean) # Tensor 24x1
else:
    raise Exception('No utility function ' + utility_function)


"""5.Objective function and L1 regularizer"""
tempL1regularizer = tf.constant(0.0)
for i in range(len(TensorWeightDict)):
    tempL1regularizer = tf.add(tempL1regularizer, tf.reduce_sum(tf.abs(TensorWeightDict[i].weight) ))
L1regularizer = tf.mul(L1_parameter ,tempL1regularizer) # is the constant/scale


# this is what we wanna maximize.
# TF DOES MINIMIZATION
objective_value = tf.sub(L1regularizer,utility_mean)

# save profits and other things for monitoring of learning curves
all_training_returns = np.zeros((iterations,))
all_validation_returns = np.zeros((iterations,))
all_test_returns = np.zeros((iterations,))

all_training_volume_square_norms = np.zeros((iterations,))
all_validation_volume_square_norms = np.zeros((iterations,))

all_training_utility = np.zeros((iterations,))
all_validation_utility = np.zeros((iterations,))
all_test_utility = np.zeros((iterations,))

all_training_negatives = np.zeros((iterations,)) # placeholder for rest of code. We are not using this.


"""6.Train function definition and loop of training"""
# Add the optimizer and learning rate
step = tf.Variable(0, trainable=False)
rate_use = tf.train.exponential_decay(learning_rate, step, 1, learning_rate_decay)
train_step = tf.train.AdamOptimizer(learning_rate=rate_use, beta1=beta_1, beta2=beta_2, epsilon=eta, use_locking=False, name='Adam').minimize(objective_value, global_step=step)
#  (KYO) This is how to use the AdamOpt
#https://www.tensorflow.org/versions/r0.9/api_docs/python/train.html#AdamOptimizer
sess.run(tf.initialize_all_variables())


feed_dict_Train={features:train_features, actuals:train_actuals,keep_prob:dropout_keep_prob}
# Validation is now really a subset of the full training set, and Prediction is the next day
feed_dict_Validation={features: validation_features, actuals: validation_actuals,keep_prob:1.00}


for i in xrange(iterations): # epoch for gradient
    train_step.run(feed_dict=feed_dict_Train,session=sess)

    if not live:
        # Training information for later analysis
        # Matlab pulls these variables using matpy and saves them in .mat
        train_profit = price_impacted_profit.eval(feed_dict_Train,session=sess)
        all_training_returns[i] = train_profit.mean()

        train_vol = new_volumes.eval(feed_dict=feed_dict_Train,session=sess)
        all_training_volume_square_norms[i] = np.mean(np.sum(np.square(train_vol), axis = 1))

        all_training_utility[i] = utility_mean.eval(feed_dict=feed_dict_Train,session=sess)

        all_test_utility[i] = utility_mean.eval(feed_dict={features: predict_features, actuals: predict_actuals, keep_prob:1.00},session=sess)
        
        test_profit = price_impacted_profit.eval(feed_dict={features: predict_features, actuals: predict_actuals, keep_prob:1.00},session=sess)
        all_test_returns[i] = test_profit.mean()

        # use_validation is given through matpy
        if use_validation:
            validation_profit = price_impacted_profit.eval(feed_dict=feed_dict_Validation,session=sess)
            all_validation_returns[i] = validation_profit.mean() 

            all_validation_utility[i] = utility_mean.eval(feed_dict=feed_dict_Validation,session=sess)
            
            validation_vol = new_volumes.eval(feed_dict=feed_dict_Validation,session=sess)
            all_validation_volume_square_norms[i] = np.mean(np.sum(np.square(validation_vol), axis = 1))

        else:
            # Some placeholders to simplify saving
            all_validation_utility[i] = 0.0
            all_validation_returns[i] = 0.0
            all_validation_volume_square_norms[i] = 0.0


        all_training_negatives[i] = 0.0 # placeholder for us



"""7.prediction for new data/ test data""" 
#you can make a prediction by eval function, with the same session.
optimalVolumes  = new_volumes.eval(feed_dict={features:predict_features,keep_prob:1.0},session=sess)


