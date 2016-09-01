
import tensorflow as tf
import numpy as np
tf.reset_default_graph()
sess = tf.Session()

"""This code 's structure
1.load data/ parameters
2. RNN construction
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
hidden_units = hidden_units[0][0]
L1_parameter = np.asscalar(L1_parameter)
L2_parameter = np.asscalar(L2_parameter)
dropout_keep_prob = float(dropout_keep_prob) # 1.0 is usual
decision_pi_rate = np.asscalar(decision_pi_rate)
utility_parameter = np.asscalar(utility_parameter)
evaluation_price_impact_rate = np.asscalar(evaluation_pi_rate)
eps_multiplier = float(eps_multiplier) # multiplier used for last layer initialization

num_nodes = train_actuals.shape[1] # output dimension
num_features = train_features.shape[1] #input dimension


#######THESE PARAMETERS ARE SPECIAL###################
num_input_cells = int(np.asscalar(num_time_steps)) # how many train examples to feed in at once
stacked_cell_size = np.asscalar(stacked_cell_size) # how many hidden LSTM units
num_test_cells = predict_features.shape[0]
peephole = int(np.asscalar(peephole))
slide = int(np.asscalar(slide))
considerOnlyLAST24 = int(np.asscalar(considerOnlyLAST24))
#In order to make the train step integers, I write this.
num_train_examples = train_features.shape[0]

if slide == 1:
    f_temp = train_features[0:0+num_input_cells,:]
    a_temp = train_actuals[0:0+num_input_cells,:]
    num_batches = (num_train_examples - num_input_cells)/24 + 1
    for row in range(num_batches):
        if row ==0: continue
        f_new = train_features[24*row:24*row+num_input_cells,:]
        f_temp = np.r_[f_temp,f_new]
        a_new =  train_actuals[24*row:24*row+num_input_cells,:]
        a_temp = np.r_[a_temp,a_new]
    train_features = f_temp
    train_actuals = a_temp
else:
    num_batches = int(np.floor(num_train_examples/num_input_cells))
    rounded_num_train_examples = num_batches * num_input_cells
    train_features = train_features[-rounded_num_train_examples:,:] # take data from the last.
    train_actuals = train_actuals[-rounded_num_train_examples:,:]

'''This is the special point of RNN data structure'''
# num_batches, num_input_cells, num_features -> we need this for large train window.
split_train_features = np.vsplit(train_features,num_batches)  
split_train_actuals = np.vsplit(train_actuals,num_batches) 

if slide ==1: # Release the memory
    del train_actuals, train_features
    train_features=split_train_features[-1] 
    train_actuals = split_train_actuals[-1]


# num_input_cells is how many time steps we feed in at once
# None will be number of batches
keep_prob = tf.placeholder("float")
features = tf.placeholder("float", [None, num_input_cells, num_features]) 
actuals = tf.placeholder("float", [None, num_input_cells, num_nodes]) # actual

# initialization for final output layer after LSTM
weights = {
    'out': tf.Variable(tf.random_normal([hidden_units, num_nodes], stddev=eps_multiplier)) # 64 x 2
}
biases = {
    'out': tf.Variable(tf.random_normal(shape=[num_nodes], stddev=eps_multiplier))
}

# (num_batches, num_input_cells, num_features) -> (num_input_cells, num_batches, num_features)
temp = tf.transpose(features, [1, 0, 2])
# Reshaping to (num_input_cells*num_batches, num_features)
temp = tf.reshape(temp, [-1, num_features])
# each LSTM unit (there are num_input_cells of them) takes in a matrix which is (num_batches x num_features)
# ex: the first unit takes the first hour in each of the num_batches blocks.
# we do this so we can take gradients in parallel.
lstm_inputs = tf.split(0, num_input_cells, temp)


'''2. RNN construction'''

# forget_bias is recommended to be 1
# NOTE: make a placeholder!
# Coz - can you make more comments here on what is going on? and what the initialization of the lstm is?
with tf.variable_scope("model",reuse=None):
  """Define the basis LSTM"""
  with tf.name_scope("LSTM_setup") as scope:
        if peephole ==1:
            lstm_cell = tf.nn.rnn_cell.LSTMCell(hidden_units, forget_bias=1.0,use_peepholes=True)
        else:
            lstm_cell = tf.nn.rnn_cell.LSTMCell(hidden_units, forget_bias=1.0,use_peepholes=False)        
        lstm_cell = tf.nn.rnn_cell.LSTMCell(hidden_units, forget_bias=1.0)
        lstm_do_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell, input_keep_prob=keep_prob, output_keep_prob=keep_prob, seed=None)
        stacked_lstm = tf.nn.rnn_cell.MultiRNNCell([lstm_do_cell] * int(stacked_cell_size)) # multilayer 
        outputs, states = tf.nn.rnn(lstm_cell, lstm_inputs, dtype=tf.float32)

# output is still a list
outputs = tf.pack(outputs) # num_input_cells X num_batches X hidden_units
outputs = tf.transpose(outputs, [1, 0, 2]) # shape is num_batches X num_input_cells X hidden

pred = tf.reshape(outputs,(-1,hidden_units) ) # pred has size (num_batches * num_input_cells, hidden )

temp_out =tf.matmul(pred, weights['out']) #+ biases['out'] #  (num_batches * num_input_cells, num_nodes )
outputs_batch_nodes = tf.reshape(temp_out,(-1,num_input_cells,num_nodes))
raw_volumes_3d = outputs_batch_nodes + biases['out'] # num_batches, num_input_cells, num_nodes


'''3.Centering and scaling'''
raw_volumes = tf.reshape(raw_volumes_3d,shape=(-1,num_nodes))
transposed_output = tf.transpose(tf.transpose(raw_volumes) - tf.transpose(tf.reduce_mean(raw_volumes, 1)))
total_volumes = tf.reduce_sum(tf.abs(transposed_output), 1)
volume_fraction = hourly_volume/total_volumes
too_much_volume = tf.to_float(total_volumes >= hourly_volume)
values_needing_scaling = tf.mul(volume_fraction, too_much_volume)
not_too_much_volume = tf.to_float(total_volumes < hourly_volume)
factors = not_too_much_volume + values_needing_scaling
new_volumes = tf.transpose(tf.mul(tf.transpose(transposed_output), factors))
new_volumes = tf.reshape(new_volumes,shape=(-1,num_input_cells,num_nodes) ) # num_batches, num_input_cells, num_nodes


'''hourly return calc'''
node_hour_returns = tf.mul(new_volumes, actuals)  # new_volumes:num_batches x num_input_cells x num_nodes, prediction_actuals
return_with_price_impact = tf.sub( node_hour_returns, tf.scalar_mul(decision_pi_rate, tf.mul(new_volumes,new_volumes)) )
hourly_returns = tf.reduce_sum(return_with_price_impact, 2) # sum over node then, num_batches x num_input_cells
# rev fee estimated at $0.06 / MW
fees = tf.scalar_mul( 0.06 , tf.reduce_sum(tf.abs(new_volumes),  2) )
# returns include fees
price_impacted_profit_MATRIX =tf.sub(hourly_returns , fees) # num_batches x num_input_cells

"""only taking last 24 hours"""
if considerOnlyLAST24 == 1:
    slice_begining_num = num_input_cells -24
    price_impacted_profit_LAST24H =tf.slice(price_impacted_profit_MATRIX ,[0,slice_begining_num],[-1,-1]) # num_batches x LAST24H_input_cells
    price_impacted_profit_MATRIX = price_impacted_profit_LAST24H
    


"""This is the utility function"""  
if utility_function == 'exponential':
    utility_parameter = -1.0/utility_parameter
    utility_temp = -1*tf.exp(tf.scalar_mul(utility_parameter,price_impacted_profit))
    utility_mean_perData = tf.reduce_mean(utility_temp,1) #per data
    utility_mean = tf.reduce_mean(utility_mean_perData)

elif utility_function == 'profit':
    utility_mean = tf.reduce_mean(price_impacted_profit)

elif utility_function == 'sharpe':
    # tempDiff = tf.transpose( tf.sub(tf.transpose(price_impacted_profit) , tf.transpose(tf.reduce_mean(price_impacted_profit,1))) ) # n_batch x -1 
    # squaredTempDiff = tf.mul(tempDiff,tempDiff) 
    # variance_perData = tf.reduce_sum(squaredTempDiff,1) 
    # utility_mean_perData = tf.div(tf.reduce_mean(price_impacted_profit,1),tf.sqrt(variance_perData))
    # utility_mean = tf.reduce_mean(utility_mean_perData)
    # reshape to vector line in FFNN, and calculate Sharpe
    price_impacted_profit_VEC = tf.reshape(price_impacted_profit_MATRIX, shape=[-1])
    tempDiff = tf.squared_difference(price_impacted_profit_VEC, tf.reduce_mean(price_impacted_profit_VEC))
    variance = tf.reduce_mean(tempDiff)
    utility_mean = tf.div(tf.reduce_mean(price_impacted_profit_VEC),tf.sqrt(variance))    

elif utility_function == 'ali_sharpe': ## mean() - (constant * std() )
    price_impacted_profit_VEC = tf.reshape(price_impacted_profit_MATRIX, shape=[-1])
    tempDiff = tf.squared_difference(price_impacted_profit_VEC, tf.reduce_mean(price_impacted_profit_VEC))
    variance = tf.reduce_mean(tempDiff)
    utility_mean = tf.sub(tf.reduce_mean(price_impacted_profit_VEC), utility_parameter*tf.sqrt(variance))    

"""5.Objective function and L1 regularizer"""
tempL1regularizer = tf.constant(0.0)
#for i in range(len(TensorWeightDict)):
#    tempL1regularizer = tf.add(tempL1regularizer, tf.reduce_sum(tf.abs(TensorWeightDict[i].weight) ))
tempL1regularizer = tf.reduce_sum(tf.abs(weights['out']) )
L1regularizer = - tf.mul(L1_parameter ,tempL1regularizer) # is the constant/scale


step = tf.Variable(0, trainable=False)
rate_use = tf.train.exponential_decay(learning_rate, step, 1, learning_rate_decay)
train_step = tf.train.AdamOptimizer(learning_rate=rate_use, beta1=beta_1, beta2=beta_2, epsilon=eta, use_locking=False, name='Adam').minimize(-tf.add(utility_mean,L1regularizer), global_step=step)
# This is how to use the AdamOpt
#tf.train.AdamOptimizer.__init__(learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-08, use_locking=False, name='Adam')
#https://www.tensorflow.org/versions/r0.9/api_docs/python/train.html#AdamOptimizer



sess.run(tf.initialize_all_variables())

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


# Here, newPredictF and newPredictA have 24 hours of normal predict data and training data just before those 24 hours. 
# For example, if training is 1~100, and predict is 101~124 and num_input_cells is 50, then newPredictF has 75~124.
# Coz - Can you say more here, Im still confused
trainFeature_beforePredict = train_features[-(num_input_cells - num_test_cells ):,:]
trainActual_beforePredict = train_actuals[-(num_input_cells - num_test_cells ):,:]

newPredictF =[ np.r_[trainFeature_beforePredict, predict_features] ]
newPredictA =[ np.r_[trainActual_beforePredict, predict_actuals] ]

feed_dict_Train = {features:split_train_features, actuals:split_train_actuals, keep_prob:dropout_keep_prob}
feed_dict_Pred = {features:newPredictF, actuals:newPredictA, keep_prob:1.0}

def shape_calc(matrix_batch_x_hours):
    std = np.sqrt(np.var(matrix_batch_x_hours))
    mean = np.mean(matrix_batch_x_hours)
    if utility_function == 'sharpe':
        return mean/std
    elif utility_function == 'ali_sharpe':
        return mean - utility_parameter*std


for i in xrange(iterations): # epoch for gradient
    train_step.run(feed_dict=feed_dict_Train,session=sess)

    if not live:
        # Training information for later analysis
        # Matlab pulls these variables using matpy and saves them in .mat
        train_profit = price_impacted_profit_MATRIX.eval(feed_dict=feed_dict_Train,session=sess)
        all_training_returns[i] = train_profit.mean()        
        
        train_vol = new_volumes.eval(feed_dict=feed_dict_Train,session=sess)
        all_training_volume_square_norms[i] = np.mean(np.sum(np.square(train_vol),2))

        all_training_utility[i] = utility_mean.eval(feed_dict=feed_dict_Train,session=sess) 
        
        test_profit = price_impacted_profit_MATRIX.eval(feed_dict=feed_dict_Pred,session=sess)[:,-24:]
        all_test_returns[i] = test_profit.mean()
		
        all_test_utility[i] = shape_calc(test_profit)


    all_validation_utility[i] = 0.0
    all_validation_returns[i] = 0.0
    all_validation_volume_square_norms[i] = 0.0
    all_training_negatives[i] = 0.0 # placeholder for us

        
"""7.prediction for new data/ test data""" 
#you can make a prediction by eval function, with the same session.
optimalVolumes = new_volumes.eval(feed_dict={features:newPredictF,keep_prob:1.0},session=sess)[0][-24:,:]

tf.reset_default_graph() 
del outputs, states, stacked_lstm, lstm_cell, lstm_do_cell 
sess.close()
