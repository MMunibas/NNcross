import numpy  as np
import pickle as pkl

#this is a DataContainer class that stores data
class DataContainer:
    def __repr__(self):
         return "DataContainer"
    def __str__(self):
         return self._data.to_string()
    def __init__(self, nt, nv, seed=None, load_test=False):

        self._num_data = 12000 #Total number of data
#---------------------------------------------------------------
#data structure of the input file
#input features for reactants (1 to 12)
#E(v,j), E(v,j=0), v, E(v=0,j), j, sqrt((j*(j+1))), E_tra, v_rel, q+, q-, centrifugal barrier height, time period
#input features for products (13 to 24)
#E(v',j'), E(v',j'=0), v, E(v'=0,j'), j', sqrt((j'*(j'+1))), E_tra, v_rel, q+, q-, centrifugal barrier height, time period
#output features
#cross section
#---------------------------------------------------------------
        self._num_features = 24 #Total number of features
        self._num_outputs  = 1  #Total number of output
        self._load_test = load_test #whether to load the test set or not
        self._random_state = np.random.RandomState(seed=seed)

        #load the data set
        data = np.genfromtxt('input.csv', delimiter=',')

        #create shuffled list of indices
        indices = self._random_state.permutation(np.arange(0,self._num_data))

        #store indices of training, validation and test data
        indices_train = indices[0:nt]
        indices_valid = indices[nt:nt+nv]
        indices_test  = indices[nt+nv:]

        #save number of training, validation and test data
        self._num_train = indices_train.shape[0]
        self._num_valid = indices_valid.shape[0]
        self._num_test  = indices_test.shape[0]

        #store 3 different sets of data
        self._data_train = data[indices_train,:]
        self._data_valid = data[indices_valid,:]
        if self._load_test:
            self._data_test  = data[indices_test,:]

        #calculate running mean/stdev to normalize the data
        n = 0
        S = np.zeros(self._num_features, dtype=float)
        m = np.zeros(self._num_features, dtype=float)
        for i in range(self._num_train):
            #loop through the descriptor and update mean/stdev
            n += 1 #keeps track of how many samples have been analyzed for mean/stdev
            for j in range(self._num_features):
                #update mean/stdev
                m_prev = m[j]
                m[j] += (self._data_train[i,j]-m[j])/n
                S[j] += (self._data_train[i,j]-m[j]) * (self._data_train[i,j]-m_prev)     

        stdev = np.sqrt(S/n)

        np.savetxt('Coeff_mval.txt',m, delimiter=',')
        np.savetxt('Coeff_stdv.txt',stdev, delimiter=',')
#        m = np.genfromtxt('Coeff_mval.txt', delimiter=',')   
#        stdev = np.genfromtxt('Coeff_stdv.txt', delimiter=',')

        #rescale features by standardization
        #training set
        for i in range(self._num_train):
            for j in range(self._num_features):
                if stdev[j] > 0.0:
                    self._data_train[i,j] = (self._data_train[i,j]-m[j])/stdev[j]
                else:
                    self._data_train[i,j] = (self._data_train[i,j]-m[j])

        #validation set
        for i in range(self._num_valid):
            for j in range(self._num_features):
                if stdev[j] > 0.0:
                    self._data_valid[i,j] = (self._data_valid[i,j]-m[j])/stdev[j]
                else:
                    self._data_valid[i,j] = (self._data_valid[i,j]-m[j])

        #test set
        if self._load_test: 
            for i in range(self._num_test):
                for j in range(self._num_features):
                    if stdev[j] > 0.0:
                        self._data_test[i,j] = (self._data_test[i,j]-m[j])/stdev[j]
                    else:
                        self._data_test[i,j] = (self._data_test[i,j]-m[j])    
        
        #for retrieving batches
        self._index_in_epoch = 0 

    #total amount of data
    @property
    def num_data(self):
        return self._num_data

    #number of training examples
    @property
    def num_train(self):
        return self._num_train

    #number of validation examples
    @property
    def num_valid(self):
        return self._num_valid

    #number of test examples
    @property
    def num_test(self):
        return self._num_test

    @property
    def num_features(self):
        return self._num_features

    @property
    def num_outputs(self):
        return self._num_outputs

    #shuffles the training data
    def shuffle_train_data(self):
        indices = self._random_state.permutation(np.arange(0,self._num_train))
        self._data_train = self._data_train[indices,:]

    #returns a batch of samples from the training set
    def next_batch(self, batch_size=1):
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        #epoch is finished, test set needs to be shuffled
        if self._index_in_epoch > self.num_train:
            #shuffle training data
            self.shuffle_train_data()
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_train
        end = self._index_in_epoch
        return self._data_train[start:end,:self._num_features], self._data_train[start:end,self._num_features:self._num_features+self._num_outputs]

    def get_train_data(self):
        return self._data_train[:,:self._num_features], self._data_train[:,self._num_features:self._num_features+self._num_outputs]

    def get_valid_data(self):
        return self._data_valid[:,:self._num_features], self._data_valid[:,self._num_features:self._num_features+self._num_outputs]

    def get_test_data(self):
        assert self._load_test
        return self._data_test[:,:self._num_features], self._data_test[:,self._num_features:self._num_features+self._num_outputs]

    def get_one_data(self, nindx):
        return self._data_valid[nindx,:self._num_features], self._data_valid[nindx,self._num_features:self._num_features+self._num_outputs]

    def get_onetest_data(self, nindx):
        return self._data_test[nindx,:self._num_features], self._data_test[nindx,self._num_features:self._num_features+self._num_outputs]
