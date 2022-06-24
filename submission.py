import numpy as np

class Node:
    '''
    Node class for the UpliftTreeRegressor class

    ...

    Attributes
    ----------
    X : np.ndarray (n * k)
        Array with the data contained in the node
    Y : np.ndarray (n)
        Array with target variables
    treatment : np.ndarray (n)
        Array with treatment flag
    min_samples_leaf : int
        Minimum number of values in a leaf
    min_samples_leaf_treatment : int
        Minimum number of treatment values in a leaf
    min_samples_leaf_control : int
        Minimum number of control values in a leaf
    left : Node
        Node to the left of the split containing values below split threshold 
    right : Node
        Node to the right of the split containing values above split threshold 
    split_threshold : float
        Threshold value for the node's split
    split_feature : int
        Index for the feature in X on which the node splits

    '''    
    def __init__(self,X,Y,treatment,depth,max_depth,min_samples_leaf,
                     min_samples_leaf_treated,min_samples_leaf_control):
        
        '''
        Initalizes the node and builds its left and right nodes

        ...

        Parameters
        ----------
        X : np.ndarray (n * k)
            Array with the data contained in the node
        Y : np.ndarray (n)
            Array with target variables
        treatment : np.ndarray (n)
            Array with treatment flag
        depth : int
            Nodes depth within the tree
        max_depth : int
            Maximum depth for the tree
        min_samples_leaf : int
            Minimum number of values in a leaf
        min_samples_leaf_treatment : int
            Minimum number of treatment values in a leaf
        min_samples_leaf_control : int
            Minimum number of control values in a leaf

        ''' 
        
        #initialize Node attributes
        self.X = X
        self.Y = Y
        self.treatment = treatment
            
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_leaf_control = min_samples_leaf_control
        self.min_samples_leaf_treated = min_samples_leaf_treated
        
        self.left = None
        self.right = None

        self.split_threshold = None
        self.split_feature = None
        best_diff = 0

        #Check if tree has reached depth. If yes stop building branches.
        if depth != max_depth:
                                
        #iterate over features and thresholds to find best split
            for col in range(len(self.X[0])):
                    
                column_values = self.X[:,col]
                #threshold algorithm
                unique_values = np.unique(column_values)
                if len(unique_values) > 10:
                    percentiles = np.percentile(column_values, [3, 5, 10, 20, 30, 50, 70, 80, 90, 95, 97])
                else:
                    percentiles = np.percentile(unique_values, [10, 50, 90])
                threshold_options = np.unique(percentiles)
                    
                for threshold in threshold_options:
                    diff = 0
                    #Check if split produces large enough leaves
                    if ((len(self.Y[column_values<threshold]) >= self.min_samples_leaf) and (len(self.Y[column_values>=threshold]) >= self.min_samples_leaf)
                        #Check if leaf has enough treated
                        and (sum(self.treatment[column_values<threshold]) >= self.min_samples_leaf_treated)
                        and (sum(self.treatment[column_values>=threshold]) >= self.min_samples_leaf_treated)
                        #and enough control
                        and ((len(treatment[column_values<threshold]) - sum(self.treatment[column_values<threshold])) >= self.min_samples_leaf_control) 
                        and ((len(treatment[column_values>=threshold]) - sum(self.treatment[column_values>=threshold])) >= self.min_samples_leaf_control)):
                                
                            #calculate uplifts for given split
                            Y_left = Y[column_values < threshold]
                            left_treatment = treatment[column_values < threshold]
                            uplift_left = ((Y_left[left_treatment == 1]).mean() - (Y_left[left_treatment == 0]).mean())
                                
                            Y_right = Y[column_values >= threshold]
                            right_treatment = treatment[column_values >= threshold]
                            uplift_right = ((Y_right[right_treatment == 1]).mean() - (Y_right[right_treatment == 0]).mean())

                            diff = abs(uplift_left - uplift_right)
                                
                            #check if this split is the best split
                            if diff > best_diff:
                                best_diff = diff
                                self.split_threshold = threshold
                                self.split_feature = col

            #if best split exists preform the split and create the left and right nodes
            if self.split_feature is not None:
                    
                column_values = self.X[:,self.split_feature]

                #left
                left_X = self.X[column_values<self.split_threshold]
                left_Y = self.Y[column_values<self.split_threshold]
                left_treatment = self.treatment[column_values<self.split_threshold]
                    
                self.left = Node(left_X,left_Y,left_treatment,depth+1,max_depth,self.min_samples_leaf,
                                 self.min_samples_leaf_treated,self.min_samples_leaf_control)
                    
                #right
                right_X = self.X[column_values>=self.split_threshold]
                right_Y = self.Y[column_values>=self.split_threshold]
                right_treatment = self.treatment[column_values>=self.split_threshold]
                    
                self.right = Node(right_X,right_Y,right_treatment,depth+1,max_depth,self.min_samples_leaf,
                                  self.min_samples_leaf_treated,self.min_samples_leaf_control)

class UpliftTreeRegressor:

    '''
    Uplift Decision Tree Class

    ...

    Attributes
    ----------
    
    max_depth : int
        Maximum depth for the tree
    min_samples_leaf : int
        Minimum number of values in a leaf
    min_samples_leaf_treatment : int
        Minimum number of treatment values in a leaf
    min_samples_leaf_control : int
        Minimum number of control values in a leaf
    root : Node
        Root of the decision tree

    Methods
    -------
    fit(X,treatment,Y)
        Generates the decision tree with the provided data
    predict(X)
        Predicts the target value for the provided X value 
    '''
                                         
    def __init__(self,max_depth = 3,
                 min_samples_leaf = 1000,
                 min_samples_leaf_treated = 300,
                 min_samples_leaf_control = 300):
        '''

        Parameters
        ----------
        depth : int
            Nodes depth within the tree
        max_depth : int
            Maximum depth for the tree
        min_samples_leaf : int
            Minimum number of values in a leaf
        min_samples_leaf_treatment : int
            Minimum number of treatment values in a leaf
        min_samples_leaf_control : int
            Minimum number of control values in a leaf

        '''                      
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_leaf_treated = min_samples_leaf_treated
        self.min_samples_leaf_control = min_samples_leaf_control
                     
        self.root = None
        
    def fit(self,X,treatment,Y):
        '''

        Parameters
        ----------
        X : np.ndarray (n * k)
            Array with the data contained in the root
        Y : np.ndarray (n)
            Array with all target variables
        treatment : np.ndarray (n)
            Array with treatment flags

        '''
        #Create root node which recursively generates subsequent branch nodes         
        self.root = Node(X,Y,treatment,0,self.max_depth,self.min_samples_leaf,
                         self.min_samples_leaf_treated,self.min_samples_leaf_control)
                     
    def predict(self,X):
        '''

        Parameters
        ----------
        X : np.ndarray (n * k)
            Array with the data contained in the root

        Returns
        -------
        Y : np.ndarray (n)
            Array containing the predicted values
        '''
		# predictictions made by recursively finding the leaf node for each row in X and assigning Y according to the mean value of all Y's in that node
        Y = np.zeros(len(X))
        idxs = np.array([True * len(X)])
        def recurse_tree(Y,X,idxs=idxs,node=self.root):
            
            if node.left == None:
                prediction = node.Y.mean()
                return Y + (idxs * prediction)
            else:
                left_idxs = X[:,self.root.split_feature]<self.root.split_threshold
                right_idxs = X[:,self.root.split_feature]>=self.root.split_threshold
        
                return recurse_tree(Y,X,left_idxs,node.left) + recurse_tree(Y,X,right_idxs,node.right)
                
        return recurse_tree(Y,X)
        
