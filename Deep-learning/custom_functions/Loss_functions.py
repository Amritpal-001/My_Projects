import numpy as np


## MSE 
def mean_square_error(x , y ):
  '''
  ### Implementation 1
  inputs:
  x = actual  ( 1*n shape)
  y = prediction( 1*n shape)
  
  returns:
  MSE loss: Float 
  '''
  diff = y - x
  diff_squared = diff ** 2
  mean_diff = diff_squared.mean()
  return(mean_diff)
  

## RMSE
def root_mean_squared_error(x , y ):
  '''
  ### Implementaion 1
  inputs:
  x = actual  ( 1*n shape)
  y = prediction( 1*n shape)
  
  returns:
  RMSE loss: Float
  '''
  diff = y - x
  diff_squared = diff ** 2
  mean_diff = diff_squared.mean()
  rmse = np.sqrt(mean_diff)
  
  return(rmse)


## RMSLE
### Implementaion 1




## MAE
def mean_absolute_error(x , y):
  '''
  ### Implementaion 1
  inputs:
  x = actual  ( 1*n shape)
  y = prediction( 1*n shape)
 
  returns:
  MAE loss: Float
  '''
  diff = y - x
  abs_diff = np.absolute(diff)
  mean_diff = abs_diff.mean()
  return(mean_diff)
