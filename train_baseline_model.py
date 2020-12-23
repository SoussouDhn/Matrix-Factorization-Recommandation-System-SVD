# Reading the trainig data set 
import numpy as np
import pandas as pd
D_train = pd.read_csv("/content/drive/My Drive/Colab Notebooks/D_train.csv")
D_train= D_train.drop(columns=['index'])

# getting the list of all the users as well as all the assets
user_set = set(D_train.id_profile)
movies_set = set(D_train.id_asset)

# initializing the weights and the parameters for the training (bi and bu)
b_u_pd = pd.DataFrame(user_set, columns=['id_profile'])
b_u_pd['bu'] = 0.0 

b_i_pd = pd.DataFrame(movies_set, columns=['id_asset'])
b_i_pd['bi'] = 0.0

mu = np.mean(D_train.r_ui)
lr = 0.00001
reg_coeif = 0.02
epochs = 15

D_train['err'] = 0.0		# setting a column to save the trainig error

# list for errors rate
mse_track = list()

# training over 15 epochs, 1 epoch takes around 45 seconds, best obtained mse 
for i in range(0, epochs):
  # Adding the weights to the adjacent lines
  D_train = D_train.merge(b_u_pd,how='left', left_on=['id_profile'], right_on=['id_profile'])
  D_train = D_train.merge(b_i_pd,how='left', left_on=['id_asset'], right_on=['id_asset'])

  # calculating all the errors
  D_train['err'] = D_train['r_ui'] - D_train['bu'] - D_train['bi'] - mu

  # Calculating the update values of the parameters for each line, 
  # they are stored in the columns used initially for the weights (to save memory)
  D_train['bu'] = lr * (-2 * D_train['err'] + 2 * reg_coeif * D_train['bu'])
  D_train['bi'] = lr * (-2 *D_train['err'] + 2 * reg_coeif * D_train['bi'])

  #summing the errors of different parameters than applying the pdate to the weights
  b_u_pd['bu'] -=  D_train.groupby(["id_profile"]).bu.sum().reset_index()['bu']
  b_i_pd['bi'] -=  D_train.groupby(["id_asset"]).bi.sum().reset_index()['bi']

  # dropping the columns since they contain values that are no more in need
  D_train= D_train.drop(columns=['bu','bi'])
  
  # calculating mean squared error
  mse_track.append(1/len(D_train) * ((D_train['err']**2).sum()))

# saving the obtained weights
b_u_pd.to_csv(path_or_buf="/content/drive/My Drive/Colab Notebooks/baseline_bu.csv"
,index=False)
b_i_pd.to_csv(path_or_buf="/content/drive/My Drive/Colab Notebooks/baseline_bi.csv"
,index=False)