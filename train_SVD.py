####################################################################
######## NOTE : unfortunately we werent able to impliment the SVD++, 
######## here is the code for the BASE SVD
####################################################################

import numpy as np
import pandas as pd

D_train = pd.read_csv("/content/drive/My Drive/Colab Notebooks/D_train.csv")
D_train= D_train.drop(columns=['index'])

# setting the parameters
mu = np.mean(D_train.r_ui)
lr = 0.001
reg_coeif = 0.02
epochs = 15
factor = 10

# getting the list of all the users as well as all the assets
user_set = set(D_train.id_profile)
movies_set = set(D_train.id_asset)

# initializing the weights
b_u_pd = pd.DataFrame(user_set, columns=['id_profile'])
b_u_pd['bu'] = 0.0 
b_i_pd = pd.DataFrame(movies_set,columns=['id_asset'])
b_i_pd['bi'] = 0.0

# we used 2 dataframes to model the vectors Pu and Qi
#initialisation de PU
p_u_pd = pd.DataFrame(user_set,columns=['id_profile'])
p_u_pd['pu'] = p_u_pd.apply(lambda x: np.random.rand(factor).tolist(), axis=1)

#initialisation de Qi
q_i_pd = pd.DataFrame(movies_set,columns=['id_asset'])
q_i_pd['qi'] = q_i_pd.apply(lambda x: np.random.rand(factor).tolist(), axis=1)

##### HELPER FUNCTIONS #######
# function used to calculate the dot product between two columns in datframe(conatins pu and qi vectors)
def dot_pu_qi(x, y):
    return np.dot(np.asarray(x), np.asarray(y))

# calculating all the errors in regard to Pu's and Qi's using this function
def calculate_delta(x, y, z):
    return ( lr * (-2 * z * np.asarray(x) + 2 * reg_coeif * np.asarray(y))).tolist()

# sum the updates of differentes weights
from functools import reduce
def test_sum(series):
  return reduce(lambda x, y: (np.asarray(x) + np.asarray(y)).tolist(), series)

# this function is used to apply the weight update
def sub_vec(x, y):
    return (np.asarray(x) - np.asarray(y)).tolist()

# TRAINING using a MINI BATCH OF 2000000, batch takes around 3 minutes
batch_size = 2000000
n_batchs = len(D_train)/batch_size
mse_track = list()
for i in range(epochs):
	for j in range(1):
		# calculating the current batch
		Batch = D_train.loc[j*batch_size:(j+1)*batch_size-1]

		# adding columns needed for calculating the error
		Batch['err'] = 0.0
		# merge the bu bi pu and qi with batch dataset 
		Batch = Batch.merge(b_u_pd,how='left', left_on=['id_profile'], right_on=['id_profile'])
		Batch = Batch.merge(b_i_pd,how='left', left_on=['id_asset'], right_on=['id_asset'])
		Batch = Batch.merge(p_u_pd,how='left', left_on=['id_profile'], right_on=['id_profile'])
		Batch = Batch.merge(q_i_pd,how='left', left_on=['id_asset'], right_on=['id_asset'])

		# calculating the error
		Batch['err'] = Batch['r_ui'] - Batch['bu'] - Batch['bi'] - mu - Batch[["pu", "qi"]].apply(lambda x : dot_pu_qi(*x), axis=1)

		# calculating the error in regard to the parameters with the use of vectors operation
		Batch['bu'] = 1/batch_size * lr * (-2 * Batch['err'] + 2 * reg_coeif * Batch['bu'])
		Batch['bi'] = 1/batch_size * lr * (-2 *Batch['err'] + 2 * reg_coeif * Batch['bi'])
		#apply fonction calculate_delta to update the weights pu and qi 
		Batch["delta_pu"] = Batch[["qi", "pu", "err"]].apply(lambda x : calculate_delta(*x), axis=1)
		Batch["delta_qi"] = Batch[["pu", "qi", "err"]].apply(lambda x : calculate_delta(*x), axis=1)

		# updating the b_u weights, todoso we create a dataframe containing the error than substract 
		#it with group by function  
		delta_bu = (Batch.groupby(["id_profile"]).bu.sum().reset_index()).rename(columns={"bu": "bu_delta"})
		#merge the b_u_pd dataframe with delta_bu to store our weights 
		b_u_pd = (b_u_pd.merge(delta_bu,how='left', left_on=['id_profile'], right_on=['id_profile'])).fillna(0)
		#applying the update 
		b_u_pd['bu'] -= b_u_pd['bu_delta']
		# drop the column when we finishe with 
		b_u_pd = b_u_pd.drop(columns=['bu_delta'])

		# updating the b_i weights, todoso we create a dataframe containing the error than substract it 
		delta_bi = (Batch.groupby(["id_asset"]).bi.sum().reset_index()).rename(columns={"bi": "bi_delta"})
		#merge the b_u_pd dataframe with delta_bu to store our weights 
		b_i_pd = (b_i_pd.merge(delta_bi,how='left', left_on=['id_asset'], right_on=['id_asset'])).fillna(0)
		#applying the update
		b_i_pd['bi'] -= b_i_pd['bi_delta']
		# drop the column when we finishe with
		b_i_pd = b_i_pd.drop(columns=['bi_delta'])

		# updating the p_u weights, todoso we create a dataframe containing the error than substract it 
		# starte with groupeby the pu of the same user with the defined function "test_sum" which allow to 
		#sum vector contained in colunm 
		delta_pu = Batch[['id_profile', 'delta_pu']].groupby('id_profile').agg({'delta_pu': [test_sum]})
		delta_pu.columns = delta_pu.columns.droplevel(1)
		# align  the update with the corresponding weights vectors "p_u_pd "
		p_u_pd = (p_u_pd.merge(delta_pu,how='left', left_on=['id_profile'], right_on=['id_profile'])).fillna(0)
		#applying the update with our defined function "sub_vec" which allow substract vectors contained in column 
		p_u_pd["pu"] = p_u_pd[["pu", "delta_pu"]].apply(lambda x : sub_vec(*x), axis=1)
		p_u_pd = p_u_pd.drop(columns=['delta_pu'])

		# updating the q_i weights, todoso we create a dataframe containing the error than substract it 
		# starte with groupeby the qi of the same user with the defined function "test_sum" which allow to 
		#sum vector contained in colunm 		
		delta_qi = Batch[['id_asset', 'delta_qi']].groupby('id_asset').agg({'delta_qi': [test_sum]})
		delta_qi.columns = delta_qi.columns.droplevel(1)
		# align  the update with the corresponding weights vectors "q_i_pd "
		q_i_pd = (q_i_pd.merge(delta_qi,how='left', left_on=['id_asset'], right_on=['id_asset'])).fillna(0)
		#applying the update with our defined function "sub_vec" which allow substract vectors contained in column 
		q_i_pd["qi"] = q_i_pd[["qi", "delta_qi"]].apply(lambda x : sub_vec(*x), axis=1)
		q_i_pd = q_i_pd.drop(columns=['delta_qi'])

		#tracking the mse for each epoch 		
		mse_track.append(1/len(Batch) * ((Batch['err']**2).sum()))
	print(mse_track)

# saving the weights
b_u_pd.to_csv(path_or_buf="/content/drive/My Drive/Colab Notebooks/SVD_bu.csv" ,index=False)
b_i_pd.to_csv(path_or_buf="/content/drive/My Drive/Colab Notebooks/SVD_bi.csv" ,index=False)
p_u_pd.to_csv(path_or_buf="/content/drive/My Drive/Colab Notebooks/SVD_pu.csv" ,index=False)
q_i_pd.to_csv(path_or_buf="/content/drive/My Drive/Colab Notebooks/SVD_qi.csv" ,index=False)