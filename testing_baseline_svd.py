import numpy as np
import pandas as pd

##### TEST BASELINE MODEL
def test_baseline():
	# loading test set
	D_test = pd.read_csv("/content/drive/My Drive/Colab Notebooks/D_test.csv")
	b_u_pd = pd.read_csv("/content/drive/My Drive/Colab Notebooks/baseline_bu.csv")	#loading weights
	b_i_pd = pd.read_csv("/content/drive/My Drive/Colab Notebooks/baseline_bi.csv")	#loading weights

	# calculating mu, setting up error column, associating weights to lines
	mu = np.mean(D_test.r_ui)

	# this column contains the error for all the columns
	D_test['err'] = 0.0

	# associating the weighs to the line
	D_test = D_test.merge(b_u_pd,how='left', left_on=['id_profile'], right_on=['id_profile'])
	D_test = D_test.merge(b_i_pd,how='left', left_on=['id_asset'], right_on=['id_asset'])

	# calculating the error for all the 22M entries
	D_test['err'] = D_test['r_ui'] - D_test['bu'] - D_test['bi'] - mu

	# MSE: 0.55
	print(1/len(D_test) * ((D_test['err']**2).sum()))

# will be used to test svd
def dot_pu_qi(x, y):
	    return np.dot(np.asarray(x), np.asarray(y))

def test_svd():
	# loading test set
	D_test = pd.read_csv("/content/drive/My Drive/Colab Notebooks/D_test.csv")
	b_u_pd = pd.read_csv("/content/drive/My Drive/Colab Notebooks/SVD_bu.csv")	#loading weights
	b_i_pd = pd.read_csv("/content/drive/My Drive/Colab Notebooks/SVD_bi.csv")	#loading weights
	p_u_pd = pd.read_csv("/content/drive/My Drive/Colab Notebooks/SVD_bu.csv")	#loading weights
	q_i_pd = pd.read_csv("/content/drive/My Drive/Colab Notebooks/SVD_bi.csv")	#loading weights

	# calculating mu, setting up error column, associating weights to lines
	mu = np.mean(D_test.r_ui)
	#initiali
	D_test = D_test.merge(b_u_pd,how='left', left_on=['id_profile'], right_on=['id_profile'])
	D_test = D_test.merge(b_i_pd,how='left', left_on=['id_asset'], right_on=['id_asset'])
	D_test = D_test.merge(p_u_pd,how='left', left_on=['id_profile'], right_on=['id_profile'])
	D_test = D_test.merge(q_i_pd,how='left', left_on=['id_asset'], right_on=['id_asset'])

	# calculating the error
	D_test['err'] = D_test['r_ui'] - D_test['bu'] - D_test['bi'] - mu - D_test[["pu", "qi"]].apply(lambda x : dot_pu_qi(*x), axis=1)
	# MSE: 2.11
	print(1/len(D_test) * ((D_test['err']**2).sum()))