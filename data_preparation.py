import numpy as np
import pandas as pd

# loading first table
bookmarks=pd.read_csv("/content/drive/My Drive/Colab Notebooks/bookmarks.csv")
bookmarks.time = 1	# we set this column to one, because we are gonna use it later in calculating interest

# loading second table
favorites=pd.read_csv("/content/drive/My Drive/Colab Notebooks/favorites.csv")
favorites.added_date = 5 # we set this column to 5, because we are gonna use it later in calculating interest

#loading the third table
ratings=pd.read_csv("/content/drive/My Drive/Colab Notebooks/ratings.csv")
ratings=ratings.drop(columns=['time'])	# we drop this column, cause it has no use to us

# with this, we merge the two tables, adding the scores +5 of favorite to the adgecent lines
bookmarks = bookmarks.merge(favorites,how='left', 
			left_on=['id_profile','id_asset'], right_on=['id_profile','id_asset'])

# with this, we merge the two tables, adding the ratings of movies to the adgecent lines
bookmarks = bookmarks.merge(ratings,how='left', 
			left_on=['id_profile','id_asset'], right_on=['id_profile','id_asset'])

# filling al he na values with 0, to facilitate the calculation
bookmarks=bookmarks.fillna(0)
# renaming so it would make beter sense
bookmarks=bookmarks.rename(columns={"time": "m_ui", "added_date": "f_ui","score":"n_ui"})

# calculating the interest
bookmarks.m_ui += bookmarks.f_ui + bookmarks.n_ui

# droping the coumns that arent usefull anymore
bookmarks = bookmarks.drop(columns=['f_ui',	'n_ui']).rename(columns={"m_ui": "r_ui"})

# saving
bookmarks.to_csv(path_or_buf="/content/drive/My Drive/Colab Notebooks/D.csv"
,index=False)

# loading the D matrix
D = pd.read_csv("/content/drive/My Drive/Colab Notebooks/D.csv")

# loading indexes
train_idx = np.load("/content/drive/My Drive/Colab Notebooks/bookmarks_idx_train.npy")
test_idx = np.load("/content/drive/My Drive/Colab Notebooks/bookmarks_idx_test.npy")


# partitioning and saving
D_train = D.loc[train_idx]
D_train.reset_index(inplace=True)
D_train.to_csv(path_or_buf="/content/drive/My Drive/Colab Notebooks/D_train.csv"
,index=False)

D_test = D.loc[test_idx]
D_test.reset_index(inplace=True)
D_train.to_csv(path_or_buf="/content/drive/My Drive/Colab Notebooks/D_test.csv"
,index=False)
