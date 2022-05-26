# %%
import pandas  as pd
test_datafile = '../Dataset/VU_DM_data/new_filled_test_set_VU_DM.csv'
df = pd.read_csv(test_datafile)
df = df.sort_values(by = ['query_id','rank','prev_books','prev_click', 'prop_starrating','prop_review_score','price_usd', 'total_comp_rate'],
                    ascending=[False, False, False, False, False, False, True, False])
df = df[['query_id', 'prop_id']]
df.columns = df.columns.str.replace('query_id', 'srch_id')
df.to_csv('../Dataset/VU_DM_data/submission11.csv', index = False)
# %%
