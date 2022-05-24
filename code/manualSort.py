# %%
import pandas  as pd
# %%

test_datafile = '../Dataset/VU_DM_data/new_filled_test_set_VU_DM.csv'

df = pd.read_csv(test_datafile, index_col=0)
# %%
df = df.sort_values(by = ['srch_id', 'booking_bool', 'click_bool', 'total_comp_rate', 'prop_starrating','prop_review_score','price_usd'],
                    ascending=[True, False, False, False, False, False, True ])
# %%
df = df[['srch_id', 'prop_id']]
df.to_csv('../Dataset/VU_DM_data/submission4.csv', index = False)
# %%
