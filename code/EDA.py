import pandas as pd 
import numpy as np 
import csv
import seaborn as sns
import matplotlib as plt
%matplotlib inline


hotel_data = pd.read_csv('training_set_VU_DM.csv')

booked = hotel_data[hotel_data["booking_bool"] == 1]
booked.count()

country_id = booked['prop_country_id']
#country_id
country_id.value_counts()

sns.countplot('prop_country_id',data=booked.sort_values(by=['prop_country_id']))



## check stats for max booked hotel destination 

max_prop_booked = 219 

country_customers = booked[booked["prop_country_id"] == max_prop_booked]

country_customers["visitor_location_country_id"].value_counts().plot(kind='pie',autopct='%1.1f%%')

## stats for independent hotels and chains 

chain_ind = booked['prop_brand_bool']
count_row = booked.shape[0]
print(count_row)
chain_ind.value_counts().plot(kind='pie',autopct='%1.1f%%')
chain_ind.value_counts()

