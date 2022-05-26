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
