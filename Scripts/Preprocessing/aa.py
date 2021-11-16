import numpy as np
import pandas as pd

main = pd.read_csv("COVID.csv")
vacine = pd.read_csv("people_vaccinated_us_timeline.csv")


# select the subset from vaccine table
vacine_reduce = vacine[["Province_State", "Date", "People_Fully_Vaccinated"]]
vacine_reduce = vacine_reduce.rename(columns={"Province_State": "State"})

# get the states list used in model:
final_d = pd.read_csv("final_data.csv")
final_d
state_list = list(set(final_d["State"]))
state_list = sorted(state_list)

#select the subset from main based on states:
main = main[main["State"].isin(state_list)]

#merge the main and vaccine table
final_table = main.merge(vacine_reduce, on=["State", "Date"], how='left')
final_table = final_table[["State", "Date", "New Cases", "New Deaths", "People_Fully_Vaccinated"]]

# Save it
final_table.to_csv(r'final_covid.csv')
