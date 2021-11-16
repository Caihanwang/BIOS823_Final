# Raw Data

1. **Two Folders: Daily_Cases & Daily_Deaths:**  
  Public COVID-19 trend data from CDC: https://covid.cdc.gov/covid-data-tracker/#datatracker-home  
2. **people_vaccinated_us_timeline.csv:**  
  Vaccination Data from JHU: https://coronavirus.jhu.edu/vaccines/us-states

## Data Cleaning Progress
* 2021-11-15:  
**Caihan:** Combine two folders (Daily_Cases and Daily_Deaths) to create "Deaths_Cases.csv". [Code](https://github.com/Caihanwang/BIOS823_Final/blob/main/Scripts/Preprocessing/Combine_State_data.ipynb)  
**Yifeng:** Combine "people_vaccined_us_timeline.csv"(Vaccine data) with "Deaths_Cases.csv"[positive cases and death cases data] to new table, named: "Deaths_Cases_Vaccine.csv". [Code](https://github.com/Caihanwang/BIOS823_Final/blob/main/Scripts/Preprocessing/Merge_Vaccination.py)
