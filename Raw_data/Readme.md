# Raw Data

1. **Two Folders: Daily_Cases & Daily_Deaths:**  
  Public COVID-19 trend data from CDC: https://covid.cdc.gov/covid-data-tracker/#datatracker-home  
2. **people_vaccinated_us_timeline.csv:**  
  Vaccination Data from JHU: https://coronavirus.jhu.edu/vaccines/us-states  
3. **COVID-19 US state policy database 9_20_2021.xlsx:**
  State policy data from UMICH: https://www.openicpsr.org/openicpsr/project/119446/version/V129/view?path=/openicpsr/119446/fcr:versions/V129/COVID-19-US-State-Policy-Database-master/COVID-19-US-state-policy-database-9_20_2021.xlsx&type=file  
4. **State character data** from: https://datacommons.org/  


## Data Cleaning Progress
* 2021-11-15:  
**Caihan:** Combine two folders (Daily_Cases and Daily_Deaths) to create "Deaths_Cases.csv". [Code](https://github.com/Caihanwang/BIOS823_Final/blob/main/Scripts/Preprocessing/Combine_State_data.ipynb)  
**Yifeng:** Combine "people_vaccined_us_timeline.csv"(Vaccine data) with "Deaths_Cases.csv"[positive cases and death cases data] to new table, named: "Deaths_Cases_Vaccine.csv". [Code](https://github.com/Caihanwang/BIOS823_Final/blob/main/Scripts/Preprocessing/Merge_Vaccination.py)  

* 2021-11-16:  
**Yuxuan:** Merge "Deaths_Cases_Vaccine.csv" with state policy database and state character data and create "final_data.csv"

