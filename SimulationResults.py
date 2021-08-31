#**Importing libraries and packages used in the program - if running a copy of this script to have several dashboards open from different files, remember to change the port**
from datetime import date
import numpy as np #version 1.21.2
import pandas as pd #version 1.3.2

#pip install openpyxl version 3.0.7
#from jupyter_dash import JupyterDash #Version 0.4.0

import dash #Version 1.21.0
import dash_core_components as dcc
import dash_html_components as html
import dash_table
from dash.dependencies import State, Input, Output
import dash_bootstrap_components as dbc #0.13.0

import plotly.express as px #Version 0.4.1
#import plotly.offline as pyo
import plotly.graph_objects as go #version 5.3.0
from plotly.subplots import make_subplots

import re #importing this allows to split stings with multiple delimeters in python

port = '8050'
#from ipywidgets import widgets


### File Reading
#**Dataframe from excel file, printing sheet names**
#creating dataframe - may take time due to its large size, sheet names will be printet once it is finished
import  tkinter as tk
from tkinter import filedialog
root = tk.Tk()
root.withdraw()
root.wm_attributes('-topmost', True)
file_path = filedialog.askopenfilename(parent=root)
print(file_path)
#file_path = "V1SimpleCharging.xlsx"
if file_path == '':
    file_path = r"C:\Users\kric\Syddansk Universitet\Zheng Ma - PhD projects-Distributed AI for Energy Business Ecosystem\PhD-Kristoffer Christensen\Dynamic tariff evaluation\Simulation results\V1SimpleCharging.xlsx"
    print('Failed to open file, default file implemented in script is used: '+str(file_path))
df = pd.ExcelFile(file_path)
print(df.sheet_names)
filename = re.split('\\\\|/|\.', file_path)
filename = filename[len(filename)-2]

#**Assigning sheets to individual dataframes and converting time coulumns to pandas datetime type. Also modifiyng times to zero seconds as this can affect the filtering later on**
#defining each sheet to a variable
parameters = df.parse('Parameters', header=0)
measurements = df.parse('Measurements', header=0)
measurements['Timestamp'] = pd.to_datetime(measurements['Timestamp'])
measurements['Timestamp']= measurements['Timestamp'] - pd.to_timedelta(measurements['Timestamp'].dt.second, unit='s') #minus with timedelta makes sure seconds are 00
householdData = df.parse('Household data', header=None)
electricityPrices = df.parse('Electricity prices', header=0)
electricityPrices['Timestamp'] = pd.to_datetime(electricityPrices['Timestamp'])
electricityPrices['Timestamp'] = electricityPrices['Timestamp'] - pd.to_timedelta(electricityPrices['Timestamp'].dt.second, unit='s')  #This is time consuming consider finding another solution for removing seconds or make them zero
DSOIncome = df.parse('Economy', header=0)
evModels = df.parse('EV Models', header=0)

#Convert timestamp to pandas datetime
householdData_timestamp = pd.to_datetime(householdData.iloc[2:,1])
householdData_timestamp = householdData_timestamp - pd.to_timedelta(householdData_timestamp.dt.second, unit='s')
#print(householdData_timestamp)

#Setting EV models as index
EVmodels_modelIndex = evModels.copy()
EVmodels_modelIndex.set_index('EV model', inplace=True)

#### Calculations and modifications for visualization in dashboard
#**Calculate total electricity price and add it to electricityPrices dataframe (this does not include the total, to save computation during simulation)**
#Calculation of total elPrice and add it to dataframe
index = 0
total_price = []
for price in electricityPrices['Spot Price [DKK/kWh]']:
    DSO_price = electricityPrices['DSO Tariff [DKK/kWh]'].iloc[index]
    TSO_price = electricityPrices['TSO Tariff [DKK/kWh]'].iloc[index]
    total_price.append(price+DSO_price+TSO_price)
    index += 1
electricityPrices['Total price [DKK/kWh]'] = total_price

#**Identify first hour of grid overload creating dataframes with the whole day and of the first hour**
#Hour of first overload
grid_cap = parameters.iloc[26,1]

overloads_bool = measurements['Total grid load'] > grid_cap
overloads_df = measurements[overloads_bool]
if overloads_df.empty:
    print("No overload occurs, last date in dataframe is returned")
    first_overload = measurements.loc[len(measurements)-1]
    overload_day = measurements[(measurements['Year']==first_overload['Timestamp'].year)&(measurements['Month']==first_overload['Timestamp'].month-1)&(measurements['Day']==first_overload['Timestamp'].day)]
    overload_day_spotPrice = electricityPrices[(electricityPrices['Timestamp'] >= overload_day['Timestamp'].iloc[0]) & (electricityPrices['Timestamp'] <= overload_day['Timestamp'].iloc[len(overload_day['Timestamp'])-1])]
else:
    overloads_sorted = overloads_df.sort_values(by="Timestamp")
    first_overload = overloads_sorted.iloc[0]

    overload_day = measurements[(measurements['Year']==first_overload['Timestamp'].year)&(measurements['Month']==first_overload['Timestamp'].month-1)&(measurements['Day']==first_overload['Timestamp'].day)] #minus 1 at month is because Month in the dataset starts from 0 where day starts from 1

    #Overload day for spot prices
    overload_day_spotPrice = electricityPrices[(electricityPrices['Timestamp'] >= overload_day['Timestamp'].iloc[0]) & (electricityPrices['Timestamp'] <= overload_day['Timestamp'].iloc[len(overload_day['Timestamp'])-1])]

timeFirstYearAfterOverload = first_overload['Timestamp'].replace(year = first_overload['Timestamp'].year+1)
print(first_overload['Timestamp'])

#**Driving distance dataframe creation**
#number of columns:
first_row_of_columns = householdData.iloc[0]
number_columns = len(first_row_of_columns)
#number of households
number_of_households = 0
index_count = 0
household_index_list = []
for column in first_row_of_columns:
    if str(column) != "nan":
        number_of_households += 1
        household_index_list.append(index_count)
    index_count += 1

df_DrivingDistanceDataFile = pd.DataFrame()
df_DrivingDistanceDataFile['Time'] = measurements['Timestamp']
counter = 0
while len(household_index_list) > counter:
    df_DrivingDistanceDataFile['domesticConsumers['+str(counter)+']'] = householdData.iloc[2:,household_index_list[counter]+4]
    df_DrivingDistanceDataFile['domesticConsumers['+str(counter)+']'].fillna(0, inplace=True)
    df_DrivingDistanceDataFile['domesticConsumers['+str(counter)+']'] = pd.to_numeric(df_DrivingDistanceDataFile['domesticConsumers['+str(counter)+']'])
    counter += 1

#**Identify all consumers with EVs and create a list with their dataframe index**
#Households having EVs
list_of_EVs = []
index_list_of_households_having_EVs = []
index_list_of_households_having_EVs_for_whole_simulation = []
households_EV_type_status = householdData.iloc[2,household_index_list]
index_count = 0
for household_EV in households_EV_type_status:
    if str(household_EV) != "no EV":
        index_list_of_households_having_EVs_for_whole_simulation.append(household_index_list[index_count])
        contain_only_zeros_bool_list = df_DrivingDistanceDataFile[
            first_row_of_columns[household_index_list[index_count]]].groupby(
            level=0).all()  # Creating list of booleans, true if not zero value and fals if it's the case
        contain_only_zeros_bool_list1 = contain_only_zeros_bool_list[
            contain_only_zeros_bool_list == True]  # Create list of only true booleans
        EV_adoption_date = df_DrivingDistanceDataFile['Time'][contain_only_zeros_bool_list1.index[0]]
        if EV_adoption_date < timeFirstYearAfterOverload:
            list_of_EVs.append(str(household_EV))
            index_list_of_households_having_EVs.append(household_index_list[index_count])
    index_count += 1

number_of_EVs = len(list_of_EVs)

#**Dataframe for charging strategy statistics and creation of lists of SoC and EV_state indicies**
#householdData.iloc[:,index_list_of_households_having_EVs]
index_for_checking_strategy = [x+1 for x in index_list_of_households_having_EVs]
#householdData.iloc[2,index_for_checking_strategy]
list_for_dataframe = []
index_count = 0
for household in householdData.iloc[0,index_list_of_households_having_EVs]:
    list_for_dataframe.append([household, householdData.iloc[2,index_for_checking_strategy[index_count]]])
    index_count += 1
df_strategyList = pd.DataFrame(list_for_dataframe, columns=['Household', 'Charging strategy'])
#print(df_strategyList)

df_strategy_counts = pd.DataFrame(df_strategyList['Charging strategy'].value_counts())
charging_strategies_list = df_strategyList['Charging strategy'].unique()
statistics_list = []
for x in charging_strategies_list:
    statistics_list.append([x, df_strategy_counts.loc[x][0]])
df_strategy_statistics = pd.DataFrame(statistics_list, columns=['Charging strategy', 'Number of adopted strategies'])
#print(df_strategy_statistics)

#List of SOC, EV_state and household/consumer indecies in dataframe
SoC_index_list = [x+3 for x in index_list_of_households_having_EVs_for_whole_simulation]
EV_state_index_list = [x+2 for x in index_list_of_households_having_EVs_for_whole_simulation]

#Dataframe of consumer indicies with the consumer name as index (only consumers having EVs)
householdList_indices = pd.DataFrame(householdData.loc[0][index_list_of_households_having_EVs_for_whole_simulation])
householdList_indices = householdList_indices.assign(index = index_list_of_households_having_EVs_for_whole_simulation)
householdList_indices = householdList_indices.set_index(0)

#**Multiindexing household dataframe. Reading directly from excel sheet is not straight forward as the consumer name column each has several sub-columns. Multiindixing solves this problem**
#Multiindexing houshold data: household[0] header with [EV state, SoC] as subheaders
header = pd.MultiIndex.from_product([householdData.iloc[0,index_list_of_households_having_EVs_for_whole_simulation], ['EV state','SoC','Driving distance','Domestic consumption']],names=['household','header']) #[householdData.iloc[0,index_list_of_households_having_EVs]
household_df = pd.DataFrame(householdData_timestamp)
header = header.insert(0,'Time')
#print(header)
headerList = ['Time']
for column in EV_state_index_list:
    household_df = household_df.join(householdData.iloc[2:,column]) #+str(householdData.iloc[0,column-2])
    household_df = household_df.join(householdData.iloc[2:,column+1])
    household_df = household_df.join(householdData.iloc[2:,column+2])
    household_df = household_df.join(householdData.iloc[2:,column+3])
    headerList.append(str(householdData.iloc[0,column-2]+' EV state'))
    headerList.append(str(householdData.iloc[0,column-2]+' SoC'))
    headerList.append(str(householdData.iloc[0,column-2]+' Driving distance'))
    #headerList.append(str(householdData.iloc[0,column-2]+'Domestic consumption'))

household_df.columns=header
household_df.reset_index(inplace=True, drop=True)

#**Creating name list for dashboard dropdown menu. From measurement dataframe**
# Creating list of column headers (measurements)
measurements_headers_with_time = list(measurements.columns.values)
i = 0
while i<7:
    measurements_headers_with_time.pop(0)
    i += 1
measurements_headers = measurements_headers_with_time
print(measurements_headers)

#**Creating name list for dashboard dropdown menu for electricity price information**
# Creating list of column headers (electricity prices)
el_prices_headers_with_time = list(electricityPrices.columns.values)
i = 0
while i<7:
    el_prices_headers_with_time.pop(0)
    i += 1
el_prices_headers = el_prices_headers_with_time
print(el_prices_headers)

#**Creating dataframe of the EV types and the number of each type for statistical visualization**
# Creating dataframe of EV types

df_EV_list = pd.DataFrame(list_of_EVs)
df_EV_counts = pd.DataFrame(df_EV_list[0].value_counts())
df_EV_types = pd.DataFrame(df_EV_list[0].unique())
# print(df_EV_types)
# print(df_EV_counts)
# print([df_EV_types.iloc[0,0], df_EV_counts.loc[df_EV_types.iloc[0,0]]])
# print(df_EV_counts.loc[df_EV_types.iloc[0,0]][0])
EV_types_count = []
for EV in df_EV_types[0]:
    EV_types_count.append([str(EV), df_EV_counts.loc[str(EV)][0], EVmodels_modelIndex.loc[str(EV)][0],
                           EVmodels_modelIndex.loc[str(EV)][1], EVmodels_modelIndex.loc[str(EV)][2]])

df_EV_types_count = pd.DataFrame(EV_types_count,
                                 columns=['Model', 'Number of models', 'Battery capacity [kWh]', 'Mileage [kWh/km]', 'Charging power [kW]'])

#**Creating list for dashboard slider including a bullet at the first overload**
# Slider list for years
slider_list_values_int = measurements['Year'].unique()

slider_list_labels = np.array([], dtype='object')

slider_list_values = []
for item in slider_list_values_int:
    slider_list_values.append(float(item) + 0.01)
    slider_list_labels = np.append(slider_list_labels, str(int(item))) #int is to remove point

monthNumberOfOverloadDate = overload_day['Timestamp'].iloc[0].month
# if statement to avoid that the point for first overload is too close to the year's point
if monthNumberOfOverloadDate in (1, 2, 3):
    monthNumberOfOverloadDate = 4
elif monthNumberOfOverloadDate in (10, 11, 12):
    monthNumberOfOverloadDate = 9
slider_location_of_overload = (monthNumberOfOverloadDate / 12) + overload_day['Timestamp'].iloc[0].year

index = 0
for x in slider_list_values:
    if slider_location_of_overload > x and slider_location_of_overload < slider_list_values[index + 1]:
        # print(index)
        # print(slider_location_of_overload)
        slider_list_values = np.insert(slider_list_values, index + 1, slider_location_of_overload)
        slider_list_labels = np.insert(slider_list_labels, index + 1, "1st_overload")
    if slider_location_of_overload + 1 > x and slider_location_of_overload + 1 < slider_list_values[index + 2] and \
            slider_list_values_int[len(slider_list_values_int) - 1] > slider_location_of_overload + 1:
        slider_list_values = np.insert(slider_list_values, index + 2, slider_location_of_overload + 1)
        slider_list_labels = np.insert(slider_list_labels, index + 2, "1_year_after")
    index += 1

#**Statistics for overloads - counting number of overloads, creating dataframe including all hours with overload, adding size og overload**
#Counting number of overloads
listOfOverloads = measurements[(measurements['Timestamp'] <= timeFirstYearAfterOverload) & (measurements['Total Maximum grid load'] > grid_cap)] #remove this part: "(measurements['Timestamp'] <= timeFirstYearAfterOverload) &" if you want number of overloads for the whole simulation
numberOfOverloads = len(listOfOverloads['Total Maximum grid load'])

#Removing irrelevant columns for dashboard
del listOfOverloads['Passed hours']
del listOfOverloads['Year']
del listOfOverloads['Month']
del listOfOverloads['Day']
del listOfOverloads['Hour']
del listOfOverloads['Minute']
del listOfOverloads['Aggregated base load']

#Add column with size of overload
overload_size_list = []
for load in listOfOverloads['Total Maximum grid load']:
    overload_size_list.append(load-grid_cap)

listOfOverloads = pd.DataFrame(listOfOverloads)
listOfOverloads['Size of overload [kWh]'] = overload_size_list

#**Driving distance function - identifying the driving distance used before and after the charge**
#household_df['Driving distance']
def driving_dist(consumer, date, before_or_next): #before_or_next selects the driving distance for either the trip before or after charging. It can be relevant to know both.
    date = pd.Timestamp(str(date)).date()
    date_after = date + pd.DateOffset(1)
    date_two_days_after = date + pd.DateOffset(2)
    if before_or_next == 'before':
        consumer_distance = household_df[(household_df['Time'] >= str(date))&(household_df['Time'] < str(date_after.date()))][consumer]['Driving distance']
        consumer_distance = consumer_distance[consumer_distance != '0']
        return consumer_distance
    else:
        consumer_distance = household_df[(household_df['Time'] >= str(date_after.date()))&(household_df['Time'] < str(date_two_days_after.date()))][consumer]['Driving distance']
        consumer_distance = consumer_distance[consumer_distance != '0']
        return consumer_distance

#**Function that creates a dataframe of all consumers charging in a chosen hour including their EV type, charging power and charging strategy**
#overload dataframe function
household_cap = parameters.iloc[27,1]
def overload_info(date):
    listOfChargingConsumers = []
    index_of_consumers = []
    consumers_EVModel = []
    consumers_strategy = []
    chargingPowers = []
    consumer_distance_before_list = []
    consumer_distance_after_list = []
    for consumerName in householdData.iloc[0,index_list_of_households_having_EVs]:
        EV_state_in_given_date = household_df[household_df['Time'] == pd.Timestamp(str(date))][consumerName]['EV state']
        if EV_state_in_given_date.iloc[0] == 'Charging':
            listOfChargingConsumers.append(consumerName)
    for chargingConsumer in listOfChargingConsumers:
        index_of_consumer = householdList_indices.loc[chargingConsumer][0]
        index_of_consumers.append(index_of_consumer)
        consumer_EVModel = householdData.iloc[2,index_of_consumer]
        consumers_EVModel.append(consumer_EVModel)
        consumers_strategy.append(householdData.iloc[2,index_of_consumer+1])
        if household_cap < EVmodels_modelIndex.loc[consumer_EVModel]['Charging power [kW]']:
            chargingPowers.append(household_cap)
        else:
            chargingPowers.append(EVmodels_modelIndex.loc[consumer_EVModel]['Charging power [kW]'])
        #Here it should be implemented the driving distance related to the charge
        consumer_distance_before = driving_dist(chargingConsumer, date, 'before')
        consumer_distance_before_list.append(consumer_distance_before)
        consumer_distance_after = driving_dist(chargingConsumer, date, 'next')
        consumer_distance_after_list.append(consumer_distance_after)
    return pd.DataFrame({'Consumer Name': listOfChargingConsumers, 'EV model': consumers_EVModel, 'Charging power [kW]': chargingPowers, 'Charging strategy': consumers_strategy, 'Driving distance BEFORE charge': consumer_distance_before_list, 'Driving distance AFTER charge': consumer_distance_after_list})
#date = '2029-12-19T16:00:00'
#df_overloadInfo = overload_info(date)
#df_overloadInfo

#**Creating dataframe for parameter table in dashboard**
df_parameter = pd.DataFrame()
parameterList = []
parameterValueList = []
index = 0
while index < len(parameters['Name']):
    currentName = parameters.iloc[index][0]
    currentValue = parameters.iloc[index][1]
    currentUnit = parameters.iloc[index][2]
    if not pd.isna(currentValue):
        parameterList.append(currentName)
        if not pd.isna(currentUnit):
            parameterValueList.append(str(currentValue)+' '+str(currentUnit))
        else:
            parameterValueList.append(str(currentValue))
    index += 1

df_parameter['Parameter'] = parameterList
df_parameter['Value'] = parameterValueList
pd.set_option('max_colwidth', 75)
#df_parameter

#**Dataframe individual consumer consumption, read directly from dataset if none is available**
consumptionColumn = householdData.iloc[2:, 12]
if consumptionColumn.isnull().values.any():  # Check if any number in the row is a NaN. This is done for just the first consumer as the simulation either outputs hourly consumption for all consumers or none.
    ConsumptionDataFile = "137consumerDataset.xlsx" #r"C:\Users\kric\Syddansk Universitet\Zheng Ma - FED project\Data\137consumerDataset.xlsx"  # If the simulation does not output consumer's consumption, then the dataset is looped through all times

    consumption_headers = householdData.iloc[0, household_index_list]
    consumption_headers = pd.concat([pd.Series(['Time']), consumption_headers])
    df_ConsumptionDataFile = pd.ExcelFile(ConsumptionDataFile).parse('ConsumptionData', names=consumption_headers)
    while len(measurements['Timestamp']) > len(df_ConsumptionDataFile['Time']):
        df_ConsumptionDataFileCopy = df_ConsumptionDataFile.copy()
        df_ConsumptionDataFile = df_ConsumptionDataFile.append(df_ConsumptionDataFileCopy, ignore_index=True)

    df_ConsumptionDataFile = df_ConsumptionDataFile.drop(
        df_ConsumptionDataFile.index[len(measurements['Timestamp']):len(df_ConsumptionDataFile['Time'])])

    df_ConsumptionDataFile['Time'] = measurements['Timestamp']  # Matching the times with the times from the simulation

    # Creating list of column headers (consumption dataset, hence the consumer names are different from simulation output names e.g. domesticConsumers[0] and Consumer_1)
    # consumption_headers_with_time = householdData.iloc[0,household_index_list]#list(df_ConsumptionDataFile.columns.values)
    consumption_headers.pop(0)
    # consumption_headers = consumption_headers_with_time
else:
    # Else the simulation output data is visualized
    df_ConsumptionDataFile = pd.DataFrame()
    df_ConsumptionDataFile['Time'] = measurements['Timestamp']
    counter = 0
    while len(household_index_list) > counter:
        df_ConsumptionDataFile['domesticConsumers[' + str(counter) + ']'] = householdData.iloc[2:,
                                                                            household_index_list[counter] + 5]
        counter += 1

    # Creating list of column headers (consumption dataset, hence the consumer names are different from simulation output names e.g. domesticConsumers[0] and Consumer_1)
    consumption_headers_with_time = list(df_ConsumptionDataFile.columns.values)
    consumption_headers_with_time.pop(0)
    consumption_headers = consumption_headers_with_time

#**dataframe of EV type and strategy for consumer**
df_consumerInfo = pd.DataFrame()
counter = 0
while len(household_index_list) > counter:
    info_list = [householdData.iloc[2,household_index_list[counter]], householdData.iloc[2,household_index_list[counter]+1]]
    df_consumerInfo['domesticConsumers['+str(counter)+']'] = info_list
    counter += 1
#df_consumerInfo

#**Function calculating average charging cost per kWh charging for the consumer and total electricity bill**
# Average charging cost per kWh function for a defined period.
# consumption calculations are removed as only the average electricity price in charging hours are found. However, consumption calculation can quickly be implemented again.
def avg_chargingCost_per_kWh(date_day1, date_day2,
                             consumerName):  # date_day should be of format: '2030-10-01 HH' Note: The end date is not included, hence if you want one day you have to type the day you want and the day after
    date_day1 = pd.to_datetime(date_day1)
    date_day2 = pd.to_datetime(date_day2)
    # Find charging power
    index_of_consumer = householdList_indices.loc[consumerName][0]
    consumers_EVModel = householdData.iloc[2, index_of_consumer]
    chargingPower = EVmodels_modelIndex.loc[consumers_EVModel]['Charging power [kW]']
    # Find battery capacity
    Cap = EVmodels_modelIndex.loc[consumers_EVModel]['Battery capacity [kWh]']
    # Following three lines results in a list of times and SoC of the specific consumer (the times are only for charging states)
    individual_charging_Soc_for_chargingStateOnly = pd.DataFrame(
        household_df[household_df[consumerName]['EV state'] == 'Charging'][consumerName]['SoC'])
    individual_charging_Soc_for_chargingStateOnly_times = pd.DataFrame(
        household_df[household_df[consumerName]['EV state'] == 'Charging']['Time'])
    SoC_df_to_time = individual_charging_Soc_for_chargingStateOnly_times.join(
        individual_charging_Soc_for_chargingStateOnly)
    # sorting only chosen date
    # tomorrow_date =  date_day + pd.Timedelta(days=1) #Used when only one day is wanted and day2 is removed
    df_for_chosen_day = SoC_df_to_time[
        (SoC_df_to_time['Time'] > date_day1) & (SoC_df_to_time['Time'] < date_day2)]  # Filter inside the chosen dates
    # return None if list is empty
    if df_for_chosen_day.empty:
        return None
    # Calculating charging efficiency (Delta_SoC*Cap)/Charging_power. In special situations where SoC list is shorter than 2 e.g. charging from 99% to 100% a error in efficiency calculation will occur. Hence, the provided efficiency is used. If eff at some point changes between models, these needs to be provided in model info
    eff = 0.84  # efficiency is 84% for all EV models
    # calculating SoC increase per hour (chargingPower * eff)/Cap
    Delta_SoC_calc = (chargingPower * eff) / Cap
    # converting SoC to consumption (Delta_SoC*Cap)/eff
    i = 0
    Total_cost_for_charge_in_period = 0
    Total_charging_consumption_in_period = 0
    Total_cost_regular_consumption_in_period = 0
    while i < len(df_for_chosen_day):
        Delta_SoC = Delta_SoC_calc
        # When the EV fully charge, delta_SoC can be smaller.
        # if 1 - float(df_for_chosen_day.iloc[i,1]) < Delta_SoC:
        # Delta_SoC = 1 - float(df_for_chosen_day.iloc[i,1])
        # print("delta SoC: "+str(Delta_SoC))
        Time_hour = pd.to_datetime(df_for_chosen_day.iloc[i, 0])
        consumption = (Delta_SoC * Cap) / eff
        # print("consumption: "+str(consumption))
        # Find total el price in the hour
        spotPrice_in_hour = electricityPrices[electricityPrices['Timestamp'] == Time_hour]['Spot Price [DKK/kWh]'].iloc[
            0]
        DSO_in_hour = electricityPrices[electricityPrices['Timestamp'] == Time_hour]['DSO Tariff [DKK/kWh]'].iloc[0]
        TSO_in_hour = electricityPrices[electricityPrices['Timestamp'] == Time_hour]['TSO Tariff [DKK/kWh]'].iloc[0]
        Total_price_in_hour = spotPrice_in_hour + DSO_in_hour + TSO_in_hour

        regular_consumption_in_hour = \
        df_ConsumptionDataFile[df_ConsumptionDataFile['Time'] == Time_hour][consumerName].iloc[0]
        Total_cost_regular_consumption_in_period += regular_consumption_in_hour * Total_price_in_hour

        # print("Total el price: "+str(Total_price_in_hour))
        # calculate total cost
        Total_cost_for_charge_in_period += Total_price_in_hour * consumption

        Total_charging_consumption_in_period += consumption
        i += 1
        # print("index i: "+str(i))

    Total_electricity_bill_for_period = Total_cost_for_charge_in_period + Total_cost_regular_consumption_in_period

    avg_cost_kWh = Total_cost_for_charge_in_period / Total_charging_consumption_in_period
    return [avg_cost_kWh, consumers_EVModel, Total_electricity_bill_for_period]

#**Function used two places in callback function to find if any special dates are chosen and if they are first or last**
def specialDateChosen(slider_value):
    overload_day_chosen = False
    overload_day_is_last_in_range = False
    overload_day_only = False
    year_after_chosen = False
    year_after_is_last_in_range = False
    value1 = slider_value[0] - 0.5
    value2 = slider_value[1] - 0.5
    slider_location_year_after = slider_location_of_overload+1
    if  value1 == slider_location_of_overload-0.5 or value2 == slider_location_of_overload-0.5:
        overload_day_chosen = True
        if value1 == slider_location_of_overload-0.5 and value2 == slider_location_of_overload-0.5:
            overload_day_only = True
        elif value1 == slider_location_of_overload - 0.5:
            overload_day_is_last_in_range = False
        else:
            overload_day_is_last_in_range = True
    else:
        overload_day_chosen = False
    if value1 == slider_location_year_after-0.5 or value2 == slider_location_year_after-0.5:
        year_after_chosen = True
        if value1 == slider_location_year_after - 0.5:
            year_after_is_last_in_range = False
        else:
            year_after_is_last_in_range = True
    return overload_day_chosen, overload_day_is_last_in_range, overload_day_only, year_after_chosen, year_after_is_last_in_range


### App layout for dash dashboard using dash_bootstrap for easier layout setup
# Interactive graphing with Dash
# Static figure for DSO income.
fig = px.line(DSOIncome, x="Year", y="DSO Earning [DKK]")

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.LITERA])
server = app.server

# Creating app layout
app.layout = html.Div([
    dcc.Tabs([
        dcc.Tab(label='Overall information', children=[
            dbc.Row(dbc.Col(html.H3("Simulation result dashboard for file: " + str(filename)), width={'size': "auto"}),
                    justify="center", style={'textAlign': 'center'}),
            dbc.Row(dbc.Col(
                dash_table.DataTable(id='parameterInfo', columns=[{"name": i, "id": i} for i in df_parameter.columns],
                                     data=df_parameter.to_dict('records'), ),
                width={'size': "auto"}
                ), justify="center"
                    ),

            # 2 smaller graphs and 1 dropdown
            dbc.Row(
                [
                    dbc.Col(dcc.Graph(id='percentage_EV_types'), width={'size': 4}
                            ),
                    dbc.Col(dcc.Graph(id='percentage_strategies'), width={'size': 4}
                            ),
                ], justify="center"
            ),
        ]),

        dcc.Tab(label='Distribution grid analysis', children=[
            # dbc.Row(dbc.Col(html.H1("Simulation results analysis"), width={'size': 6, 'offset': 3},),),

            dbc.Row(dbc.Col(html.H3("Date and hour for first overload: " + str(first_overload['Timestamp'])),
                            width={'size': "auto"}, ), justify="center", style={'textAlign': 'center'}),
            dbc.Row(dbc.Col(html.H4("Total number of overloads a year after first overload: " + str(numberOfOverloads)), width={'size': "auto"}, ),
                    justify="center", style={'textAlign': 'center'}),

            dbc.Row(dbc.Col(html.Div(children=dcc.Dropdown(id='measurements_category', options=[{'label': x, 'value': x}
                                                                                                for x in
                                                                                                measurements_headers],
                                                           # category is a list
                                                           value='Aggregated base load',  # Initial value
                                                           clearable=False,
                                                           disabled=False
                                                           ),
                                     style={
                                         'height': '50px',
                                         # 'margin-left': '10%',
                                         # 'width':'80%',
                                         # 'float': 'center'
                                     }
                                     ),
                            width={'size': 6, 'offset': 3}
                            )
                    ),

            dbc.Row(dbc.Col(dcc.RangeSlider(
                id='my-range-slider',
                min=min(measurements['Year'].unique()),
                max=max(measurements['Year'].unique()) + 0.01,
                # 0.01 is inserted to make space for showing the last eyar on the slider
                step=None,
                marks={
                    str(x): {'label': str(y)}
                    for x, y in zip(slider_list_values, slider_list_labels)
                },
                value=[2020, 2021])),  # style={'height': '100px', 'width': '100%','display': 'inline-block'} ),
            ),

            dbc.Row(dbc.Col(dcc.Graph(id='line_graph'))),
            # figure={}) #To make a static figure replace empty dict with figure.
            dbc.Row(dbc.Col(html.H4("Datalist of hours with overload"), width={'size': "auto"}), justify="center",
                    style={'textAlign': 'center'}),
            dbc.Row(dbc.Col(
                dash_table.DataTable(
                    id='table',
                    columns=[{"name": i, "id": i} for i in listOfOverloads.columns],
                    data=listOfOverloads.to_dict('records'), ),

                width={'size': "auto"}
            ), justify="center"),
            dbc.Row(dbc.Col(html.H4("Overload details"), width={'size': "auto"}), justify="center",
                    style={'textAlign': 'center'}),
            dbc.Row(dbc.Col(dcc.Dropdown(id='overload_times', options=[{'label': x, 'value': x}
                                                                       for x in listOfOverloads['Timestamp']],
                                         # category is a list
                                         value=listOfOverloads['Timestamp'].iloc[0],  # Initial value
                                         clearable=False,
                                         disabled=False
                                         ),
                            style={
                                'height': '50px',
                                # 'margin-left': '10%',
                                # 'width':'80%',
                                # 'float': 'center'
                            },
                            width={'size': 6, 'offset': 3})
                    ),
            dbc.Row(dbc.Col(dash_table.DataTable(id='table_overloadInfo',
                                                 columns=[{"name": 'Consumer Name', "id": 'Consumer Name'},
                                                          {"name": 'EV model', "id": 'EV model'},
                                                          {"name": 'Charging power [kW]', "id": 'Charging power [kW]'},
                                                          {"name": 'Charging strategy', "id": 'Charging strategy'},
                                                          {"name": 'Driving distance BEFORE charge',
                                                           "id": 'Driving distance BEFORE charge'},
                                                          {"name": 'Driving distance AFTER charge',
                                                           "id": 'Driving distance AFTER charge'}], ),
                            width={'size': "auto"}
                            ), justify="center"),

        ]),

        dcc.Tab(label='EVs and EV owners', children=[
            dbc.Row(dbc.Col(html.H4("EV model information"), width={'size': "auto"}, ), justify="center",
                    style={'textAlign': 'center'}),
            dbc.Row(
                # Bar chart - info of ev model
                dbc.Col(
                    [
                        # EV model dropdown list
                        dbc.Row(dcc.Dropdown(id='EV_info', options=[{'label': x, 'value': x}
                                                                    for x in EVmodels_modelIndex.columns.values],
                                             # category is a list
                                             value='Battery capacity [kWh]',  # Initial value
                                             clearable=False,
                                             style={
                                                 # 'height':'10px',
                                                 'width': '50%',
                                                 'display': 'inline-block',
                                                 # 'float': 'right',
                                                 'margin-left': '10%'
                                             }
                                             ),
                                justify="center"),
                        dbc.Row(dcc.Graph(id='EV_type_info'),
                                style={
                                    # 'height':'100px',
                                    # 'float': 'right',
                                    # 'margin-right': '10px',
                                    'width': '100%',
                                    'display': 'inline-block'
                                }
                                ),
                    ], width={'size': 10}
                ),
                justify="center"),
            dbc.Row(
                dbc.Col(html.H4("Average price per charged kWh in a given time period and the total electricity bill"),
                        width={'size': "auto"}, ), justify="center", style={'textAlign': 'center'}),
            dbc.Row(
                [
                    dbc.Col(
                        dbc.FormGroup([

                            dbc.Label("Start date"),
                            dbc.Input(id='date1_text', value='2020-01-01 00:00', type='text',
                                      style={'width': '100%', 'height': 30}, ),
                        ]), width={'size': 2, 'offset': 1, 'order': 1}
                    ),

                    dbc.Col(
                        dbc.FormGroup([
                            dbc.Label("End date"),
                            dbc.Input(id='date2_text', placeholder='YYYY-MM-DD or YYYY-MM-DD HH:mm', type='text',
                                      style={'width': '100%', 'height': 30}, ),
                        ]),
                        width={'size': 2, 'order': 2}
                    ),

                    dbc.Col(
                        dbc.Button("Apply dates", id="dates_button", color="primary", style={'height': '100%'},
                                   n_clicks=0),
                        width={'size': 1, 'order': 3}
                    ),
                    dbc.Col(
                        html.H4(" ", id="button_push_message"),
                        width={'size': 2, 'order': 4},
                    ),
                ], align="center", no_gutters=False
            ),

            dbc.Row(dbc.Progress(id="progress")),
            # Economic graph
            dbc.Row(
                dbc.Col(dcc.Graph(id='avgPrice'))
            ),
            dbc.Row(
                dbc.Col(dcc.Graph(id='totalBill'))
            ),
            dbc.Row(dbc.Col(html.Div(children=dcc.Dropdown(id='consumerNumber', options=[{'label': x, 'value': x}
                                                                                         for x in consumption_headers],
                                                           # category is a list
                                                           value='domesticConsumers[0]',  # Initial value
                                                           clearable=False,
                                                           disabled=False
                                                           ),
                                     style={
                                         'height': '50px',
                                         # 'margin-left': '10%',
                                         # 'width':'80%',
                                         # 'float': 'center'
                                     }
                                     ),
                            width={'size': 6, 'offset': 3}
                            )
                    ),

            dbc.Row(dbc.Col(dash_table.DataTable(id='table_consumer_info',
                                                 columns=[{"name": 'EV model', "id": 'EV model'},
                                                          {"name": 'Charging strategy', "id": 'Charging strategy'}], ),
                            width={'size': "auto"}
                            ), justify="center"),

            dbc.Row(dbc.Col(dcc.RangeSlider(
                id='individual_consumption_range_slider',
                min=min(measurements['Year'].unique()),
                max=max(measurements['Year'].unique()) + 0.01,
                # 0.01 is inserted to make space for showing the last eyar on the slider
                step=None,
                marks={
                    str(x): {'label': str(y)}
                    for x, y in zip(slider_list_values, slider_list_labels)
                },
                value=[2020, 2021])),  # style={'height': '100px', 'width': '100%','display': 'inline-block'} ),
            ),

            dbc.Row(dbc.Col(dcc.Graph(id='line_graph_individual_consumption'))),
            dbc.Row(dbc.Col(dcc.Graph(id='line_graph_individual_driving_distance'))),
        ]),

        dcc.Tab(label='Electricity market incl. tariffs', children=[
            dbc.Row(dbc.Col(html.Div(children=dcc.Dropdown(id='el_prices', options=[{'label': x, 'value': x}
                                                                                    for x in el_prices_headers],
                                                           # category is a list
                                                           value='Spot Price [DKK/kWh]',  # Initial value
                                                           clearable=False,
                                                           disabled=False
                                                           ),
                                     style={
                                         'height': '50px',
                                         # 'margin-left': '10%',
                                         # 'width':'80%',
                                         # 'float': 'center'
                                     }
                                     ),
                            width={'size': 6, 'offset': 3}
                            )
                    ),

            dbc.Row(dbc.Col(dcc.RangeSlider(
                id='my-range-slider1',
                min=min(measurements['Year'].unique()),
                max=max(measurements['Year'].unique()) + 0.01,
                # 0.01 is inserted to make space for showing the last eyar on the slider
                step=None,
                marks={
                    str(x): {'label': str(y)}
                    for x, y in zip(slider_list_values, slider_list_labels)
                },
                value=[2020, 2021])),  # style={'height': '100px', 'width': '100%','display': 'inline-block'} ),
            ),

            dbc.Row(dbc.Col(dcc.Graph(id='el_price_graph'))),
            # figure={}) #To make a static figure replace empty dict with figure
        ]),
        dcc.Tab(label='DSO income', children=[
            dbc.Row(dbc.Col(dcc.Graph(figure=fig))),
        ]),
    ])
])

### Callback functions for dashboard
# Callback
# callback operator

@app.callback([Output('line_graph_individual_consumption', 'figure'),
               Output('line_graph_individual_driving_distance', 'figure'),
               Output('table_consumer_info', 'data')],
              [Input(component_id='consumerNumber', component_property='value'),
               Input(component_id='individual_consumption_range_slider', component_property='value')
               ]
              )
def overloadInfo_dataTableUpdate(dropdownValue, rangeSlider):
    rangeSlider = [round(x-0.5) for x in rangeSlider]

    year_interval = list(
        range(rangeSlider[0], rangeSlider[1] + 1))  # +1 is because the range does not include the last number

    dff = df_ConsumptionDataFile[df_ConsumptionDataFile['Time'].dt.year.isin(year_interval)]
    dff1 = df_DrivingDistanceDataFile[df_DrivingDistanceDataFile['Time'].dt.year.isin(year_interval)]

    fig = px.line(dff, x="Time", y=str(dropdownValue))
    fig.update_layout(title={'text': 'Individual consumer base load for: '+str(dropdownValue)}, yaxis_title="Base load [kWh]")

    fig1 = px.line(dff1, x="Time", y=str(dropdownValue))
    contain_only_zeros_bool_list = df_DrivingDistanceDataFile[dropdownValue].groupby(
        level=0).all()  # Creating list of booleans, true if not zero value and fals if it's the case
    contain_only_zeros_bool_list1 = contain_only_zeros_bool_list[
        contain_only_zeros_bool_list == True]  # Create list of only true booleans
    if contain_only_zeros_bool_list1.empty:  # Checking if the list is empty
        fig1.update_layout(title={'text': 'Indivual driving distances: This consumer has no EV'},
                           title_font_color='red', yaxis_title="Driving distance [km]")
    else:
        EV_adoption_date = df_DrivingDistanceDataFile['Time'][contain_only_zeros_bool_list1.index[0]]
        fig1.update_layout(
            title={'text': 'Indivual driving distances: '+str(dropdownValue)+' adopts an EV at ' + str(EV_adoption_date)}, yaxis_title="Driving distance [km]")

    dff2 = pd.DataFrame(
        {'EV model': [df_consumerInfo[dropdownValue][0]], 'Charging strategy': [df_consumerInfo[dropdownValue][1]]})
    data_ob = dff2.to_dict('records')

    return fig, fig1, data_ob


@app.callback(Output('table_overloadInfo', 'data'),
              Input(component_id='overload_times', component_property='value')
              )
def overloadInfo_dataTableUpdate(dropdownValue):
    df = overload_info(dropdownValue)
    data_ob = df.to_dict('records')
    return (data_ob)


@app.callback(Output(component_id='el_price_graph', component_property='figure'),
              [Input(component_id='el_prices', component_property='value'),
               Input(component_id='my-range-slider1', component_property='value')
               ])
def Electricity_market(dropdown_value, slider_value):
    # boolean for checking if overload time is chosen
    specialDayBooleans = specialDateChosen(slider_value)
    overload_day_chosen = specialDayBooleans[0]
    overload_day_is_last_in_range = specialDayBooleans[1]
    overload_day_only = specialDayBooleans[2]
    year_after_chosen = specialDayBooleans[3]
    year_after_is_last_in_range = specialDayBooleans[4]

    slider_value = [round(x - 0.5) for x in slider_value]
    year_interval = list(
        range(slider_value[0], slider_value[1] + 1))  # +1 is because the range does not include the last number
    dff = electricityPrices[electricityPrices['Year'].isin(year_interval)]

    if overload_day_chosen and year_after_chosen:
        dff = dff[(first_overload['Timestamp'] <= dff['Timestamp']) & (dff['Timestamp'] <= timeFirstYearAfterOverload)]
    elif overload_day_chosen == True:
        if overload_day_only == True:
            dff = dff[(dff['Timestamp'] >= overload_day['Timestamp'].iloc[0]) & (
                        dff['Timestamp'] <= overload_day['Timestamp'].iloc[len(overload_day['Timestamp']) - 1])]
        elif overload_day_is_last_in_range == True:
            dff = dff[dff['Timestamp'] <= first_overload['Timestamp']]
        else:
            dff = dff[first_overload['Timestamp'] <= dff['Timestamp']]
    elif year_after_chosen == True:
        if year_after_is_last_in_range == True:
            dff = dff[dff['Timestamp'] <= timeFirstYearAfterOverload]
        else:
            dff = dff[timeFirstYearAfterOverload <= dff['Timestamp']]

    if overload_day_only == False:
        fig = px.line(dff, x="Timestamp", y=str(dropdown_value))
        fig.update_layout(title={'text': 'Electricity prices and tariffs'})
    else:
        fig = make_subplots()
        fig.add_trace(go.Scatter(x=dff["Timestamp"], y=dff['Spot Price [DKK/kWh]'], name="Spot price"))
        fig.add_trace(go.Scatter(x=dff["Timestamp"], y=dff['TSO Tariff [DKK/kWh]'], name="TSO Tariff"))
        fig.add_trace(go.Scatter(x=dff["Timestamp"], y=dff['DSO Tariff [DKK/kWh]'], name="DSO Tariff"))
        fig.add_trace(go.Scatter(x=dff["Timestamp"], y=dff['Total price [DKK/kWh]'], name="Total price"))
        fig.update_layout(title={'text': 'Electricity prices and tariffs for first day with overload'})

    return fig


@app.callback(
    [
        Output(component_id='avgPrice', component_property='figure'),
        Output(component_id='totalBill', component_property='figure')
    ],
    [
        Input(component_id='date1_text', component_property='value'),
        Input(component_id='date2_text', component_property='value'),
        Input(component_id='dates_button', component_property='n_clicks')
    ]
)
def EVs_and_EV_owners(date1, date2, n_clicks):
    fig4_price = px.bar(x=['consumer1', 'consumer2'], y=[2, 3], title='Default inital graph layout - choose dates',
                        labels=dict(x="Consumer", y="Avg. charging price per kWh"))
    fig5_totalBill = px.bar(x=['consumer1', 'consumer2'], y=[2, 3], title='Default inital graph layout - choose dates',
                            labels=dict(x="Consumer", y="Total electricity bill for period"))

    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    if 'dates_button' in changed_id:
        print("button push registered")
        pd.to_datetime(date1)
        pd.to_datetime(date2)
        consumerNamesList = householdData.iloc[
            0, index_list_of_households_having_EVs_for_whole_simulation]  # List of consumer names that has an EV.
        i = 0
        total_avg = 0
        avg_per_consumer = []
        consumer = []
        EVModel_list = []
        Total_bill_per_consumer = []
        for name in consumerNamesList:
            avg_value_and_EVModel = avg_chargingCost_per_kWh(date1, date2, name)
            if isinstance(avg_value_and_EVModel, list):
                avg_value = avg_value_and_EVModel[0]
                EVModel = avg_value_and_EVModel[1]
                Total_bill = avg_value_and_EVModel[2]
                consumer.append(name)
                avg_per_consumer.append(avg_value)
                EVModel_list.append(EVModel)
                Total_bill_per_consumer.append(Total_bill)
                total_avg = total_avg + avg_value  # divided by two to find new average
                i += 1
            if int(len(consumerNamesList) / 10) == i:
                print("10%")
            if int(len(consumerNamesList) / 5) == i:
                print("20%")
            if int(len(consumerNamesList) / 3.3333) == i:
                print("30%")
            if int(len(consumerNamesList) / 2.5) == i:
                print("40%")
            if int(len(consumerNamesList) / 2) == i:
                print("50%")
            if int(len(consumerNamesList) / 1.3333) == i:
                print("75%")
            if int(len(consumerNamesList) / 1.11111) == i:
                print("90%")
        total_avg = total_avg / i

        # Creating dataframe
        dataFrame_data = {'consumer': consumer, 'avg_per_consumer': avg_per_consumer, 'EVModel_list': EVModel_list, 'Total_bill_per_consumer': Total_bill_per_consumer}
        d_df = pd.DataFrame(data=dataFrame_data)

        d_df['consumerNumber'] = [int(str(re.split('\[|\]', i)[1])) for i in d_df['consumer']]
        # sort dataframe to make sure colors for EV types are the same for all scenarios making it easier to compare simulations
        d_df.sort_values(['EVModel_list', 'consumerNumber'], ascending=[True, True], inplace=True)

        fig4_price = px.bar(d_df, x="consumer", y="avg_per_consumer", title='Average charging cost', color="EVModel_list",
                            labels={"consumer":"Consumer", "avg_per_consumer":"Avg. charging price per kWh", "EVModel_list":"EV model"})
        fig4_price.add_hline(y=total_avg, line_color="black", line_dash="dot",
                             annotation_text="Avg. of all consumers' avg.: " + str("%.2f" % total_avg),
                             annotation_position="top left", annotation_font_color="black")
        fig4_price.update_yaxes(range=[0, 2])

        fig5_totalBill = px.bar(d_df, x="consumer", y="Total_bill_per_consumer", title='Total electricity bill',
                                color="EVModel_list", labels={"consumer": "Consumer", "Total_bill_per_consumer": "Total electricity bill [DKK]", "EVModel_list":"EV model"})
        fig5_totalBill.update_yaxes(range=[0, 110000])

    return fig4_price, fig5_totalBill


@app.callback(
    [Output(component_id='line_graph', component_property='figure'),
     Output(component_id='percentage_EV_types', component_property='figure'),
     Output(component_id='EV_type_info', component_property='figure'),
     Output(component_id='percentage_strategies', component_property='figure'),
     ],
    [Input(component_id='measurements_category', component_property='value'),
     Input(component_id='my-range-slider', component_property='value'),
     Input(component_id='EV_info', component_property='value')
     ]
)
# callback function
def update_graph(dropdown_value, slider_value, EV_info_value):
    # global overload_day_only
    # boolean for checking if overload time is chosen
    specialDayBooleans = specialDateChosen(slider_value)
    overload_day_chosen = specialDayBooleans[0]
    overload_day_is_last_in_range = specialDayBooleans[1]
    overload_day_only = specialDayBooleans[2]
    year_after_chosen = specialDayBooleans[3]
    year_after_is_last_in_range = specialDayBooleans[4]

    slider_value = [round(x - 0.5) for x in slider_value]

    year_interval = list(
        range(slider_value[0], slider_value[1] + 1))  # +1 is because the range does not include the last number
    dff = measurements[measurements['Year'].isin(year_interval)]

    if overload_day_chosen and year_after_chosen:
        dff = dff[(first_overload['Timestamp'] <= dff['Timestamp']) & (dff['Timestamp'] <= timeFirstYearAfterOverload)]
    elif overload_day_chosen == True:
        if overload_day_only == True:
            dff = dff[(dff['Timestamp'] >= overload_day['Timestamp'].iloc[0]) & (
                        dff['Timestamp'] <= overload_day['Timestamp'].iloc[len(overload_day['Timestamp']) - 1])]
        elif overload_day_is_last_in_range == True:
            dff = dff[dff['Timestamp'] <= first_overload['Timestamp']]
            # print(dff.tail())
        else:
            dff = dff[first_overload['Timestamp'] <= dff['Timestamp']]
    elif year_after_chosen == True:
        if year_after_is_last_in_range == True:
            dff = dff[dff['Timestamp'] <= timeFirstYearAfterOverload]
        else:
            dff = dff[timeFirstYearAfterOverload <= dff['Timestamp']]

    fig1_pie = px.pie(df_EV_types_count, values='Number of models', names='Model', title='EV model distribution to one year after first overload',
                      hover_data=['Battery capacity [kWh]', 'Mileage [kWh/km]',
                                  'Charging power [kW]'])  # labels={'Battery capacity [kWh]':'life expectancy'}
    fig2_bar = px.bar(evModels, x='EV model', y=EV_info_value, title=str(EV_info_value))
    fig3_pie = px.pie(df_strategy_statistics, values='Number of adopted strategies', names='Charging strategy',
                      title='Charging strategy distribution to one year after first overload')

    if overload_day_only == False:
        fig = px.line(dff, x="Timestamp", y=str(dropdown_value))
    else:
        fig = make_subplots(specs=[[{"secondary_y": True}]])

    # fig.update_layout(yaxis={'title':'text'}, title={'text':'some_text', 'font':{'size':28},'x':0.5,'xanchor':'center'})
    if overload_day_only == True:
        # for item in measurements_headers:
        # fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(
            go.Bar(x=dff["Timestamp"], y=dff["Number of charging EVs"], name="Number of charging EVs", opacity=0.8,
                   yaxis="y2"))
        fig.add_trace(go.Scatter(x=dff["Timestamp"], y=dff["Total Maximum grid load"],
                                 name="Total Maximum grid load"))  # secondary_y=False
        fig.add_trace(go.Scatter(x=dff["Timestamp"], y=dff["Aggregated base load"], name="Aggregated base load"))
        fig.add_trace(
            go.Scatter(x=dff["Timestamp"], y=dff["Aggregated charging load"], name="Aggregated charging load"))
        fig.add_trace(
            go.Scatter(x=overload_day_spotPrice["Timestamp"], y=overload_day_spotPrice["Total price [DKK/kWh]"],
                       name="Total electricity price", yaxis="y3"))
        # fig.update_yaxes(range=[0, grid_cap+20], title_text="Load [kW]", secondary_y=False)
        # fig.update_yaxes(range=[0, first_overload["Number of charging EVs"]+10], title_text="Number of charging EVs", secondary_y=True)
        fig.add_hline(y=grid_cap, line_color="red", line_dash="dot",
                      annotation_text="Grid capacity: " + str("%.2f" % grid_cap), annotation_position="top left",
                      annotation_font_color="red")
        fig.update_layout(xaxis={'title': 'Time [Hour]'},
                          yaxis=dict(title="Load [kW]", range=[0, grid_cap + 20]),
                          yaxis2=dict(title="Number of charging EVs",
                                      range=[0, first_overload["Number of charging EVs"] + 10], anchor="free",
                                      overlaying="y", side="right", position=0.99),
                          yaxis3=dict(title="Electricity price [DKK/kWh]", anchor="x", overlaying="y", side="right"),
                          title={'text': 'First day with overload', 'font': {'size': 28}, 'x': 0.5,
                                 'xanchor': 'center'})
    elif dropdown_value == "Total number of EVs" or dropdown_value == "Number of charging EVs" or dropdown_value == "Number of driving EVs":
        fig.update_layout(xaxis={'title': 'Time'},
                          title={'text': str(dropdown_value), 'font': {'size': 28}, 'x': 0.5, 'xanchor': 'center'})
    else:
        # fig.update_layout(xaxis={'title':'Time'}, yaxis={'title':str(dropdown_value)+" [kW]"}, title={'text':str(dropdown_value), 'font':{'size':28},'x':0.5,'xanchor':'center'})
        fig.add_hline(y=grid_cap, line_color="red", line_dash="dot",
                      annotation_text="Grid capacity: " + str("%.2f" % grid_cap), annotation_position="top left",
                      annotation_font_color="red")
        fig.update_yaxes(range=[0, grid_cap + 20])
        fig.update_layout(xaxis={'title': 'Time'}, yaxis={'title': str(dropdown_value) + " [kW]"},
                          title={'text': str(dropdown_value), 'font': {'size': 28}, 'x': 0.5, 'xanchor': 'center'})
    return fig, fig1_pie, fig2_bar, fig3_pie

    # print(value_choice)
    # dff = measurements[measurements_headers==value_choice] #always create a copy of the dataframe

#### Dashboard run on local webbrowser window
#Run app
import webbrowser
from threading import Timer
def open_browser():
    webbrowser.open_new("http://localhost:{}".format(port))
if __name__=='__main__':
    Timer(1, open_browser).start();
    app.run_server(debug=True, port=port, use_reloader=False)