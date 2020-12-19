import pandas, numpy, folium
import additional_func as af
from folium.plugins import HeatMap
import seaborn as sns
import matplotlib.pyplot as plt
pandas.set_option('display.max_columns', None)

data_frame_circuits = pandas.read_csv('F1_data/circuits.csv')
data_frame_constructor_results = pandas.read_csv('F1_data/constructor_results.csv')
data_frame_constructor_standings = pandas.read_csv('F1_data/constructor_standings.csv')
data_frame_constructors = pandas.read_csv('F1_data/constructors.csv')
data_frame_driver_standings = pandas.read_csv('F1_data/driver_standings.csv')
data_frame_drivers = pandas.read_csv('F1_data/drivers.csv')
data_frame_lap_times = pandas.read_csv('F1_data/lap_times.csv')
data_frame_pit_stops = pandas.read_csv('F1_data/pit_stops.csv')
data_frame_qualifying = pandas.read_csv('F1_data/qualifying.csv')
data_frame_races = pandas.read_csv('F1_data/races.csv')
data_frame_results = pandas.read_csv('F1_data/results.csv')
data_frame_seasons = pandas.read_csv('F1_data/seasons.csv')
data_frame_status = pandas.read_csv('F1_data/status.csv')

print('circuits', data_frame_circuits.shape, '  ', data_frame_circuits.columns)
print('constructor_results', data_frame_constructor_results.shape, '  ', data_frame_constructor_results.columns)
print('constructor_standings', data_frame_constructor_standings.shape, '  ', data_frame_constructor_standings.columns)
print('constructors', data_frame_constructors.shape, '  ', data_frame_constructors.columns)
print('driver_standings', data_frame_driver_standings.shape, '  ', data_frame_driver_standings.columns)
print('drivers', data_frame_drivers.shape, '  ', data_frame_drivers.columns)
print('lap_times', data_frame_lap_times.shape, '  ', data_frame_lap_times.columns)
print('pit_stops', data_frame_pit_stops.shape, '  ', data_frame_pit_stops.columns)
print('qualifying', data_frame_qualifying.shape, '  ', data_frame_qualifying.columns)
print('races', data_frame_races.shape, '  ', data_frame_races.columns)
print('results', data_frame_results.shape, '  ', data_frame_results.columns)
print('seasons', data_frame_seasons.shape, '  ', data_frame_seasons.columns)
print('status', data_frame_status.shape, '  ', data_frame_status.columns)

data_frame_circuits.dropna()
print('DATAFRAME BEFORE MERGE')
print(data_frame_results.columns)
data_frame_results = data_frame_results.merge(data_frame_constructors, left_on='constructorId', right_on='constructorId', how='right')
# data_frame_results.drop(['nationality'])
data_frame_results = data_frame_results.merge(data_frame_races, left_on='raceId', right_on='raceId', how='right')
data_frame_results = data_frame_results.merge(data_frame_status, left_on='statusId', right_on='statusId', how='right')
data_frame_results = data_frame_results.merge(data_frame_drivers, left_on='driverId', right_on='driverId', how='right')
print('DATAFRAME AFTER MERGE')
print(data_frame_results.columns)

for feature in data_frame_results.columns:
    if data_frame_results[feature].dtype == object:
        data_frame_results[feature] = pandas.Categorical(data_frame_results[feature])
        sex_map_train = dict(zip(data_frame_results[feature].cat.codes, data_frame_results[feature]))
        data_frame_results[feature] = data_frame_results[feature].cat.codes

attribute = ['resultId', 'raceId', 'driverId', 'constructorId', 'number_x', 'grid',
       'position', 'positionText', 'positionOrder', 'points', 'laps', 'time_x',
       'milliseconds', 'fastestLap', 'rank', 'fastestLapTime',
       'fastestLapSpeed', 'statusId', 'constructorRef', 'name_x',
       'nationality_x', 'url_x', 'year', 'round', 'circuitId', 'name_y',
       'date', 'time_y', 'url_y', 'status', 'driverRef', 'number_y', 'code',
       'forename', 'surname', 'dob', 'nationality_y', 'url']
plt.figure()
corr = data_frame_results[['nationality_y'] + attribute[0:]].corr()
sns_hmap = sns.heatmap(abs(corr))
sns_hmap.set_title("correlation PANDAS + SEABORN")
plt.figure()
corr = data_frame_results[['nationality_y'] + attribute[14:]].corr()
sns_hmap = sns.heatmap(abs(corr))
sns_hmap.set_title("correlation PANDAS + SEABORN")


map = folium.Map([0, 0], zoom_start=2)
for coodr in data_frame_circuits[['lat', 'lng']].to_numpy():
    folium.Marker(location=coodr, popup="",
                  icon=folium.Icon(color='red', icon='info-sign')).add_to(map)
map.save('map.html')


plt.show()
pandas.reset_option('max_columns')



