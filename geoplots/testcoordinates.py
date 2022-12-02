import pandas as pd 
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt

#import city data
NO_cities = pd.read_csv("Data/maps/NO_cities_coordinates.csv")
#selection of cities
cities = ["Oslo", "Bergen", "Trondheim", "Stavanger", "Bodø", "Tromsø", "Hamar", "Narvik"]
#extract latitudes and longitudes
lats = []
lons = []
for i in range(len(NO_cities)):
    if NO_cities.city[i] in cities:
        lats.append(NO_cities.lat[i])
        lons.append(NO_cities.lng[i])

#make map
map = Basemap(llcrnrlon=1, urcrnrlon=25, llcrnrlat=55, urcrnrlat=70, resolution='i', projection='tmerc', lat_0=0, lon_0=0)
map.drawmapboundary(fill_color='aqua')
map.fillcontinents(color='lightgrey', lake_color='aqua')
map.drawcoastlines(linewidth=0.3)
map.drawcountries(linewidth=0.3)

#fylkegrense
#map.readshapefile("Data/maps/basisdata_norge/administrative_enheter_fylke", 'states', drawbounds = True) #NOT WORKING YET


#map.drawstates()
#map.drawcounties() 
#draw cities
x,y = map(lons, lats)
map.scatter(x, y, marker='D', color='red', zorder=100)

plt.show()


#TODO: ADD FOREIGN CITIES