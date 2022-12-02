from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt

#choose map/projection
#map = Basemap(projection='cyl', lat_0=0, lon_0=0) #enire earth
map = Basemap(llcrnrlon=2, urcrnrlon=25, llcrnrlat=57, urcrnrlat=70, resolution='i', projection='tmerc', lat_0=0, lon_0=0)


#fill with color
map.drawmapboundary(fill_color='aqua')
map.fillcontinents(color='coral', lake_color='aqua')
map.drawcoastlines()
map.drawcountries(linewidth=2, color='k')
map.drawstates()
map.drawcounties()




#Plot Oslo:
#x,y = map(10.7, 59.9) #Oslo (first latitude, then longitude)
#map.plot(x,y, marker='D', color='m')

#Plot Oslo, Bergen, Trondheim
lats = [59.9, 60.4, 63.44] #latitudes 
lons = [10.7, 5.32, 10.42] #longitudes
x,y = map(lons, lats)
map.scatter(x, y, marker='D', color='red', zorder=100)

#route from Oslo to Bergen, to Trondheim
lats = [59.9, 60.4, 63.44] #latitudes 
lons = [10.7, 5.32, 10.42] #longitudes
x,y = map(lons, lats)
map.plot(x, y, marker=None, color='m', zorder = 99)

#another route from Oslo to Trondheim
lats = [59.9, 63.44] #latitudes 
lons = [10.7, 10.42] #longitudes
x,y = map(lons, lats)
map.plot(x, y, marker=None, color='g', linewidth=2, zorder = 99)


plt.show()
#plt.savefig('test.png')

