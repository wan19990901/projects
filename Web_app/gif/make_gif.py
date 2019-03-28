#!/usr/bin/env python3

import pandas as pd
import matplotlib
matplotlib.use('Agg') # used for making plots under linux command line
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import animation,rc
from mpl_toolkits.basemap import Basemap
import numpy as np
import sys

#print(sys.argv)
## analyze parameters
var_reg = sys.argv[1];
var_reg = var_reg.replace('_', ' ')
var_starttime = sys.argv[2];
var_endtime = sys.argv[3];
str_starttime = var_starttime.split('-')
str_endtime = var_endtime.split('-')
start_year = int(str_starttime[0])
start_month = int(str_starttime[1])
start_day = int(str_starttime[2])
end_year = int(str_endtime[0])
end_month = int(str_endtime[1])
end_day = int(str_endtime[2])
#print(var_reg, start_year, start_month, start_day, end_year, end_month, end_day)

## load data
terror=pd.read_csv('/var/www/html/gif/data/globalterrorismdb_0617dist.csv',encoding='ISO-8859-1')
terror.rename(columns={'iyear':'Year','imonth':'Month','iday':'Day','country_txt':'Country','region_txt':'Region','attacktype1_txt':'AttackType','target1':'Target','nkill':'Killed','nwound':'Wounded','summary':'Summary','gname':'Group','targtype1_txt':'Target_type','weaptype1_txt':'Weapon_type','motive':'Motive'},inplace=True)
terror=terror[['Year','Month','Day','Country','Region','city','latitude','longitude','AttackType','Killed','Wounded','Target','Summary','Group','Target_type','Weapon_type','Motive']]
terror.set_index(['Year', 'Country'])
terror['casualities']=terror['Killed']+terror['Wounded']
#terror_usa=terror[terror['Country']=='United States']
#print(terror.Country.unique())
#print("load data success!")
## handle parameters
if var_reg == 'World':
    pass
else:
    terror=terror[terror['Country']==var_reg]
#print(var_reg)
terror = terror[((terror['Year']>start_year) | \
                ((terror['Year']==start_year)& ((terror['Month']>start_month)|((terror['Month']==start_month)&(terror['Day']>=start_day))))) \
                & \
                ((terror['Year']<end_year) | \
                ((terror['Year']==end_year)& ((terror['Month']<end_month)|((terror['Month']==end_month)&(terror['Day']<=end_day)))))]
#print("transform data success!")

## make plots
# global terrorist attacks
m3 = Basemap(projection='mill',llcrnrlat=-80,urcrnrlat=80, llcrnrlon=-180,urcrnrlon=180,lat_ts=20,resolution='c',lat_0=True,lat_1=True)
lat_100=list(terror[terror['casualities']>=75].latitude)
long_100=list(terror[terror['casualities']>=75].longitude)
x_100,y_100=m3(long_100,lat_100)
m3.plot(x_100, y_100,'go',markersize=5,color = 'r')
lat_=list(terror[terror['casualities']<75].latitude)
long_=list(terror[terror['casualities']<75].longitude)
x_,y_=m3(long_,lat_)
m3.plot(x_, y_,'go',markersize=2,color = 'b',alpha=0.4)
m3.drawcoastlines()
m3.drawcountries()
m3.fillcontinents(lake_color='aqua')
m3.drawmapboundary(fill_color='aqua')
fig=plt.gcf()
fig.set_size_inches(10,6)
plt.title('Total Terrorist Attacks')
plt.legend(loc='lower left',handles=[mpatches.Patch(color='b', label = "< 75 casualities"),
                    mpatches.Patch(color='red',label='> 75 casualities')])
plt.savefig('/var/www/html/gif/data/staticfig.png', dpi=300)

# gif animation
fig = plt.figure(figsize = (10,8))
def animate(Year):
    ax = plt.axes()
    ax.clear()
    ax.set_title('Animation Of Terrorist Activities'+'\n'+'Year:' +str(Year))
    m6 = Basemap(projection='mill',llcrnrlat=-80,urcrnrlat=80, llcrnrlon=-180,urcrnrlon=180,lat_ts=20,resolution='c')
    lat6=list(terror[terror['Year']==Year].latitude)
    long6=list(terror[terror['Year']==Year].longitude)
    x6,y6=m6(long6,lat6)
    m6.scatter(x6, y6,s=[(kill+wound)*0.1 for kill,wound in zip(terror[terror['Year']==Year].Killed,terror[terror['Year']==Year].Wounded)],color = 'r')
    m6.drawcoastlines()
    m6.drawcountries()
    m6.fillcontinents(zorder = 1,alpha=0.4)
    m6.drawmapboundary()
ani = animation.FuncAnimation(fig,animate,list(terror.Year.unique()), interval = 1500)    
ani.save('/var/www/html/gif/data/animation.gif', writer='imagemagick', fps=1)
plt.close(1)
