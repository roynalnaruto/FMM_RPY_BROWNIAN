#from subprocess import call
#import os


#for x in range(2, 2):
#	temp = open("temp", "w")
#	y = x*100
#	temp.write(str(y))
#	os.system("../objs/rpy < temp")
#	temp.close()
#	os.system("rm temp");




import csv
csvFile = open("../timings/npos_time", "r")
npos=[]
timeTaken=[]
count=0

for line  in csvFile:
	line = line.rstrip()
	word=line.split(',')	
	print(word[0])
	print(word[1])
	npos.append(word[0])
	num = int(word[1])
	timeTaken.append(num/1000)
	count=count+1

import matplotlib.pyplot as plt
plt.plot(npos, timeTaken, 'bo--', label="Number of Particles vs Total Time taken")
plt.axis([0,60000,0,50000])
plt.xlabel('No. of Particles')
plt.ylabel('Total Time Taken')
plt.title('Number of particles vs Total Time Taken')
plt.legend()
plt.savefig("./plot01.png")
plt.clf()
csvFile.close()
