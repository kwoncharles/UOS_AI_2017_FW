#_*_ coding:utf-8 _*_

fp = open("input_data.txt","r")
lines = fp.readlines()

fp.close()


fish = list()


for line in lines:
    print line
    fish.append(line.split())

salmon = list()
seabass = list()

# The ratio of (length : tail length) is lower than 8 is salmon, else seabass
for i in range(len(fish)):
    
    if (int(fish[i][0])/int(fish[i][1]) < 8):
        salmon.append(fish[i])
    else:
        seabass.append(fish[i])


lines = list()

# print with string type and save

print "salmon : "
for i in range(len(salmon)):
    line = "Body :"+salmon[i][0]+" Tail :"+salmon[i][1]+" ===> salmon\n"
    lines.append(line)
    print line

print "seabass :"
for i in range(len(seabass)):
    line = "Body :"+seabass[i][0]+" Tail :"+seabass[i][1]+" ===> seabass\n"
    lines.append(line)
    print line

fp = open("output_result.txt","w")

for line in lines:
    fp.write(line)

fp.close()
