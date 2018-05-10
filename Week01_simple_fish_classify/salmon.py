#_*_ coding:utf-8 _*_

# input data 파일 오픈 및 파일 내용 저장

fp = open("input_data.txt","r")
lines = fp.readlines()

fp.close()


fish = list() # 고기들의 정보를 저장할 list 선언


for line in lines:
    print line # 읽은 파일의 내용 출력
    fish.append(line.split()) # split을 통해 문자열 내의 공백을 제외한 값을 list에 저장

# 어종을 나누어 저장할 list 선언

salmon = list()
seabass = list()

# (총길이 : 꼬리길이) 의 비율이 8보다 작은 물고기들은 연어, 아닌 고기들은 농어로 분류

for i in range(len(fish)):
    
    # 숫자 연산을 위해 int형으로 형변환 후 계산
    
    if (int(fish[i][0])/int(fish[i][1]) < 8):
        salmon.append(fish[i])
    else:
        seabass.append(fish[i])


lines = list() # 파일출력을 위해 문자열을 저장할 list선언


# 연어,농어들의 수만큼 반복문을 돌며 문자열형태로 출력 및 저장

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

# 입력모드로 파일 오픈 후 파일로 문자열을 출력 !

fp = open("output_result.txt","w")

for line in lines:
    fp.write(line)

fp.close()
