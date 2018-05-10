#_*_ coding: utf-8 _*_

import Tkinter as tk
import sys

map = []
map.append(range(1, 6))
map.append(range(6, 11))
map.append(range(11, 16))
map.append(range(16, 21))
map.append(range(21, 26))

# 막힌 곳은 0으로 표시
map[0][3] = 0
map[1][1] = 0
map[2][1] = 0
map[1][3] = 0
map[2][3] = 0
map[4][3] = 0

# 노드의 좌표로부터 이름 반환
def getNodeName(location):
    return map[location[0]][location[1]]


# 해당 노드로 이동 가능한지 확인
def isExist(location, toVisit, alreadyVisited):
    if location[0] < 0:
        return False
    
    if location[1] < 0:
        return False
    
    if location[0] > 4:
        return False
    
    if location[1] > 4:
        return False
    
    # 막힌 곳 판정
    if getNodeName(location) == 0:
        return False
    
    # 이미 방문해야 할 목록에 들어있는지 판정
    if location in toVisit:
        return False
    
    # 이미 방문했던 곳 판정
    if location in alreadyVisited:
        return False
    
    return True


# 윈도우 콜백 클래스
class App:
    def __init__(self, master):
        # 맵을 그릴 캔버스 생성
        self.canvas = tk.Canvas(master, width = 800, height = 600)
        self.canvas.pack()
        
        # 버튼 생성
        self.button = tk.Button(master, text = 'run', command = self.run)
        self.button.pack(side=tk.BOTTOM)
        
        # 맵 그리기
        for row in range(len(map)):
            for col in range(len(map[0])):
                if map[row][col] == 0:
                    fillColor = 'black'
                else:
                    fillColor = 'white'
                
                self.canvas.create_rectangle(col * 100, row * 100, col * 100 + 100, row * 100 + 100, fill = fillColor, outline = 'blue')
                self.canvas.create_text(col * 100 + 50, row * 100 + 50, text = map[row][col])
        
        # A* 초기화
        self.start = [2, 0, 0] # 행 , 열  , g(n)(계산용))
        
        # 이동한 횟수를 저장할 변수 count 선언
        self.count = 0
        
        node = getNodeName(self.start)
        
        self.alreadyVisited = []
        self.toVisit = []
        self.toVisit.append(self.start)
        self.visited= []
        self.astar = [] # astar 리스트 선언 및 초기화
        
        # class의 다른 method에서도 root를 사용할 수 있도록 저장
        self.root = master
    
    def run(self):
        if len(self.toVisit) != 0:
            print "step {}\n\n".format(self.count+1)
            
            current = self.toVisit.pop(0)
            #temp = self.astar.pop(0) # astar 값도 삭제
            
            # 현재 노드 칠하기
            row = current[0]
            col = current[1]
            self.canvas.create_rectangle(col * 100, row * 100, col * 100 + 100, row * 100 + 100, fill = 'red', outline = 'blue')
            
            # 색칠된 current 노드를 alreadyVisit 리스트에 추가
            self.alreadyVisited.append([current[0],current[1]])
            self.visited.append([])
            nodeName = getNodeName(current)
            
            # 목표지점에 도달했다면 현재까지의 경로와 거쳐 온 노드 수 출력
            if(nodeName == 15):
                self.button.pack_forget() # 버튼 삭제
                fp = open("output.txt", "w")
                fp.write("Astar Shortest Path\n:")
                for i in range(len(self.alreadyVisited)):
                    
                    if i == len(self.alreadyVisited)-1:
                        #label = tk.Label(self.root, text= "{}".format(self.alreadyVisited[i]))
                        label = tk.Label(self.root, text= "[{}, {}]".format(self.alreadyVisited[i][0],
                                                                            self.alreadyVisited[i][1]))
                        label.pack()
                            
                        row = self.alreadyVisited[i][0]
                        col = self.alreadyVisited[i][1]
                        self.canvas.create_rectangle(col * 100, row * 100, col * 100 + 100, row * 100 + 100, fill = 'gold', outline = 'blue')
                                                                            
                        fp.write(str(self.alreadyVisited[i]))
                        break
                    if abs((self.alreadyVisited[i][0]-self.alreadyVisited[i+1][0])) + \
                        abs((self.alreadyVisited[i][1]-self.alreadyVisited[i+1][1])) > 1:
                        continue
                    label = tk.Label(self.root, text= "{} -> ".format(self.alreadyVisited[i]))
                    label.pack()
                    
                    row = self.alreadyVisited[i][0]
                    col = self.alreadyVisited[i][1]
                    self.canvas.create_rectangle(col * 100, row * 100, col * 100 + 100, row * 100 + 100, fill = 'gold', outline = 'blue')
            
            
                    fp.write("{} -> ".format(str(self.alreadyVisited[i])))
                
                fp.write("\n\nFinished !\nSteps to find Shortest Path : {}".format(self.count))
                label = tk.Label(self.root, text= "Finished !\nSteps to find Shortest Path : {}".format(self.count))
                label.pack()
            
            # 현재 노드의 자식 노드(인접 노드)를 방문해야 할 리스트에 추가
            childList = []
            childList.append([current[0], current[1] + 1] )
            childList.append([current[0] - 1, current[1]])
            childList.append([current[0], current[1] - 1])
            childList.append([current[0] + 1, current[1]])
            
            
            for child in childList:
                # 갈 수 있는 노드인 경우에만 추가
                if isExist(child, self.toVisit, self.alreadyVisited) == True:
                    
                    child = [child[0],child[1],current[2]+1,[current[0],current[1]]] # g(n), 이전 방문 노드의 값을 노드에 추가
                    self.toVisit.append(child)
                    num1 = abs(child[0] - 2) + abs(child[1] - 4) # h(n)
                    num2 = child[2] # g(n)
                    num3 = num1 + num2 # f(n)
                    
                    self.astar.append([num3,num1,num2,child]) # [ f(n), h(n), g(n), 노드 ] 순서로 astar리스트에 덧붙인다
    
    
            print "current toVisit value :\n",self.toVisit
            
            # astar 리스트를 f(n)값 기준으로 오름차순 정렬
            self.astar.sort()
            
            print "\ncurrent Astar value : \n /// [ f(n) , h(n) , g(n) , [row, column, g(n)] ] ///\n"
            for i in range(len(self.astar)):
                print "[{},{}] : {}\n".format(self.astar[i][3][0],self.astar[i][3][1], self.astar[i])
            # astar에서 f(n)이 가장 작은 값을 가졌던 노드를 toVisit의 맨 앞으로 삽입 (pop했을 때 나올 수 있도록)
            
            self.toVisit.remove(self.astar[0][3]) # astar는 리스트의 첫번째 값을 기준으로
            # 정렬되어있기 때문에 astar[0]은 f(n)이 가장 작은 노드이다
            self.toVisit.insert(0,self.astar[0][3])
            
            self.astar.pop(0) # toVisit의 맨 앞으로 넘겨준 astar 값 삭제
            
            print "\nvisited Node :", self.alreadyVisited,"\n\n"
            
            self.count += 1




root = tk.Tk()
app = App(root)
root.mainloop()
