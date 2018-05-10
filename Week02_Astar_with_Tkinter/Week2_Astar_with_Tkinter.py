#_*_ coding: utf-8 _*_

import Tkinter as tk
import sys

map = []
map.append(range(1, 6))
map.append(range(6, 11))
map.append(range(11, 16))
map.append(range(16, 21))
map.append(range(21, 26))

# Closed parts are marked as 0
map[0][3] = 0
map[1][1] = 0
map[2][1] = 0
map[1][3] = 0
map[2][3] = 0
map[4][3] = 0


def getNodeName(location):
    return map[location[0]][location[1]]


# Figure whether can visit the node out
def isExist(location, toVisit, alreadyVisited):
    if location[0] < 0:
        return False
    
    if location[1] < 0:
        return False
    
    if location[0] > 4:
        return False
    
    if location[1] > 4:
        return False
    
    # If it's closed
    if getNodeName(location) == 0:
        return False
    
    # If it's already in toVisit list
    if location in toVisit:
        return False
    
    # If it's already visited
    if location in alreadyVisited:
        return False
    
    return True


# Window callback
class App:
    def __init__(self, master):
        # Create canvas
        self.canvas = tk.Canvas(master, width = 800, height = 600)
        self.canvas.pack()
        
        # Create button
        self.button = tk.Button(master, text = 'run', command = self.run)
        self.button.pack(side=tk.BOTTOM)
        
        # Plot the map
        for row in range(len(map)):
            for col in range(len(map[0])):
                if map[row][col] == 0:
                    fillColor = 'black'
                else:
                    fillColor = 'white'
                
                self.canvas.create_rectangle(col * 100, row * 100, col * 100 + 100, row * 100 + 100, fill = fillColor, outline = 'blue')
                self.canvas.create_text(col * 100 + 50, row * 100 + 50, text = map[row][col])
        
        # Initialize A*
        self.start = [2, 0, 0] # row, col, g(n)(for computing))
        
        # save moving counts
        self.count = 0
        
        node = getNodeName(self.start)
        
        self.alreadyVisited = []
        self.toVisit = []
        self.toVisit.append(self.start)
        self.visited= []
        self.astar = [] #
        
        self.root = master
    
    def run(self):
        if len(self.toVisit) != 0:
            print "step {}\n\n".format(self.count+1)
            
            current = self.toVisit.pop(0)
            
            # paint current node
            row = current[0]
            col = current[1]
            self.canvas.create_rectangle(col * 100, row * 100, col * 100 + 100, row * 100 + 100, fill = 'red', outline = 'blue')
            
            # Add painted node to alreadyVisit list
            self.alreadyVisited.append([current[0],current[1]])
            self.visited.append([])
            nodeName = getNodeName(current)
            
            # If it got at goal node,then print the route and passed nodes count
            if(nodeName == 15):
                self.button.pack_forget()
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
            
            # Add child of current node to list to visit.
            childList = []
            childList.append([current[0], current[1] + 1] )
            childList.append([current[0] - 1, current[1]])
            childList.append([current[0], current[1] - 1])
            childList.append([current[0] + 1, current[1]])
            
            
            for child in childList:
                # Add only if it is can visit
                if isExist(child, self.toVisit, self.alreadyVisited) == True:
                    
                    # g(n), Add the value of previous visited node
                    child = [child[0],child[1],current[2]+1,[current[0],current[1]]] # g(n), 이전 방문 노드의 값을 노드에 추가
                    self.toVisit.append(child)
                    num1 = abs(child[0] - 2) + abs(child[1] - 4) # h(n)
                    num2 = child[2] # g(n)
                    num3 = num1 + num2 # f(n)
                    
                    self.astar.append([num3,num1,num2,child]) # [ f(n), h(n), g(n), node ]
    
    
            print "current toVisit value :\n",self.toVisit
            
            # Sort ascending order
            self.astar.sort()
            
            print "\ncurrent Astar value : \n /// [ f(n) , h(n) , g(n) , [row, column, g(n)] ] ///\n"
            for i in range(len(self.astar)):
                print "[{},{}] : {}\n".format(self.astar[i][3][0],self.astar[i][3][1], self.astar[i])
            # Insert the node that has a minimum value in the head of toVisit list
            # So when we call the pop function, we can get the minimum node
            
            self.toVisit.remove(self.astar[0][3])
            
            # Astar list is sorted based on first index's value(f(n)
            # So, astar[0] is the node that has the smallest f(n) value
            self.toVisit.insert(0,self.astar[0][3])
            
            # Delete the used value
            
            print "\nvisited Node :", self.alreadyVisited,"\n\n"
            
            self.count += 1




root = tk.Tk()
app = App(root)
root.mainloop()
