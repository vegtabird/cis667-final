from ai import AIcontroller
from ai import checkWiner
import tkinter
from tkinter import *
from cnn import CNN
import random
board_size = 7
model_path = ".\model\m1_7.ckpt"
class Gamer():
    
    def __init__(self,size,AiMode,cnn):
        self.size = size          
        self.BLACK = 1.0
        self.WHITE = 2.0
        self.player = self.BLACK
        self.prob = 0.2
        self.node=0
        self.netAi = AIcontroller(self.BLACK,1,cnn)
        self.ui_winner = 0
    def transferPosToAxis(self,position):
        col = position % self.size
        if position < self.size:
            row = 0
        else:
            row = (position-col)/self.size
            row = int(row)
        return [row,col]
    def transferAxisToPos(self,point):
        return point[0] * self.size + point[1]

    def initialize(self,size,mode,step):
        self.step = step
        self.node=0
        self.netAi.initilaize(self.BLACK,mode)
        self.size = size
        self.player = self.WHITE
        self.board =  [[0.0 for i in range(self.size)] for i in range(self.size)]
        center = size/2
        center = int(center)
        self.board[center][center] =self.BLACK
        self.remaining = [i for i in range(self.size*self.size)]
        self.remaining.remove(center*self.size+center)
    def playRandom(self,color):
        total = len(self.remaining)
        r = random.randint(0,total-1)
        positoin = self.remaining[r]
        point = self.transferPosToAxis(positoin)
        self.board[point[0]][point[1]] = color
        self.remaining.remove(positoin)
        return point
    def playAi(self,color):
        point,node = self.netAi.playNextStep(self.board,self.remaining)
        self.board[point[0]][point[1]] = color
        self.node+=node
        position = self.transferAxisToPos(point)
        self.remaining.remove(position)
        return point

    def chess_loop(self):
        winner = 0
        while(1):
            if(len(self.remaining) == 0):
                break
            next_position = [0,0]
            if self.player == self.BLACK:
                print("ai player (%d)-(%d)" %(self.size,self.step))
                next_position = self.playAi(self.player)
            else:
                print("baseline player (%d)-(%d)" %(self.size,self.step))
                next_position = self.playRandom(self.player)
            over = checkWiner(self.board,next_position)
            if over:
                winner = self.player
                break
            else:
                p = random.random()
                if p > self.prob:
                    self.player = self.BLACK if self.player == self.WHITE else self.WHITE
        return winner,self.node


    def draw_board(self):
        for i in range(self.size):
            self.can.create_line((20,20+i*30),(self.width-20,20+i*30),width=1)
            self.can.create_line((20+i*30,20),(20+i*30,self.width-20),width=1)

    def game_play(self):
        self.board = [[0.0] for i in range(self.size*self.size)]
        self.remaining = [i for i in range(self.size*self.size)]           
        self.player = self.BLACK
        winner = self.chess_loop()
        return winner

    def playAiWithUi(self,color):
        point = self.playAi(color)
        self.draw_chess(point)
        over = checkWiner(self.board,point)
        if over:
            print("you are lose")
            self.ui_winner = self.BLACK
            return
        p = random.random()
        if p > self.prob:
            print("you turn")
            self.player = self.WHITE
        else:
            self.player = self.BLACK
            self.playAiWithUi(self.BLACK)
    def draw_chess(self,point):
        x =20+(point[1]*30)
        y = 20+point[0]*30
        color = "black"
        if self.board[point[0]][point[1]] == self.BLACK:
            color = "black"
        elif self.board[point[0]][point[1]] == self.WHITE:
            color = "white"
        self.can.create_oval(x-11,y-11,x+11,y+11,fill=color)
    def test_ai(self,event):
        while(1):
            if(len(self.remaining) == 0):
                break
            next_position = [0,0]
            if self.player == self.BLACK:
                print("ai play")
                next_position = self.playAi(self.player)
            else:
                print("basline line play")
                next_position = self.playRandom(self.player)
            over = checkWiner(self.board,next_position)
            self.draw_chess(next_position)
            if over:
                if self.player == self.BLACK:
                    print("winner is ai")
                else:
                    print("winner is baseline")
                print("total node is ")
                print(self.node)
                break
            else:
                p = random.random()
                if p > self.prob:
                    self.player = self.BLACK if self.player == self.WHITE else self.WHITE
    
    def human_play(self,event):
        if self.player == self.BLACK or self.ui_winner !=0.0:
            return
        x = event.x
        y=event.y
        if x >=20 and x <=self.width-20 and y >= 20 and y <=self.width:

            row = (y-20)/30
            row = round(row)
            col =(x-20)/30
            col=round(col)
            if self.board[row][col] == 0:
                self.board[row][col] = self.WHITE
                self.draw_chess([row,col])
                over = checkWiner(self.board,[row,col])
                if over:
                    print("you are win")
                    self.ui_winner = self.BLACK
                    return
                p = random.random()
                if p > self.prob:
                    print("ai turn")
                    self.player = self.BLACK
                    self.playAiWithUi(self.player)

    def game_play_withUi(self):
        self.board =  [[0.0 for i in range(self.size)] for i in range(self.size)]
        self.remaining = [i for i in range(self.size*self.size)]
        self.root = Tk()
        self.root.title("game")
        self.width = (self.size-1)*30+40
        self.can = Canvas(self.root,bg="yellow",width=self.width,height=self.width)
        self.can.grid(row=0, column=0) 
        self.draw_board()
        center = self.size/2
        center = int(center)
        self.board[center][center] = self.BLACK
        self.player = self.WHITE
        pos = self.transferAxisToPos([center,center])
        self.remaining.remove(pos)
        self.draw_chess([center,center])
        self.can.bind("<Button-1>",self.human_play)
        self.root.mainloop()
if __name__ == "__main__":
    cnn = CNN(board_size)
    cnn.restore(model_path)
    game = Gamer(board_size,1,cnn)
    print("game start")
    print("each turn will have 0.2 possible to continue play")
    game.game_play_withUi()
        
            
    