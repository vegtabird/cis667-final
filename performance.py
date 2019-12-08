from game import Gamer
from cnn_2 import CNN
import matplotlib.pyplot as plt
#mode=1 using cnn 
#mode=2 without cnn
size = [5,6,7,8,9]
per_game_len = 100
modelPath = ".\model\model3.ckpt"
mode = 1
if __name__ == '__main__':

    size_list = [str(i)+"*"+str(i)for i in size]
    win_list = []
    node_list = []
    win_without_list = []
    node_without_list=[]

    my_cnn = CNN()
    my_cnn.restore(modelPath)
    game = Gamer(size[0],mode,my_cnn)
    baseline_win = 0
    ai_win = 0
    node = 0
    for i in range(len(size)):
        f = open("./result_with.txt","a")
        f.write("(%d) size\n"%(size[i]))
        node = 0
        ai_win = 0
        baseline_win=0
        for j in range(per_game_len):
            game.initialize(size[i],mode,j)
            winner,t_node = game.chess_loop()
            node+=t_node
            if winner == 1:
                 print("ai_win")
                 ai_win+=1
            else:
                print("base_win")
                baseline_win+=1
        win_list.append(ai_win)
        node_list.append(node)
        f = open("./result_with.txt",'a')
        f.write("result_with:(%d) total_round : (%d)\n"%(ai_win,j))
        f.write("base_win:(%d) total_round : (%d)\n"%(baseline_win,j))
        f.write("using node is (%d)\n"%(node))
        f.close()  
    ai_win = 0
    node = 0
    baseline_win = 0
    for i in range(len(size)):
        baseline_win = 0
        ai_win = 0
        node = 0
        f = open("./result_without.txt","a")
        f.write("(%d) size\n"%(size[i]))
        for j in range(per_game_len):
            game.initialize(size[i],2,j)
            winner,t_node = game.chess_loop()
            node+=t_node
            if winner == 1:
                print("ai_without_win")
                ai_win+=1
            else:
                print("base_win")
                baseline_win+=1
        f = open("./result_without.txt",'a')
        f.write("ai_without_win:(%d) total_round : (%d)\n"%(ai_win,j))
        f.write("base_win:(%d) total_round : (%d)\n"%(baseline_win,j))
        f.write("using node is (%d)\n"%(node))
        f.close()  
        win_without_list.append(ai_win)
        node_without_list.append(node)
    
    plt.figure(1)
    win_pic = plt.subplot(2,1,1)
    node_pic = plt.subplot(2,1,2)
    x_list = [40*i for i in range(len(size))]
    x1_list = [i+10 for i in x_list]
    plt.sca(win_pic)
    plt.bar(x_list,win_list,width=10,label="cnn",fc="r",tick_label=size_list)
    plt.bar(x1_list,win_without_list,width=10,label="no cnn",fc="y")
    plt.legend()
    plt.sca(node_pic)
    plt.bar(x_list,node_list,width=10,label="cnn",fc="r" ,tick_label=size_list)
    plt.bar(x1_list,node_without_list,width=10,label="no cnn",fc="y")
    plt.legend()
    plt.savefig("mode2.jpg")
    plt.show()
    