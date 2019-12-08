# cis667-final
A gobang board game with cnn
# Dependence
--tensorflow 1.14  
-- pythhon 3.7
-- Anaconda3
# file
cnn.py,cnn_1.py,cnn_2.py:the cnn modal with different configuration  
game.py:game play class and show ui board  
performance.py:test ai performance with 5 board size,each size play 100 games  
logs : the log information during cnn training  
mode : the trained model   

# step
1.Create virtual environment and activate(optional)  
```
conda create -n tensorflow
```
```
activate tensorflow
```
2.install tensorflow   
  
3.start game with ui  
```
python game.py
```
This will show a board to play,the ai always play black first

4.run performance to compare the ai with baseline in 5 different board size,each size play 100 games
```
ptyhon performance.py
```
5.show train loss  
```
tensorboard --log = logs/log1 --host=127.0.0.1
```
6.train  
before traiing,download the data here
```
python cnn.py
```
