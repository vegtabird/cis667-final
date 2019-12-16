import tensorflow as tf
from SGFfile import SGF

#0for empty;
#1 for black;
 #2 for white;
 #this is a modal with 1 conv 3*3 and 1024 fc neural
 #path to save the log
logPath = "logs/log3_9"
#path to save the model
modePath = "model/m3_9.ckpt"
#size of board
board_size = 9
class CNN():
    def __init__(self,size):
        self.graph = tf.Graph()
        self.session = tf.InteractiveSession(graph=self.graph)
        self.x = tf.placeholder(tf.float32,[None,size*size])
        self.y = tf.placeholder(tf.float32,[None,size*size])
        self.w_conv1 = self.weight_variable([3,3,1,32])
        self.b_conv1 = self.bias_variable([32])
        self.x_image = tf.reshape(self.x,[-1,size,size,1])
        self.conv1 = tf.nn.relu(self.conv2d(self.x_image,self.w_conv1)+self.b_conv1)
        self.pool1 = self.max_pool(self.conv1)
        max_size = self.pool1.shape[1].value
        self.w_fc1 = self.weight_variable([max_size*max_size*32,1024])
        self.b_fc1 = self.bias_variable([1024])
        self.pool1_flat = tf.reshape(self.pool1,[-1,max_size*max_size*32])
        self.fc1 = tf.nn.relu(tf.matmul(self.pool1_flat,self.w_fc1)+self.b_fc1)


        self.w_fc2 = self.weight_variable([1024,size*size])
        self.b_fc2 = self.bias_variable([size*size])
        self.y_conv = tf.nn.softmax(tf.matmul(self.fc1,self.w_fc2)+self.b_fc2)

        self.sorted_pred = tf.argsort(self.y_conv,direction="DESCENDING")
        self.cross_entropy = -tf.reduce_sum(self.y * tf.log(self.y_conv))
        self.train_step = tf.train.AdamOptimizer(1e-4).minimize(self.cross_entropy)
        self.saver = tf.train.Saver()
        tf.summary.scalar("loss",self.cross_entropy)
        self.merged_summary = tf.summary.merge_all()
        self.session.run(tf.global_variables_initializer())
        self.correction_prediction = tf.equal(tf.argmax(self.y_conv,1),tf.argmax(self.y,1))
    @staticmethod
    def weight_variable(shape):
        return tf.Variable(tf.truncated_normal(shape,stddev=0.1))
    @staticmethod
    def bias_variable(shape):
        return tf.Variable(tf.constant(0.1,shape=shape))
    @staticmethod
    def conv2d(x,weight):
        return tf.nn.conv2d(x,weight,strides=[1,1,1,1],padding='SAME')
    @staticmethod
    def max_pool(x):
        return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

    def expend(self,board):
        new = [[0.0 for i in range(15)] for i in range(15)]
        size = len(board)
        extra = (15-size)/2
        extra = int(extra)
        for i in range(size):
            for l in range(size):
                new[extra+i][extra+l] = board[i][l]
        return new,extra

    def prediction(self,board):
        data = []
        tmp = []
        result = []
        finded = 0
        size = len(board)
        for row in board:
            for point in row:
                tmp.append(point)
        data.append(tmp)
        sorted = self.session.run(self.sorted_pred,feed_dict={self.x:data})
        for dis in sorted[0]:
            col = dis%size
            if dis < size:
                row = 0
            else:
                row = (dis - col)/size
                row = int(row)
            if board[row][col] == 0.0:
                finded += 1
                result.append([row,col])
            if finded >= 10:
                break
        return result
    
    def save(self,path):
        saver = tf.train.Saver(write_version=tf.train.SaverDef.V2)
        saver.save(self.session,path)
    
    def restore(self,path):
        self.saver.restore(self.session,path)
if __name__ == "__main__":
    _cnn = CNN(board_size)
    sgf = SGF()
    batch = 0
    files = sgf.getAllFileName('.\sgf\\')
    train_file = files[:2000]
    summary_writer = tf.summary.FileWriter(logPath)
    batch = 0
    for file in train_file:
        x,y = sgf.transferDataToTrain(file,1)
        n_x,n_y = sgf.shrinkTrainToSize(board_size,x,y)
        _cnn.session.run(_cnn.train_step,feed_dict={_cnn.x:n_x,_cnn.y:n_y})
        summary = _cnn.session.run(_cnn.merged_summary,feed_dict={_cnn.x:n_x,_cnn.y:n_y})

        summary_writer.add_summary(summary,batch)
        batch += 1
        print(batch)
    _cnn.save(modePath)

