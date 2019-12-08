import tensorflow as tf
from SGFfile import SGF
#0for empty;
#1 for black;
#2 for white;
#this is a model with 2 conv and 300 fc neural
class CNN():
    def __init__(self):
        self.session = tf.InteractiveSession()
        self.x = tf.placeholder(tf.float32,[None,225])
        self.y = tf.placeholder(tf.float32,[None,225])
        self.w_conv1 = self.weight_variable([5,5,1,32])
        self.b_conv1 = self.bias_variable([32])
        self.x_image = tf.reshape(self.x,[-1,15,15,1])
        self.conv1 = tf.nn.relu(self.conv2d(self.x_image,self.w_conv1)+self.b_conv1)
        self.pool1 = self.max_pool(self.conv1)

        self.w_conv2 = self.weight_variable([5,5,32,64])
        self.b_conv2 = self.bias_variable([64])
        self.conv2 = tf.nn.relu(self.conv2d(self.pool1,self.w_conv2)+self.b_conv2)
        self.pool2 = self.max_pool(self.conv2)

        self.w_fc1 = self.weight_variable([4*4*64,300])
        self.b_fc1 = self.bias_variable([300])
        self.pool2_flat = tf.reshape(self.pool2,[-1,4*4*64])
        self.fc1 = tf.nn.relu(tf.matmul(self.pool2_flat,self.w_fc1)+self.b_fc1)


        self.w_fc2 = self.weight_variable([300,225])
        self.b_fc2 = self.bias_variable([225])
        self.y_conv = tf.nn.softmax(tf.matmul(self.fc1,self.w_fc2)+self.b_fc2)
        self.sorted_pred = tf.argsort(self.y_conv,direction="DESCENDING")
        self.cross_entropy = -tf.reduce_sum(self.y * tf.log(self.y_conv))
        self.train_step = tf.train.AdamOptimizer(1e-4).minimize(self.cross_entropy)
        tf.summary.scalar("loss",self.cross_entropy)
        self.merged_summary = tf.summary.merge_all()
        self.saver = tf.train.Saver()
        self.session.run(tf.global_variables_initializer())
        correction_prediction = tf.equal(tf.argmax(self.y_conv,1),tf.argmax(self.y,1))
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
        new_board = board
        extra = 0
        if len(board) < 15:
           new_board,extra= self.expend(board) 
        data = []
        tmp = []
        result = []
        finded = 0
        for row in new_board:
            for point in row:
                    tmp.append(point)
        data.append(tmp)
        left_col = extra
        right_col = extra+len(board)-1
        top_row = extra
        bottom_row = extra+len(board)-1
        sorted = self.session.run(self.sorted_pred,feed_dict={self.x:data})
        for dis in sorted[0]:
            col = dis%15
            if dis < 15:
                row = 0
            else:
                row = (dis - col)/15
                row = int(row)
            if col >=left_col and col <= right_col and row >= top_row and row <= bottom_row:
                col = col - extra
                row = row-extra
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
    _cnn = CNN()
    sgf = SGF()
    batch = 0
    files = sgf.getAllFileName('.\sgf\\')
    train_file = files[:2000]
    summary_writer = tf.summary.FileWriter("./logs/log2")
    batch = 0
    for file in train_file:
        x,y = sgf.transferDataToTrain(file,1)
        _cnn.session.run(_cnn.train_step,feed_dict={_cnn.x:x,_cnn.y:y})
        summary = _cnn.session.run(_cnn.merged_summary,feed_dict={_cnn.x:x,_cnn.y:y})

        summary_writer.add_summary(summary,batch)
        batch += 1
        print(batch)
    _cnn.save('.\model\model2.ckpt')   