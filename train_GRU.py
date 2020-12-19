import tensorflow as tf
import numpy as np
import time
import datetime
import os
import network
from sklearn.metrics import average_precision_score,recall_score,precision_score

from sklearn import svm
from tensorflow.contrib.tensorboard.plugins import projector

FLAGS = tf.app.flags.FLAGS#####传递参数

tf.app.flags.DEFINE_string('summary_dir', '.', 'path to store summary')


def main(_):
    # the path to save models
    save_path = './model/'
    print('reading wordembedding')
    wordembedding = np.load('./data/vec.npy')
    print('reading training data')
    train_y = np.load('./data/train_y.npy')
    train_word = np.load('./data/train_word.npy',allow_pickle=True)
    train_pos1 = np.load('./data/train_pos1.npy',allow_pickle=True)
    train_pos2 = np.load('./data/train_pos2.npy',allow_pickle=True)
    settings = network.Settings()
    settings.vocab_size = len(wordembedding)
    settings.num_classes = len(train_y[0])################################
    big_num = settings.big_num##batch_size


    with tf.Graph().as_default():
        sess = tf.Session()####Session()是 Tensorflow 控制和输出文件的执行的语句. 运行 session.run() 可以获得你要得知的运算结果
        with sess.as_default():

            initializer = tf.contrib.layers.xavier_initializer()
            with tf.variable_scope("model", reuse=None, initializer=initializer):###定义作用阈
                m = network.GRU(is_training=True, word_embeddings=wordembedding, settings=settings)
            global_step = tf.Variable(0, name="global_step", trainable=False)###使用变量Variables去更新参数
            optimizer = tf.train.AdamOptimizer(0.0005)#####Adam优化算法,相比于传统SDG,速度更快，不容易陷入局部点
            train_op = optimizer.minimize(m.final_loss, global_step=global_step)###计算梯度并使用
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            saver = tf.train.Saver(max_to_keep=None)
            merged_summary = tf.summary.merge_all()##将所有summary全部保存到磁盘，以便tensorboard显示。
            summary_writer = tf.summary.FileWriter(FLAGS.summary_dir + '/train_loss', sess.graph)
            def train_step(word_batch, pos1_batch, pos2_batch, y_batch, big_num):
                feed_dict = {}
                total_shape = []
                total_num = 0
                total_word = []
                total_pos1 = []
                total_pos2 = []
                #for i in range(len(word_batch)):
                for i in range(len(word_batch)):
                    total_shape.append(total_num)
                    total_num += len(word_batch[i])
                    for word in word_batch[i]:
                        total_word.append(word)
                    for pos1 in pos1_batch[i]:
                        total_pos1.append(pos1)
                    for pos2 in pos2_batch[i]:
                        total_pos2.append(pos2)
                total_shape.append(total_num)
                total_shape = np.array(total_shape)
                total_word = np.array(total_word)
                total_pos1 = np.array(total_pos1)
                total_pos2 = np.array(total_pos2)
                feed_dict[m.total_shape] = total_shape
                feed_dict[m.input_word] = total_word
                feed_dict[m.input_pos1] = total_pos1
                feed_dict[m.input_pos2] = total_pos2
                feed_dict[m.input_y] = y_batch

                temp, step, loss, accuracy,summary, l2_loss, final_loss ,senout1,input_y2= sess.run(
                    [train_op, global_step, m.total_loss, m.accuracy, merged_summary, m.l2_loss, m.final_loss,m.sen_out,m.input_y1],
                    feed_dict)


                global current_step
                current_step = tf.train.global_step(sess, global_step)
                time_str = datetime.datetime.now().isoformat()
                accuracy = np.reshape(np.array(accuracy), (big_num))
                acc = np.mean(accuracy)
                summary_writer.add_summary(summary, step)####写入TB
                # if step % 50== 0:
                tempstr = "{}: step {}, sigmoid_loss {:g}, acc {:g}".format(time_str, step, loss, acc)
                print(tempstr)

            ###################################################################################################
            for one_epoch in range(15):
                temp_order = list(range(len(train_word)))
                np.random.shuffle(temp_order)
                for i in range(int(len(temp_order) / float(settings.big_num))):
                    temp_word = []
                    temp_pos1 = []
                    temp_pos2 = []
                    temp_y = []
                    temp_input = temp_order[i * settings.big_num:(i + 1) * settings.big_num]
                    for k in temp_input:
                        temp_word.append(train_word[k])
                        temp_pos1.append(train_pos1[k])
                        temp_pos2.append(train_pos2[k])
                        temp_y.append(train_y[k])
                    num = 0
                    for single_word in temp_word:
                        num += 1
                    if num > 1500:
                        print('out of range')
                        continue
                    temp_word = np.array(temp_word)
                    temp_pos1 = np.array(temp_pos1)
                    temp_pos2 = np.array(temp_pos2)
                    temp_y = np.array(temp_y)
                    train_step(temp_word, temp_pos1, temp_pos2, temp_y, settings.big_num)
                    current_step = tf.train.global_step(sess, global_step)
                    if current_step>350and current_step % 20== 0:
                        print('saving model')
                        path = saver.save(sess, save_path + 'ATT_GRU_model_128_16v2', global_step=current_step)
                        tempstr1 = 'have saved model to ' + path
                        print(tempstr1)


if __name__ == "__main__":
    tf.app.run()
