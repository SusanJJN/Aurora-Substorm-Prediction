import os
import keras
import keras.backend as K
from keras.models import Sequential, Model
from keras.layers import ConvLSTM2D, Conv3D, Conv2D, Dense, Flatten, BatchNormalization, Input, LSTM, TimeDistributed, Conv2DTranspose, UpSampling2D, MaxPooling2D, merge, Reshape, Lambda
import tensorflow as tf
from keras.models import load_model
from keras.optimizers import RMSprop, Adam, SGD
from data.seq_generator import *
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.callbacks import TensorBoard
from skimage.measure import compare_ssim, compare_psnr
from sklearn.metrics import mean_squared_error
from sklearn import linear_model
import warnings
from sklearn.metrics import mean_absolute_error
from keras.utils import np_utils, multi_gpu_model

img_rows = 120
img_cols = img_rows
img_chns = 1
batch_size = 8
tra_npz_path = '/home/jjn/susan/AuroraPrediction/data/train_npzs/tra/'
test_npz_path = '/home/jjn/susan/AuroraPrediction/data/train_npzs/val/'

class ParallelModelCheckpoint(ModelCheckpoint):
    def __init__(self, model, filepath, monitor='acc', verbose=0,
                 save_best_only=True, save_weights_only=False,
                 mode='auto', period=1):
        self.single_model = model
        super(ParallelModelCheckpoint, self).__init__(filepath, monitor, verbose, save_best_only, save_weights_only,
                                                      mode, period)

    def set_model(self, model):
        super(ParallelModelCheckpoint, self).set_model(self.single_model)

class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
#         self.val_losses = []
        self.ssim = []
#         self.val_ssim = []
 
    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
#         self.val_losses.append(logs.get('val_loss'))
        self.ssim.append(logs.get('ssim'))
#         self.val_ssim.append(logs.get('val_ssim'))
        

def ssim(y_true, y_pred):
    score = tf.image.ssim(y_true, y_pred, 1.0)
    return score

def intensity_score(y_true, y_pred):
    score = tf.keras.losses.MAE(tf.reduce_sum(y_true,[1,2,3]), tf.reduce_sum(y_pred,[1,2,3]))
    return score


def read_data(basic_frames, interval_frames, npz_path, npz_file):

#     npz_files = os.listdir(npz_path)
#     npz_files.sort()

#     for i in range(len(npz_files)):
    basic_seq, next_seq = get_3chn_seq(npz_path, npz_file, basic_frames, interval_frames, img_rows, img_chns)
    basic_images = basic_seq
    next_images = next_seq

    basic_imgs = basic_images.reshape(basic_images.shape[0], basic_frames, img_rows, img_rows, img_chns)
    next_imgs = next_images.reshape(next_images.shape[0], basic_frames, img_rows, img_rows, img_chns)
    basic_imgs = np.transpose(basic_imgs, (0,1,4,3,2))
    next_imgs = np.transpose(next_imgs, (0,1,4,3,2))
    basic_imgs = basic_imgs[:,:interval_frames,:,:,:]
#     print(basic_imgs.shape, next_imgs.shape)

    return basic_imgs, next_imgs


def split_tra_val(basic_frames, interval_frames, input_frames, basic_imgs, next_images, shuffle):
    index = 88 * (120 - basic_frames - interval_frames + 1)
    tra_basic = basic_imgs[:index, :input_frames]
    val_basic = basic_imgs[index:, :input_frames]
    tra_next = next_images[:index]
    val_next = next_images[index:]
    if shuffle:
        np.random.seed(200)
        np.random.shuffle(tra_basic)
        np.random.seed(200)
        np.random.shuffle(tra_next)
    return tra_basic, val_basic, tra_next, val_next

def init_model(input_frames):
    filters = [32, 64, 128, 64, 32]
    images = Input(shape=(input_frames, img_rows, img_cols, img_chns), name='input_images')
    # print(images.shape)

    lstm_1 = ConvLSTM2D(filters=filters[0], kernel_size=(3, 3), strides=1, padding='same', return_sequences=True,
                        name='lstm_1')(images)
    pool_1 = TimeDistributed(MaxPooling2D(), name='pool_1')(lstm_1)
    bn_1 = BatchNormalization(name='bn_1')(pool_1)

    lstm_2 = ConvLSTM2D(filters=filters[1], kernel_size=(3, 3), strides=1, padding='same', return_sequences=True,
                        name='lstm_2')(bn_1)
    pool_2 = TimeDistributed(MaxPooling2D(), name='pool_2')(lstm_2)
    bn_2 = BatchNormalization(name='bn_2')(pool_2)

    lstm_3 = ConvLSTM2D(filters=filters[2], kernel_size=(3, 3), strides=1, padding='same', return_sequences=False,
                        name='lstm_3')(bn_2)
    # pool_3 = TimeDistributed(MaxPooling2D(), name='pool_3')(lstm_3)
    bn_3 = BatchNormalization(name='bn_3')(lstm_3)

    dec_1 = Conv2DTranspose(filters=filters[3], kernel_size=3, strides=1, padding='same', name='dec_1')(bn_3)
    pool_4 = UpSampling2D(name='pool_4')(dec_1)
    bn_4 = BatchNormalization(name='bn_4')(pool_4)

    dec_2 = Conv2DTranspose(filters=filters[4], kernel_size=3, strides=1, padding='same', name='dec_2')(bn_4)
    pool_5 = UpSampling2D(name='pool_5')(dec_2)
    bn_5 = BatchNormalization(name='bn_5')(pool_5)

    dec_3 = Conv2DTranspose(filters=img_chns, kernel_size=3, strides=1, activation='sigmoid', padding='same',
                            name='dec_3')(bn_5)
    output = dec_3

    model = Model(images, output)
    return model


def training(model, tra_basic, val_basic, tra_next, val_next, log_path, epoch_num=10):

    opt = RMSprop(lr=0.001)
    # opt = Adam(lr=0.001)
    model.compile(loss='mean_squared_error', optimizer=opt, metrics=[ssim])
    tensorboard = TensorBoard(log_dir=log_path)
    history = LossHistory()
#     checkpoint = ModelCheckpoint(log_path + 'nice.h5', monitor='val_ssim', mode='auto', save_best_only='True')
#     callback_lists = [tensorboard, checkpoint]

    for i in range(epoch_num):
        print('epoch:', i)
#         model.fit(tra_basic, tra_next, batch_size=8, epochs=1, validation_data=[val_basic, val_next],
#                   callbacks=callback_lists)
        model.fit(tra_basic, tra_next, batch_size=8, epochs=1, validation_data=[val_basic, val_next],
                  callbacks=[history, tensorboard])
#         model.save(log_path + str(i) + '.h5')
        model.save_weights(log_path + str(i) + '.h5')
#         print(history.losses)
        with open(log_path+'train_loss.txt', 'a') as f:
            f.write(str(history.losses)+'\n')
#         with open(log_path+'val_loss.txt', 'a') as f:
#             f.write(str(history.val_losses)+'\n')
        with open(log_path+'train_ssim.txt', 'a') as f:
            f.write(str(history.ssim)+'\n')
#         with open(log_path+'val_ssim.txt', 'a') as f:
#             f.write(str(history.val_ssim)+'\n')


def train_on(model, tra_basic, val_basic, tra_next, val_next, log_path, epoch_num):

    opt = RMSprop(lr=0.001)
    # opt = Adam(lr=0.001)
    model.compile(loss='mean_squared_error', optimizer=opt, metrics=[ssim])
    tensorboard = TensorBoard(log_dir=log_path)
    checkpoint = ModelCheckpoint(log_path + 'nice.h5', monitor='val_ssim', mode='auto', save_best_only='True')
    callback_lists = [tensorboard, checkpoint]

    for i in range(epoch_num):
        print('epoch:', 100-epoch_num+i)
        model.fit(tra_basic, tra_next, batch_size=8, epochs=1, validation_data=[val_basic, val_next],
                  callbacks=callback_lists)
        model.save(log_path + str(100-epoch_num+i) + '.h5')


def test(test_basic, test_next, input_frames, model_path):
    opt = RMSprop(lr=0.001)
    model = init_model(input_frames)
    model.load_weights(model_path, by_name=True)
    model.compile(loss='mean_squared_error', optimizer=opt, metrics=[ssim, intensity_score])
    scores = model.evaluate(test_basic, test_next, batch_size=1)

    # parallel_model = multi_gpu_model(model, gpus=2)
    # parallel_model.load_weights(model_path, by_name=True)
    # parallel_model.compile(loss='mean_squared_error', optimizer=opt, metrics=[ssim, intensity_score])
    # scores = parallel_model.evaluate(test_basic, test_next, batch_size=32)
    return scores


def lr_model(frame, true_sum, pred_sum):
    x = pred_sum
    #     print(x.shape)
    y = true_sum
    m = linear_model.LinearRegression(fit_intercept=False)
    m.fit(x, y)
    a = m.coef_
    b = m.intercept_
    print(m.coef_, m.intercept_)
    new_y = m.predict(x)
    # plt.figure()
    # plt.plot(x, y, '.', label='ori_sum')
    # plt.plot(x, new_y, marker='+', label='corr_sum')
    # y0 = a * 400
    # plt.annotate(r'$y=%.2fx$' % a, xy=(400, y0), xycoords='data', xytext=(+20, +0),
    #              textcoords='offset points', fontsize=12)
    # plt.xlabel('pred_sum')
    # plt.ylabel('true_sum')
    # plt.legend()
    # plt.title('frame ' + str(frame))
    #
    # plt.savefig('/home/jjn/susan/Substorm/new_scripts/results/0522/lr_models/'+str(frame))
    # plt.show()

    return m


def training_results_seq(model, result_path, tra_basic, tra_next, frame):
    # tra_next.shape = (n,30,64,64,1)
    warnings.filterwarnings("ignore")

    for i in range(tra_basic.shape[0]):
        # print(i)
        gen_imgs = model.predict(tra_basic[i:i + 1])
        last_sum = np.sum(tra_basic[i, -1, :, :, 0])
        true_sum = np.sum(tra_next[i, frame, :,:,0])
        pred_sum = np.sum(gen_imgs[0, :, :, 0])
        ssim = compare_ssim(tra_next[i, frame, :,:,0], gen_imgs[0, :, :, 0], data_range=1)
        mse = mean_squared_error(tra_next[i, frame, :,:,0], gen_imgs[0, :, :, 0])
        with open(result_path + 'last_sum.txt', 'a') as f:
            f.write(str(last_sum) + '\n')
        with open(result_path + 'true_sum.txt', 'a') as f:
            f.write(str(true_sum) + '\n')
        with open(result_path + 'pred_sum.txt', 'a') as f:
            f.write(str(pred_sum) + '\n')
        with open(result_path + 'ssim.txt', 'a') as f:
            f.write(str(ssim) + '\n')

        with open(result_path + 'mse.txt', 'a') as f:
            f.write(str(mse) + '\n')


def training_results(model, result_path, tra_basic, tra_next):
    # tra_next.shape = (n,64,64,1)
    warnings.filterwarnings("ignore")

    for i in range(tra_basic.shape[0]):
        # print(i)
        gen_imgs = model.predict(tra_basic[i:i + 1])
        last_sum = np.sum(tra_basic[i, -1, :, :, 0])
        true_sum = np.sum(tra_next[i])
        pred_sum = np.sum(gen_imgs[0, :, :, 0])
        ssim = compare_ssim(tra_next[i, :, :, 0], gen_imgs[0, :, :, 0], data_range=1)
        mse = mean_squared_error(tra_next[i, :, :, 0], gen_imgs[0, :, :, 0])
        with open(result_path + 'last_sum.txt', 'a') as f:
            f.write(str(last_sum) + '\n')
        with open(result_path + 'true_sum.txt', 'a') as f:
            f.write(str(true_sum) + '\n')
        with open(result_path + 'pred_sum.txt', 'a') as f:
            f.write(str(pred_sum) + '\n')
        with open(result_path + 'ssim.txt', 'a') as f:
            f.write(str(ssim) + '\n')

        with open(result_path + 'mse.txt', 'a') as f:
            f.write(str(mse) + '\n')

def get_correction_factor(result_path, frame):
    true_data = pd.read_csv(result_path + 'true_sum.txt')
    true_sum = true_data.values

    pred_data = pd.read_csv(result_path + 'pred_sum.txt')
    pred_sum = pred_data.values

    m = lr_model(frame, true_sum, pred_sum)
    return m.coef_


# def test_results_seq(model, result_path, val_basic, val_next, intensity_factor, frame):
def test_results_seq(model, result_path, val_basic, val_next, frame):
    for i in range(val_basic.shape[0]):
        # print(i)
        gen_imgs = model.predict(val_basic[i:i + 1])
        # corr_imgs = gen_imgs * intensity_factor

        last_sum = np.sum(val_basic[i, -1, :, :, 0])
        true_sum = np.sum(val_next[i, frame, :, :, 0])
        pred_sum = np.sum(gen_imgs[0, :, :, 0])
        # corr_sum = np.sum(corr_imgs[0, :, :, 0])
        ssim = compare_ssim(val_next[i, frame, :, :, 0], gen_imgs[0, :, :, 0], data_range=1)
        # corr_ssim = compare_ssim(val_next[i, frame, :, :, 0], corr_imgs[0, :, :, 0], data_range=1)
        mse = mean_squared_error(val_next[i, frame, :, :, 0], gen_imgs[0, :, :, 0])
        # corr_mse = mean_squared_error(val_next[i, frame, :, :, 0], corr_imgs[0, :, :, 0])

        with open(result_path + 'last_sum.txt', 'a') as f:
            f.write(str(last_sum) + '\n')
        with open(result_path + 'true_sum.txt', 'a') as f:
            f.write(str(true_sum) + '\n')
        with open(result_path + 'pred_sum.txt', 'a') as f:
            f.write(str(pred_sum) + '\n')
        # with open(result_path + 'corr_sum.txt', 'a') as f:
        #     f.write(str(corr_sum) + '\n')

        with open(result_path + 'ssim.txt', 'a') as f:
            f.write(str(ssim) + '\n')
        # with open(result_path + 'corr_ssim.txt', 'a') as f:
        #     f.write(str(corr_ssim) + '\n')

        with open(result_path + 'mse.txt', 'a') as f:
            f.write(str(mse) + '\n')
        # with open(result_path + 'corr_mse.txt', 'a') as f:
        #     f.write(str(corr_mse) + '\n')


def test_basic_seq(result_path, val_basic, frame):
    for i in range(val_basic.shape[0]):


        last_sum = np.sum(val_basic[i, frame, :, :, 0])


        with open(result_path + 'basic_sum.txt', 'a') as f:
            f.write(str(last_sum) + '\n')


def test_results(model, result_path, val_basic, val_next, intensity_factor):
    for i in range(val_basic.shape[0]):
        # print(i)
        gen_imgs = model.predict(val_basic[i:i + 1])
        corr_imgs = gen_imgs * intensity_factor

        last_sum = np.sum(val_basic[i, -1, :, :, 0])
        true_sum = np.sum(val_next[i])
        pred_sum = np.sum(gen_imgs[0, :, :, 0])
        corr_sum = np.sum(corr_imgs[0, :, :, 0])
        ssim = compare_ssim(val_next[i, :, :, 0], gen_imgs[0, :, :, 0], data_range=1)
        corr_ssim = compare_ssim(val_next[i, :, :, 0], corr_imgs[0, :, :, 0], data_range=1)
        mse = mean_squared_error(val_next[i, :, :, 0], gen_imgs[0, :, :, 0])
        corr_mse = mean_squared_error(val_next[i, :, :, 0], corr_imgs[0, :, :, 0])

        with open(result_path + 'last_sum.txt', 'a') as f:
            f.write(str(last_sum) + '\n')
        with open(result_path + 'true_sum.txt', 'a') as f:
            f.write(str(true_sum) + '\n')
        with open(result_path + 'pred_sum.txt', 'a') as f:
            f.write(str(pred_sum) + '\n')
        # with open(result_path + 'corr_sum.txt', 'a') as f:
        #     f.write(str(corr_sum) + '\n')

        with open(result_path + 'ssim.txt', 'a') as f:
            f.write(str(ssim) + '\n')
        # with open(result_path + 'corr_ssim.txt', 'a') as f:
        #     f.write(str(corr_ssim) + '\n')

        with open(result_path + 'mse.txt', 'a') as f:
            f.write(str(mse) + '\n')
        # with open(result_path + 'corr_mse.txt', 'a') as f:
        #     f.write(str(corr_mse) + '\n')

def show_results(result_path):
    last_data = pd.read_csv(result_path + 'last_sum.txt', header=None)
    last_sum = last_data.values

    true_data = pd.read_csv(result_path + 'true_sum.txt', header=None)
    true_sum = true_data.values
    # print(true_sum.shape)

    pred_data = pd.read_csv(result_path + 'pred_sum.txt', header=None)
    pred_sum = pred_data.values
    # print(true_sum.shape)

    # corr_data = pd.read_csv(result_path + 'corr_sum.txt')
    # corr_sum = corr_data.values

    ssim_data = pd.read_csv(result_path + 'ssim.txt', header=None)
    ssim_list = ssim_data.values

    # corr_ssim_data = pd.read_csv(result_path + 'corr_ssim.txt')
    # corr_ssim_list = corr_ssim_data.values

    mse_data = pd.read_csv(result_path + 'mse.txt', header=None)
    mse_list = mse_data.values

    # corr_mse_data = pd.read_csv(result_path + 'corr_mse.txt')
    # corr_mse_list = corr_mse_data.values

    avg_last = np.mean(last_sum)
    avg_true = np.mean(true_sum)
    avg_pred = np.mean(pred_sum)
    # avg_corr = np.mean(corr_sum)
    # print('avg_last:', avg_last, 'avg_true:', avg_true, 'avg_pred:', avg_pred, 'avg_corr:', avg_corr)
    print('avg_last:', avg_last, 'avg_true:', avg_true, 'avg_pred:', avg_pred)

    pred_mae = abs(avg_true - avg_pred)
    # corr_mae = abs(avg_true - avg_corr)
    print('pred_mae:', pred_mae)

    avg_ssim = np.mean(ssim_list)
    # avg_corr_ssim = np.mean(corr_ssim_list)
    print('avg_ssim:', avg_ssim)

    avg_mse = np.mean(mse_list)
    # avg_corr_mse = np.mean(corr_mse_list)
    print('avg_mse:', avg_mse)

    return pred_mae, avg_ssim, avg_mse


def mlt_distribute_show(avg_true, avg_pred, avg_copy, mlt, frame, ylim, save_path):
    latitudes = [i for i in range(30)]

    plt.figure()
    new_x = ['0', '90', '85', '80', '75', '70', '65', '60']
    plt.xticks(latitudes, labels=new_x)
    ax = plt.gca()
    ax.xaxis.set_major_locator(plt.MultipleLocator(5))
    ax.yaxis.set_major_locator(plt.MultipleLocator(0.02))
    # if frame == 1:
    #     plt.title('average intensity profile on '+str(mlt)+'MLT of 3rd minute')
    # else:
    #     plt.title('average intensity profile on ' + str(mlt) + 'MLT of '+str(3*frame+3)+'th minute')
    plt.xlabel('latitude')
    plt.ylabel('pixel intensity')
#     plt.title('average intensity profile at '+str(mlt)+'MLT of frame '+str(frame))
    plt.ylim(top=ylim)
    
    plt.plot(latitudes, avg_true, marker='.', label='ground truth')
    plt.plot(latitudes, avg_pred, marker='.', label='predicted')
    # plt.plot(latitudes, avg_copy, label='copy last frame')
    plt.legend(loc='upper left')
    plt.savefig(save_path+str(mlt)+'MLT/'+str(frame)+'.jpg')


def mlt_distribute_contrast(avg_true, avg_preds, mlt, frame, ylim, save_path):
    latitudes = [i for i in range(30)]
    plt.figure()
    new_x = ['0', '90', '85', '80', '75', '70', '65', '60']
    plt.xticks(latitudes, labels=new_x)
    ax = plt.gca()
    ax.xaxis.set_major_locator(plt.MultipleLocator(5))
    plt.title('average intensity distribution on '+str(mlt)+'MLT of frame'+str(frame))
    plt.xlabel('latitude')
    plt.ylabel('intensity')
    plt.ylim(top=ylim)
    plt.plot(latitudes, avg_true, label='ground truth')
    plt.plot(latitudes, avg_preds[0], label='predicted by 5f-input model')
    plt.plot(latitudes, avg_preds[1], label='predicted by 10f-input model')
    plt.plot(latitudes, avg_preds[2], label='predicted by 15f-input model')
    plt.legend(loc='upper left')
    plt.savefig(save_path+str(mlt)+'MLT/'+str(frame)+'.jpg')


def mlt_distribute_cases(true_dist, pred_dist, mlt, frame, index, save_path):
    latitudes = [i for i in range(30)]

    plt.figure()
    new_x = ['0', '90', '85', '80', '75', '70', '65', '60']
    plt.xticks(latitudes, labels=new_x)
    ax = plt.gca()
    ax.xaxis.set_major_locator(plt.MultipleLocator(5))
#     plt.title('average intensity distribution on '+str(mlt)+'MLT of frame '+str(frame)+'in case ' +str(index))
    plt.xlabel('latitude')
    plt.ylabel('pixel intensity')
    # plt.ylim(top=ylim)
    plt.plot(latitudes, true_dist, marker='.', label='ground truth')
    plt.plot(latitudes, pred_dist, marker='.', label='predicted')
    # plt.plot(latitudes, avg_copy, label='copy last frame')
    plt.legend(loc='upper left')
    plt.savefig(save_path+str(index)+'/'+str(mlt)+'/'+str(frame)+'.jpg')


def mlt_distribute_case(true_dist, pred_dist, mlt, frame, ylim, index, save_path):
    latitudes = [i for i in range(30)]

    plt.figure()
    new_x = ['0', '90', '85', '80', '75', '70', '65', '60']
    plt.xticks(latitudes, labels=new_x)
    ax = plt.gca()
    ax.xaxis.set_major_locator(plt.MultipleLocator(5))
    plt.title('average intensity distribution on '+str(mlt)+'MLT of frame'+str(frame))
    plt.xlabel('latitude')
    plt.ylabel('pixel intensity')
    plt.ylim(top=ylim)
    plt.plot(latitudes, true_dist, marker='.', label='ground truth')
    plt.plot(latitudes, pred_dist, marker='.', label='predicted')
    # plt.plot(latitudes, avg_copy, label='copy last frame')
    plt.legend(loc='upper left')
#     plt.savefig(save_path+str(index)+'/'+str(mlt)+'/'+str(frame)+'.jpg')



def get_pixels(data):
    pixel_list = []
    for i in range(30, 60):
        pixel_list.append(data[i, 30])

    return pixel_list

def case_test_1(index, model_no, result_path):
    mse_lists = np.array([])
    ssim_lists = np.array([])
    true_sums = []
    pred_sums = []
    mae_list = np.array([])
    for frame in range(1,11):
    #     print(frame)
#         result_path = '/home/jjn/susan/AuroraPrediction/results/0828/'+str(frame)+'/'+str(model_no[frame-1])+'/'
    #     print(result_path)
        true_data = pd.read_csv(result_path + 'true_sum.txt', header=None)
        true_sum = true_data.values
        pred_data = pd.read_csv(result_path + 'pred_sum.txt', header=None)
        pred_sum = pred_data.values
        ssim_data = pd.read_csv(result_path + 'ssim.txt', header=None)
        ssim_list = ssim_data.values
        mse_data = pd.read_csv(result_path + 'mse.txt', header=None)
        mse_list = mse_data.values

        ssim_lists = np.append(ssim_lists, ssim_list[index])
        true_sums.append(true_sum[index])
        pred_sums.append(pred_sum[index])
        mse_lists = np.append(mse_lists, mse_list[index])
        
        mae_list = np.append(mae_list, abs(true_sum[index] - pred_sum[index]))
    avg_mae = np.mean(mae_list)
    avg_ssim = np.mean(ssim_lists)
    avg_mse = np.mean(mse_lists)

#     frames = [i for i in range(1, 41)]
#     plt.figure()
#     plt.plot(frames, true_sums, '.-', label='ground truth')
#     plt.plot(frames, pred_sums, '.-', label='predicted')
#     plt.legend()

#     plt.figure()
#     plt.plot(frames, ssim_lists, '.-', color='orange')
    return avg_mae, avg_ssim, avg_mse
#     return ssim_lists, true_sums, pred_sums, avg_mae