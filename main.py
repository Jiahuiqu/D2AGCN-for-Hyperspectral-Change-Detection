import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import random
from matplotlib import cm
import spectral as spy
from sklearn import metrics
import time
from sklearn import preprocessing
import torch
import GCN
import slic_simple

import warnings
warnings.filterwarnings("ignore")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

Seed_List = [0]  # Random seed points


def Draw_Classification_Map(label, name: str, scale: float = 4.0, dpi: int = 400):
    '''
    get classification map , then save to given path
    :param label: classification label, 2D
    :param name: saving path and file's name
    :param scale: scale of image. If equals to 1, then saving-size is just the label-size
    :param dpi: default is OK
    :return: null
    '''
    fig, ax = plt.subplots()
    numlabel = np.array(label)
    v = spy.imshow(classes=numlabel.astype(np.int16), fignum=fig.number)
    ax.set_axis_off()
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    fig.set_size_inches(label.shape[1] * scale / dpi, label.shape[0] * scale / dpi)
    foo_fig = plt.gcf()  # 'get current figure'
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    foo_fig.savefig(name + '.png', format='png', transparent=True, dpi=dpi, pad_inches=0)
    pass


def GT_To_One_Hot(gt, class_count):
    '''
    Convet Gt to one-hot labels
    :param gt:
    :param class_count:
    :return:
    '''
    GT_One_Hot = []
    for i in range(gt.shape[0]):
        for j in range(gt.shape[1]):
            temp = np.zeros(class_count, dtype=np.float32)
            if gt[i, j] != 0:
                temp[int(gt[i, j]) - 1] = 1
            GT_One_Hot.append(temp)
    GT_One_Hot = np.reshape(GT_One_Hot, [height, width, class_count])
    return GT_One_Hot


def compute_loss(predict: torch.Tensor, reallabel_onehot: torch.Tensor, reallabel_mask: torch.Tensor):
    real_labels = reallabel_onehot
    we = -torch.mul(real_labels, torch.log(predict))
    we = torch.mul(we, reallabel_mask)
    pool_cross_entropy = torch.sum(we)
    return pool_cross_entropy


def compute_crossentropy(index, data, gt, criteon):
    data_new = torch.cat([data[ind, :].unsqueeze(0) for ind in index], dim=0)
    return criteon(data_new, gt.long())


def SAM_vector(H_i, H_j):
    SAM_value = np.math.sqrt(torch.dot(H_i, H_i)) * np.math.sqrt(torch.dot(H_j, H_j))
    SAM_value = torch.tensor(SAM_value)
    SAM_value = torch.dot(H_i, H_j) / SAM_value
    if SAM_value > 1 or SAM_value < -1:
        SAM_value = 1
    SAM_value = np.math.acos(SAM_value)
    SAM_value = torch.tensor(SAM_value)
    return SAM_value


def cross_superpixel_init(Q, A, net_input, idx_temp):
    # Calculate the initial value of the superpixel
    I = torch.eye(A.shape[0], A.shape[0], requires_grad=False).to(device)
    A = A + I

    [h, w, c] = net_input.shape

    # print(self.Q.shape)
    norm_col_Q = torch.sum(Q, 0, keepdim=True)
    x_HSI_flatten = net_input.reshape([h * w, -1])
    superpixels_flatten_HSI = torch.mm(Q.t(), x_HSI_flatten)

    V_HSI = superpixels_flatten_HSI / norm_col_Q.t().to(device)
    Z_HSI = x_HSI_flatten

    Q = Q.cpu().numpy()
    A = A.cpu().numpy()
    P_HSI = torch.zeros([h * w, superpixels_flatten_HSI.shape[0]]).to(device)
    for i in range(h * w):
        j = np.argwhere(Q[i])  # Find which superpixel block node i is in (j)
        index = np.argwhere(A[j].reshape(1, A.shape[0]))[:, 1]  # 1-Order neighbors of the jth superpixel block
        for k in range(len(index)):
            # print(index[k])
            P_HSI[i, index[k]] = torch.exp(-0.2 * SAM_vector(Z_HSI[i, :], V_HSI[k, :]))
            # P_HSI[i, index[k]] = torch.exp(-0.2 * torch.pow(torch.norm(Z_HSI[i, :] - V_HSI[k, :]), 2))

    P_H = P_HSI.cpu().numpy()
    sio.savemat('P_H_' + str(idx_temp) + '.mat', {'P_H': P_H})

    norm_col_P_HSI = torch.sum(P_HSI, 0, keepdim=True)

    H_HSI = torch.mm(P_HSI.t(), x_HSI_flatten)
    H_HSI = H_HSI / norm_col_P_HSI.t().to(device)

    return H_HSI, P_HSI


def evaluate_performance(network_output, train_samples_gt, train_samples_gt_onehot, require_AA_KPP=False,
                         printFlag=True):
    if False == require_AA_KPP:
        with torch.no_grad():
            available_label_idx = (train_samples_gt != 0).float()
            available_label_count = available_label_idx.sum()
            correct_prediction = torch.where(
                torch.argmax(network_output, 1) == torch.argmax(train_samples_gt_onehot, 1),
                available_label_idx, zeros).sum()
            OA = correct_prediction.cpu() / available_label_count

            return OA
    else:
        with torch.no_grad():
            # OA
            available_label_idx = (train_samples_gt != 0).float()
            available_label_count = available_label_idx.sum()
            correct_prediction = torch.where(
                torch.argmax(network_output, 1) == torch.argmax(train_samples_gt_onehot, 1),
                available_label_idx, zeros).sum()
            OA = correct_prediction.cpu() / available_label_count
            OA = OA.cpu().numpy()

            zero_vector = np.zeros([class_count])
            output_data = network_output.cpu().numpy()
            train_samples_gt = train_samples_gt.cpu().numpy()
            train_samples_gt_onehot = train_samples_gt_onehot.cpu().numpy()

            output_data = np.reshape(output_data, [m * n, class_count])
            idx = np.argmax(output_data, axis=-1)
            for z in range(output_data.shape[0]):
                if ~(zero_vector == output_data[z]).all():
                    idx[z] += 1
            # idx = idx + train_samples_gt
            count_perclass = np.zeros([class_count])
            correct_perclass = np.zeros([class_count])
            for x in range(len(train_samples_gt)):
                if train_samples_gt[x] != 0:
                    count_perclass[int(train_samples_gt[x] - 1)] += 1
                    if train_samples_gt[x] == idx[x]:
                        correct_perclass[int(train_samples_gt[x] - 1)] += 1
            test_AC_list = correct_perclass / count_perclass
            test_AA = np.average(test_AC_list)

            # Kappa
            test_pre_label_list = []
            test_real_label_list = []
            output_data = np.reshape(output_data, [m * n, class_count])
            idx = np.argmax(output_data, axis=-1)
            idx = np.reshape(idx, [m, n])
            for ii in range(m):
                for jj in range(n):
                    if Test_GT[ii][jj] != 0:
                        test_pre_label_list.append(idx[ii][jj] + 1)
                        test_real_label_list.append(Test_GT[ii][jj])
            test_pre_label_list = np.array(test_pre_label_list)
            test_real_label_list = np.array(test_real_label_list)
            kappa = metrics.cohen_kappa_score(test_pre_label_list.astype(np.int16),
                                              test_real_label_list.astype(np.int16))
            test_kpp = kappa

            # print
            if printFlag:
                print("test OA=", OA, "AA=", test_AA, 'kpp=', test_kpp)
                print('acc per class:')
                print(test_AC_list)

            OA_ALL.append(OA)
            AA_ALL.append(test_AA)
            KPP_ALL.append(test_kpp)
            AVG_ALL.append(test_AC_list)

            # save data
            f = open('results/' + dataset_name + '_results.txt', 'a+')
            str_results = '\n======================' \
                          + " learning rate=" + str(learning_rate) \
                          + " epochs=" + str(max_epoch) \
                          + " train ratio=" + str(train_ratio) \
                          + " val ratio=" + str(val_ratio) \
                          + " ======================" \
                          + "\nOA=" + str(OA) \
                          + "\nAA=" + str(test_AA) \
                          + '\nkpp=' + str(test_kpp) \
                          + '\nacc per class:' + str(test_AC_list) + "\n"
            f.write(str_results)
            f.close()
            return OA


for (FLAG, curr_train_ratio, Scale) in [(0, 200, 450)]:
    torch.cuda.empty_cache()
    OA_ALL = []
    AA_ALL = []
    KPP_ALL = []
    AVG_ALL = []
    Train_Time_ALL = []
    Test_Time_ALL = []

    # load dataset

    data_mat_1 = sio.loadmat('data/barbara_2013.mat')
    data_mat_2 = sio.loadmat('data/barbara_2014.mat')
    data1 = data_mat_1['HypeRvieW']
    data2 = data_mat_2['HypeRvieW']
    gt_mat = sio.loadmat('data/label.mat')
    gt = gt_mat['HypeRvieW']

    samples_type = ['ratio', 'same_num'][FLAG]  # ratio or number

    # parameter preset
    val_ratio = 0.001
    class_count = 2  # class
    learning_rate = 0.0001  # learning rate
    max_epoch = 1000  # iterations
    dataset_name = "Barbara"  # dataset name
    train_ratio = 0.005 if samples_type == "ratio" else curr_train_ratio
    superpixel_scale = Scale
    train_samples_per_class = curr_train_ratio
    val_samples = class_count
    cmap = cm.get_cmap('jet', class_count + 1)
    plt.set_cmap(cmap)
    m, n, d = data1.shape  # shape of dataset

    # standardization
    height, width, bands = data1.shape
    data1 = np.reshape(data1, [height * width, bands])
    minMax = preprocessing.StandardScaler()
    data1 = minMax.fit_transform(data1)
    data1 = np.reshape(data1, [height, width, bands])
    # orig_data = data2
    height, width, bands = data2.shape
    data2 = np.reshape(data2, [height * width, bands])
    minMax = preprocessing.StandardScaler()
    data2 = minMax.fit_transform(data2)
    data2 = np.reshape(data2, [height, width, bands])

    # print the number of samples per class
    gt_reshape = np.reshape(gt, [-1])
    for i in range(class_count):
        idx = np.where(gt_reshape == i + 1)[-1]
        samplesCount = len(idx)
        print(samplesCount)

    for curr_seed in Seed_List:
        random.seed(curr_seed)
        gt_reshape = np.reshape(gt, [-1])
        train_rand_idx = []
        val_rand_idx = []
        if samples_type == 'ratio':  # Take a certain percentage of training
            for i in range(class_count):
                idx = np.where(gt_reshape == i + 1)[-1]
                samplesCount = len(idx)
                rand_list = [i for i in range(samplesCount)]
                rand_idx = random.sample(rand_list,
                                         np.ceil(samplesCount * train_ratio).astype('int32'))  # 随机数数量 四舍五入(改为上取整)
                rand_real_idx_per_class = idx[rand_idx]
                train_rand_idx.append(rand_real_idx_per_class)
            train_rand_idx = np.array(train_rand_idx)
            train_data_index = []
            for c in range(train_rand_idx.shape[0]):
                a = train_rand_idx[c]
                for j in range(a.shape[0]):
                    train_data_index.append(a[j])
            train_data_index = np.array(train_data_index)
            sio.savemat('train_index.mat', {'index': train_data_index})

            train_data_index = set(train_data_index)
            all_data_index = [i for i in range(len(gt_reshape))]
            all_data_index = set(all_data_index)

            # the index of the background
            background_idx = np.where(gt_reshape == 0)[-1]
            background_idx = set(background_idx)
            test_data_index = all_data_index - train_data_index - background_idx

            # the validation set
            val_data_count = int(val_ratio * (len(test_data_index) + len(train_data_index)))
            val_data_index = random.sample(test_data_index, val_data_count)
            val_data_index = set(val_data_index)
            test_data_index = test_data_index - val_data_index

            test_data_index = list(test_data_index)
            train_data_index = list(train_data_index)
            val_data_index = list(val_data_index)
            sio.savemat('test_index.mat', {'index': test_data_index})

        if samples_type == 'same_num':
            for i in range(class_count):
                idx = np.where(gt_reshape == i + 1)[-1]
                samplesCount = len(idx)
                real_train_samples_per_class = train_samples_per_class
                rand_list = [i for i in range(samplesCount)]
                if real_train_samples_per_class > samplesCount:
                    real_train_samples_per_class = samplesCount
                rand_idx = random.sample(rand_list,
                                         real_train_samples_per_class)
                rand_real_idx_per_class_train = idx[rand_idx[0:real_train_samples_per_class]]
                train_rand_idx.append(rand_real_idx_per_class_train)
            train_rand_idx = np.array(train_rand_idx)
            val_rand_idx = np.array(val_rand_idx)
            train_data_index = []
            for c in range(train_rand_idx.shape[0]):
                a = train_rand_idx[c]
                for j in range(a.shape[0]):
                    train_data_index.append(a[j])
            train_data_index = np.array(train_data_index)

            train_data_index = set(train_data_index)
            all_data_index = [i for i in range(len(gt_reshape))]
            all_data_index = set(all_data_index)

            # the index of the background
            background_idx = np.where(gt_reshape == 0)[-1]
            background_idx = set(background_idx)
            test_data_index = all_data_index - train_data_index - background_idx

            # the validation set
            val_data_count = int(val_samples)
            val_data_index = random.sample(test_data_index, val_data_count)
            val_data_index = set(val_data_index)

            test_data_index = test_data_index - val_data_index

            test_data_index = list(test_data_index)
            train_data_index = list(train_data_index)
            val_data_index = list(val_data_index)

        # train set
        train_samples_gt = np.zeros(gt_reshape.shape)
        for i in range(len(train_data_index)):
            train_samples_gt[train_data_index[i]] = gt_reshape[train_data_index[i]]
            pass

        # test set
        test_samples_gt = np.zeros(gt_reshape.shape)
        for i in range(len(test_data_index)):
            test_samples_gt[test_data_index[i]] = gt_reshape[test_data_index[i]]
            pass

        Test_GT = np.reshape(test_samples_gt, [m, n])  # 测试样本图

        # validation set
        val_samples_gt = np.zeros(gt_reshape.shape)
        for i in range(len(val_data_index)):
            val_samples_gt[val_data_index[i]] = gt_reshape[val_data_index[i]]
            pass

        train_samples_gt = np.reshape(train_samples_gt, [height, width])
        test_samples_gt = np.reshape(test_samples_gt, [height, width])
        val_samples_gt = np.reshape(val_samples_gt, [height, width])

        train_samples_gt_onehot = GT_To_One_Hot(train_samples_gt, class_count)
        test_samples_gt_onehot = GT_To_One_Hot(test_samples_gt, class_count)
        val_samples_gt_onehot = GT_To_One_Hot(val_samples_gt, class_count)

        train_samples_gt_onehot = np.reshape(train_samples_gt_onehot, [-1, class_count]).astype(int)
        test_samples_gt_onehot = np.reshape(test_samples_gt_onehot, [-1, class_count]).astype(int)
        val_samples_gt_onehot = np.reshape(val_samples_gt_onehot, [-1, class_count]).astype(int)

        # one-hot
        # train set
        train_label_mask = np.zeros([m * n, class_count])
        temp_ones = np.ones([class_count])
        train_samples_gt = np.reshape(train_samples_gt, [m * n])
        for i in range(m * n):
            if train_samples_gt[i] != 0:
                train_label_mask[i] = temp_ones
        train_label_mask = np.reshape(train_label_mask, [m * n, class_count])

        # test set
        test_label_mask = np.zeros([m * n, class_count])
        temp_ones = np.ones([class_count])
        test_samples_gt = np.reshape(test_samples_gt, [m * n])
        for i in range(m * n):
            if test_samples_gt[i] != 0:
                test_label_mask[i] = temp_ones
        test_label_mask = np.reshape(test_label_mask, [m * n, class_count])

        # validation set
        val_label_mask = np.zeros([m * n, class_count])
        temp_ones = np.ones([class_count])
        val_samples_gt = np.reshape(val_samples_gt, [m * n])
        for i in range(m * n):
            if val_samples_gt[i] != 0:
                val_label_mask[i] = temp_ones
        val_label_mask = np.reshape(val_label_mask, [m * n, class_count])

        ls = slic_simple.SP_SLIC(data1, data2, np.reshape(train_samples_gt, [height, width]), class_count - 1)
        tic0 = time.time()
        ##### Q1 = Q2
        Q1, S1, A1, Q2, S2, A2, Seg = ls.simple_superpixel_no_LDA(scale=superpixel_scale)
        toc0 = time.time()
        SLIC_Time = toc0 - tic0

        print("SLIC costs time: {}".format(SLIC_Time))
        Q1 = torch.from_numpy(Q1).to(device)
        A1 = torch.from_numpy(A1).to(device)
        Q2 = torch.from_numpy(Q2).to(device)
        A2 = torch.from_numpy(A2).to(device)

        # transform to GPU
        train_samples_gt = torch.from_numpy(train_samples_gt.astype(np.float32)).to(device)
        test_samples_gt = torch.from_numpy(test_samples_gt.astype(np.float32)).to(device)
        val_samples_gt = torch.from_numpy(val_samples_gt.astype(np.float32)).to(device)

        train_samples_gt_onehot = torch.from_numpy(train_samples_gt_onehot.astype(np.float32)).to(device)
        test_samples_gt_onehot = torch.from_numpy(test_samples_gt_onehot.astype(np.float32)).to(device)
        val_samples_gt_onehot = torch.from_numpy(val_samples_gt_onehot.astype(np.float32)).to(device)

        train_label_mask = torch.from_numpy(train_label_mask.astype(np.float32)).to(device)
        test_label_mask = torch.from_numpy(test_label_mask.astype(np.float32)).to(device)
        val_label_mask = torch.from_numpy(val_label_mask.astype(np.float32)).to(device)

        net_input_1 = np.array(data1, np.float32)
        net_input_1 = torch.from_numpy(net_input_1.astype(np.float32)).to(device)
        net_input_2 = np.array(data2, np.float32)
        net_input_2 = torch.from_numpy(net_input_2.astype(np.float32)).to(device)

        zeros = torch.zeros([m * n]).to(device).float()

        net = GCN.CEGCN(height, width, bands, class_count, Q1, A1, Q2, A2)
        print("parameters", net.parameters(), len(list(net.parameters())))
        net.to(device)

        # training
        optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
        best_loss = 99999
        net.train()
        tic1 = time.clock()
        gt = torch.from_numpy(gt_reshape).to(device) - torch.ones_like(torch.from_numpy(gt_reshape)).to(device)
        gt_new = torch.cat([gt[ind].unsqueeze(0) for ind in train_data_index], dim=0)
        for i in range(max_epoch + 1):
            optimizer.zero_grad()  # zero the gradient buffers
            output = net(net_input_1, net_input_2, i, curr_seed)
            loss = compute_loss(output, train_samples_gt_onehot, train_label_mask)
            loss.backward(retain_graph=False)
            optimizer.step()  # Does the update
            if i % 10 == 0:
                with torch.no_grad():
                    net.eval()
                    trainloss = compute_loss(output, train_samples_gt_onehot, train_label_mask)
                    trainOA = evaluate_performance(output, train_samples_gt, train_samples_gt_onehot)
                    valloss = compute_loss(output, val_samples_gt_onehot, val_label_mask)
                    valOA = evaluate_performance(output, val_samples_gt, val_samples_gt_onehot)
                    print(
                        "{}\ttrain loss={}\t train OA={} val loss={}\t val OA={}".format(str(i + 1), trainloss, trainOA,
                                                                                         valloss, valOA))

                    if valloss < best_loss:
                        best_loss = valloss
                        torch.save(net.state_dict(), "model/best_model.pt")
                        print('save model...')
                torch.cuda.empty_cache()
                net.train()
        toc1 = time.clock()
        print("\n\n====================training done. starting evaluation...========================\n")
        training_time = toc1 - tic1 + SLIC_Time
        Train_Time_ALL.append(training_time)

        # testing
        torch.cuda.empty_cache()
        with torch.no_grad():
            net.load_state_dict(torch.load("model/best_model.pt"))
            net.eval()
            tic2 = time.clock()
            output = net(net_input_1, net_input_2, i, curr_seed)
            toc2 = time.clock()
            testloss = compute_loss(output, test_samples_gt_onehot, test_label_mask)
            testOA = evaluate_performance(output, test_samples_gt, test_samples_gt_onehot, require_AA_KPP=True,
                                          printFlag=False)
            print("{}\ttest loss={}\t test OA={}".format(str(i + 1), testloss, testOA))

            classification_map = torch.argmax(output, 1) + 1
            background_idx = list(background_idx)
            classification_map[background_idx] = 0
            classification_map = classification_map.reshape([height, width]).cpu()
            Draw_Classification_Map(classification_map, "results/" + dataset_name + str(testOA))
            pred = np.array(classification_map)
            sio.savemat("results/pred.mat", {'pred': pred})

    torch.cuda.empty_cache()
    del net

    OA_ALL = np.array(OA_ALL)
    AA_ALL = np.array(AA_ALL)
    KPP_ALL = np.array(KPP_ALL)
    AVG_ALL = np.array(AVG_ALL)
    Train_Time_ALL = np.array(Train_Time_ALL)
    Test_Time_ALL = np.array(Test_Time_ALL)

    print("\ntrain_ratio={}".format(curr_train_ratio),
          "\n==============================================================================")
    print('OA=', np.mean(OA_ALL), '+-', np.std(OA_ALL))
    print('AA=', np.mean(AA_ALL), '+-', np.std(AA_ALL))
    print('Kpp=', np.mean(KPP_ALL), '+-', np.std(KPP_ALL))
    print('AVG=', np.mean(AVG_ALL, 0), '+-', np.std(AVG_ALL, 0))
    print("Average training time:{}".format(np.mean(Train_Time_ALL)))
    print("Average testing time:{}".format(np.mean(Test_Time_ALL)))

    f = open('results/' + dataset_name + '_results.txt', 'a+')
    str_results = '\n\n************************************************' \
                  + "\ntrain_ratio={}".format(curr_train_ratio) \
                  + '\nOA=' + str(np.mean(OA_ALL)) + '+-' + str(np.std(OA_ALL)) \
                  + '\nAA=' + str(np.mean(AA_ALL)) + '+-' + str(np.std(AA_ALL)) \
                  + '\nKpp=' + str(np.mean(KPP_ALL)) + '+-' + str(np.std(KPP_ALL)) \
                  + '\nAVG=' + str(np.mean(AVG_ALL, 0)) + '+-' + str(np.std(AVG_ALL, 0)) \
                  + "\nAverage training time:{}".format(np.mean(Train_Time_ALL)) \
                  + "\nAverage testing time:{}".format(np.mean(Test_Time_ALL))
    f.write(str_results)
    f.close()