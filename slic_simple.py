import numpy as np
import matplotlib.pyplot as plt
from skimage.segmentation import slic, mark_boundaries
from sklearn import preprocessing
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


def SegmentsLabelProcess(labels):
    labels = np.array(labels, np.int64)
    H, W = labels.shape
    ls = list(set(np.reshape(labels, [-1]).tolist()))

    dic = {}
    for i in range(len(ls)):
        dic[ls[i]] = i

    new_labels = labels
    for i in range(H):
        for j in range(W):
            new_labels[i, j] = dic[new_labels[i, j]]
    return new_labels


class SLIC(object):
    def __init__(self, fuse, img1, img2, n_segments=1000, compactness=20, max_iter=20, sigma=0, min_size_factor=0.3,
                 max_size_factor=2):
        self.n_segments = n_segments
        self.compactness = compactness
        self.max_iter = max_iter
        self.min_size_factor = min_size_factor
        self.max_size_factor = max_size_factor
        self.sigma = sigma
        self.img1 = img1
        self.img2 = img2
        # data standardization
        # height, width, bands = fuse.shape
        # data = np.reshape(fuse, [height * width, bands])
        # minMax = preprocessing.StandardScaler()
        # data = minMax.fit_transform(data)
        # self.fuse = np.reshape(data, [height, width, bands])
        self.fuse = fuse

    def get_Q_and_S_and_Segments(self):
        img = self.fuse
        (h, w, d) = img.shape
        (h1, w1, d1) = self.img1.shape
        (h2, w2, d2) = self.img2.shape

        segments = slic(img, n_segments=self.n_segments, compactness=self.compactness,
                        convert2lab=False, sigma=self.sigma, enforce_connectivity=True,
                        min_size_factor=self.min_size_factor, max_size_factor=self.max_size_factor, slic_zero=False)

        if segments.max() + 1 != len(list(set(np.reshape(segments, [-1]).tolist()))): segments = SegmentsLabelProcess(
            segments)
        self.segments = segments
        superpixel_count = segments.max() + 1
        self.superpixel_count = superpixel_count
        print("superpixel_count", superpixel_count)

        """
        show image
        """
        out = mark_boundaries(img[:, :, [50, 70, 90]], segments)
        plt.figure()
        plt.imshow(out)
        plt.show()

        segments = np.reshape(segments, [-1])
        S1 = np.zeros([superpixel_count, d1], dtype=np.float32)
        Q1 = np.zeros([w1 * h1, superpixel_count], dtype=np.float32)
        x1 = np.reshape(self.img1, [-1, d1])
        for i in range(superpixel_count):
            idx = np.where(segments == i)[0]
            count = len(idx)
            pixels = x1[idx]
            superpixel = np.sum(pixels, 0) / count
            S1[i] = superpixel
            Q1[idx, i] = 1

        self.S1 = S1
        self.Q1 = Q1

        S2 = np.zeros([superpixel_count, d2], dtype=np.float32)
        Q2 = np.zeros([w1 * h2, superpixel_count], dtype=np.float32)
        x2 = np.reshape(self.img2, [-1, d2])
        for i in range(superpixel_count):
            idx = np.where(segments == i)[0]
            count = len(idx)
            pixels = x2[idx]
            superpixel = np.sum(pixels, 0) / count
            S2[i] = superpixel
            Q2[idx, i] = 1

        self.S2 = S2
        self.Q2 = Q2

        return Q1, S1, Q2, S2, self.segments

    def get_A(self, sigma: float):
        """
         get adjacency matrix
        :return:
        """
        A1 = np.zeros([self.superpixel_count, self.superpixel_count], dtype=np.float32)
        A2 = np.zeros([self.superpixel_count, self.superpixel_count], dtype=np.float32)
        (h, w) = self.segments.shape
        for i in range(h - 2):
            for j in range(w - 2):
                sub = self.segments[i:i + 2, j:j + 2]
                sub_max = np.max(sub).astype(np.int32)
                sub_min = np.min(sub).astype(np.int32)
                if sub_max != sub_min:
                    idx1 = sub_max
                    idx2 = sub_min
                    if A1[idx1, idx2] != 0:
                        continue

                    pix1 = self.S1[idx1]
                    pix2 = self.S1[idx2]
                    diss = np.exp(-np.sum(np.square(pix1 - pix2)) / sigma ** 2)
                    A1[idx1, idx2] = A1[idx2, idx1] = diss

        for i in range(h - 2):
            for j in range(w - 2):
                sub = self.segments[i:i + 2, j:j + 2]
                sub_max = np.max(sub).astype(np.int32)
                sub_min = np.min(sub).astype(np.int32)
                if sub_max != sub_min:
                    idx1 = sub_max
                    idx2 = sub_min
                    if A2[idx1, idx2] != 0:
                        continue

                    pix1 = self.S2[idx1]
                    pix2 = self.S2[idx2]
                    diss = np.exp(-np.sum(np.square(pix1 - pix2)) / sigma ** 2)
                    A2[idx1, idx2] = A2[idx2, idx1] = diss

        return A1, A2


class SP_SLIC(object):
    def __init__(self, data1, data2, labels, n_component):

        self.data1 = data1
        self.data2 = data2
        self.init_labels = labels
        self.n_component = n_component
        self.height, self.width, self.bands = data1.shape
        self.fuse = np.concatenate((data1, data2), axis=2)
        self.x_flatt = np.reshape(data1, [self.width * self.height, self.bands])
        self.y_flatt = np.reshape(labels, [self.height * self.width])
        self.labels = labels

    def LDA_Process(self, curr_labels):
        """
        :param curr_labels: height * width
        :return:
        """
        curr_labels = np.reshape(curr_labels, [-1])
        idx = np.where(curr_labels != 0)[0]
        x = self.x_flatt[idx]
        y = curr_labels[idx]
        print("------------")
        print(y.shape)
        lda = LinearDiscriminantAnalysis()  # n_components=self.n_component
        lda.fit(x, y)
        X_new = lda.transform(self.x_flatt)
        return np.reshape(X_new, [self.height, self.width, -1])

    def SLIC_Process(self, fuse, img1, img2, scale=300):

        n_segments_init = self.height * self.width / scale
        # print("n_segments_init", n_segments_init)
        myslic = SLIC(fuse, img1, img2, n_segments=n_segments_init, compactness=20, sigma=1, min_size_factor=0.1,
                      max_size_factor=2)
        Q1, S1, Q2, S2, Segments = myslic.get_Q_and_S_and_Segments()
        A1, A2 = myslic.get_A(sigma=10)
        return Q1, S1, A1, Q2, S2, A2, Segments

    def simple_superpixel(self, scale):
        curr_labels = self.init_labels
        X = self.LDA_Process(curr_labels)
        Q1, S1, A1, Q2, S2, A2, Seg = self.SLIC_Process(self.fuse, X, X, scale=scale)
        return Q1, S1, A1, Q2, S2, A2, Seg

    def simple_superpixel_no_LDA(self, scale):

        Q1, S1, A1, Q2, S2, A2, Seg = self.SLIC_Process(self.fuse, self.data1, self.data2, scale=scale)

        return Q1, S1, A1, Q2, S2, A2, Seg