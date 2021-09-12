

from css import CurvatureScaleSpace, SlicedCurvatureScaleSpace
import numpy as np
import matplotlib.pyplot as plt
import cv2

import scipy.signal as sg
# from sklearn.mixture import GMM, DPGMM


def simple_signal(np_points):
    curve = np.zeros(shape=(2, np_points))
    t = np.linspace(-4, 4, np_points)

    curve[0, :] = 5 * np.cos(t) - np.cos(6 * t)
    curve[1, :] = 15 * np.sin(t) - np.sin(6 * t)
    return curve

def simple_signal_from_im(im_path):
    im = cv2.imread(im_path)
    imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(imgray, 127, 255, 0)
    out = cv2.Laplacian(thresh, cv2.CV_8U)
    result = np.where(out == 255)


    x = result[1]
    y = result[0]

    curve = np.zeros(shape=(2, x.__len__()))
    curve[0, :] = x
    curve[1, :] = y
    return curve


def some_function(some_args):
    """ 
    :Parameters:
    """
    pass


def my_run():
    ims = [r'/home/shmuelgr/Downloads/i0.png', r'/home/shmuelgr/Downloads/i1.jpg', r'/home/shmuelgr/Downloads/i2.jpg']

    curve_arr = []
    vcs_arr = []
    for im in ims:
        curve = simple_signal_from_im(im)
        c = CurvatureScaleSpace()
        # cs = c.generate_css(curve, curve.shape[1], 0.01)
        cs = c.generate_css(curve, curve.shape[1], 1)

        vcs, maxs = c.generate_visual_css(cs, 9, True)
        # ecs = c.generate_eigen_css(cs)
        # print ecs.shape

        curve = curve.astype(int)

        curve_arr.append(curve)
        vcs_arr.append(vcs)

    curve = curve_arr[0]
    plt.figure('Sample Curve0')
    plt.plot(curve[0,:], curve[1,:], marker='.',color='r', ls='')
    plt.plot(curve[0, 0:10], curve[1, 0:10], marker='.', color='b', ls='')

    curve = curve_arr[1]
    plt.figure('Sample Curve1')
    plt.plot(curve[0,:], curve[1,:], marker='.',color='r', ls='')
    plt.plot(curve[0, 0:10], curve[1, 0:10], marker='.', color='b', ls='')

    curve = curve_arr[2]
    plt.figure('Sample Curve2')
    plt.plot(curve[0,:], curve[1,:], marker='.',color='r', ls='')
    plt.plot(curve[0, 0:10], curve[1, 0:10], marker='.', color='b', ls='')

    vcs = vcs_arr[0]
    plt.figure('CSS0')
    plt.plot(vcs)

    vcs = vcs_arr[1]
    plt.figure('CSS1')
    plt.plot(vcs)

    vcs = vcs_arr[2]
    plt.figure('CSS2')
    plt.plot(vcs)

        # plt.figure('maxs')
        # x = [i[0] for i in maxs]
        # y = [i[1] for i in maxs]
        # plt.plot(x, y, marker='.', color='b', ls='')


        # plt.figure('EigenCSS')
        # plt.plot(ecs)

    plt.show()


def run():
    curve = simple_signal(np_points=400)
    c = CurvatureScaleSpace()
    cs = c.generate_css(curve, curve.shape[1], 0.01)
    # vcs = c.generate_visual_css(cs, 9)
    # ecs = c.generate_eigen_css(cs)
    # print ecs.shape

    vcs, maxs = c.generate_visual_css(cs, 9, True)

    plt.figure('Sample Curve')
    plt.plot(curve[0,:], curve[1,:], marker='.',color='r', ls='')

    for max in maxs:
        plt.plot(curve[0, max[0]], curve[1, max[0]], marker='.', color='b', ls='')


    plt.figure('CSS')
    plt.plot(vcs)

    # plt.figure('EigenCSS')
    # plt.plot(ecs)

    plt.show()

if __name__ == '__main__':
    run()
