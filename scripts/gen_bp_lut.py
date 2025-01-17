import numpy as np
import matplotlib.pyplot as plt
import tqdm
import optparse


def obtain_smooth_b(lut_appmult, half_ws):
    LUT_ROW_NUM, LUT_COL_NUM = lut_appmult.shape
    smooth_b = np.zeros((LUT_ROW_NUM, LUT_COL_NUM), dtype=np.float32)
    # moving average
    for a in tqdm.tqdm(range(LUT_ROW_NUM)):
        for b in range(LUT_COL_NUM):
            b_start = max(0, b - half_ws)
            b_end = min(LUT_COL_NUM, b + half_ws + 1)
            smooth_b[a][b] = np.mean(lut_appmult[a][b_start:b_end])
    return smooth_b

    
def obtain_grad_b_v2(smooth_b, lut_appmult, half_ws):
    LUT_ROW_NUM, LUT_COL_NUM = smooth_b.shape
    grad_b = np.zeros((LUT_ROW_NUM, LUT_COL_NUM), dtype=np.float32)
    assert half_ws >= 1
    for a in range(LUT_ROW_NUM):
        for b in range(LUT_COL_NUM):
            if b >= half_ws + 1 and b <= LUT_COL_NUM - 1 - half_ws - 1:
                grad_b[a][b] = (smooth_b[a][b+1] - smooth_b[a][b-1]) / 2
            else:
                # get max & min value of lut_appmult[a][0:LUT_COL_NUM]
                max_val = np.max(lut_appmult[a][0:LUT_COL_NUM])
                min_val = np.min(lut_appmult[a][0:LUT_COL_NUM])
                assert max_val >= min_val
                grad_b[a][b] = (max_val - min_val) / LUT_COL_NUM
    return grad_b
    

def obtain_smooth_a(lut_appmult, half_ws):
    LUT_ROW_NUM, LUT_COL_NUM = lut_appmult.shape
    smooth_a = np.zeros((LUT_ROW_NUM, LUT_COL_NUM), dtype=np.float32)
    # moving average
    for b in tqdm.tqdm(range(LUT_COL_NUM)):
        for a in range(LUT_ROW_NUM):
            a_start = max(0, a - half_ws)
            a_end = min(LUT_ROW_NUM, a + half_ws + 1)
            smooth_a[a][b] = np.mean(lut_appmult[a_start:a_end, b])
    return smooth_a


def obtain_grad_a_v2(smooth_a, lut_appmult, half_ws):
    LUT_ROW_NUM, LUT_COL_NUM = smooth_a.shape
    grad_a = np.zeros((LUT_ROW_NUM, LUT_COL_NUM), dtype=np.float32)
    assert half_ws >= 1
    for b in range(LUT_COL_NUM):
        for a in range(LUT_ROW_NUM):
            if a >= half_ws + 1 and a <= LUT_ROW_NUM - 1 - half_ws - 1:
                grad_a[a][b] = (smooth_a[a+1][b] - smooth_a[a-1][b]) / 2
            else:
                # get max & min value of lut_appmult[0:LUT_ROW_NUM][b]
                max_val = np.max(lut_appmult[0:LUT_ROW_NUM][b])
                min_val = np.min(lut_appmult[0:LUT_ROW_NUM][b])
                assert max_val >= min_val
                grad_a[a][b] = (max_val - min_val) / LUT_ROW_NUM
    return grad_a

    
def main():
    # parse command line arguments
    parser = optparse.OptionParser()
    parser.add_option('-f', '--file', dest='file_name', help='file name of the lookup table', default=None)
    parser.add_option('-w', '--half_ws_a', dest='half_ws_a', help='half window size for smoothing grad_a', default=32)
    # parser.add_option('-x', '--half_ws_b', dest='half_ws_b', help='half window size for smoothing grad_b', default=32)
    (options, args) = parser.parse_args()

    # parameters
    file_name = options.file_name
    half_ws_a = int(options.half_ws_a)
    # half_ws_b = int(options.half_ws_b)
    half_ws_b = half_ws_a
    if file_name.find('mul8') != -1 or file_name.find('8_8') != -1:
        QUANTIZATION_BIT = 8
    elif file_name.find('mul7') != -1 or file_name.find('7_7') != -1:
        QUANTIZATION_BIT = 7
    elif file_name.find('mul4') != -1 or file_name.find('4_4') != -1:
        QUANTIZATION_BIT = 4
    elif file_name.find('mul6') != -1 or file_name.find('6_6') != -1:
        QUANTIZATION_BIT = 6
    else:
        raise ValueError('Unknown bit width')
    LUT_MAXVAL = 2 ** QUANTIZATION_BIT - 1
    LUT_ROW_NUM = LUT_MAXVAL + 1
    LUT_COL_NUM = LUT_MAXVAL + 1

    # load lookup table from file
    state = 'init'
    lut_appmult = np.zeros((LUT_ROW_NUM, LUT_COL_NUM), dtype=np.float32)
    lut_accmult = np.zeros((LUT_ROW_NUM, LUT_COL_NUM), dtype=np.float32)
    grad_a_ste = np.zeros((LUT_ROW_NUM, LUT_COL_NUM), dtype=np.float32)
    grad_b_ste = np.zeros((LUT_ROW_NUM, LUT_COL_NUM), dtype=np.float32)
    with open(file_name, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if line.startswith('LUT for approximate multiplier:'):
                state = 'load_appmult'
                continue
            if state != 'load_appmult':
                continue
            a, b, res = line.split()
            a, b, res = int(a), int(b), int(res)
            lut_appmult[a][b] = res
            lut_accmult[a][b] = a * b
            grad_a_ste[a][b] = b
            grad_b_ste[a][b] = a

    smooth_a = obtain_smooth_a(lut_appmult, half_ws_a)
    smooth_b = obtain_smooth_b(lut_appmult, half_ws_b)
    grad_a = obtain_grad_a_v2(smooth_a, lut_appmult, half_ws_a)
    grad_b = obtain_grad_b_v2(smooth_b, lut_appmult, half_ws_b)

    # compare error
    print(f'INFO: file_name={file_name}, half_ws_a={half_ws_a}, half_ws_b={half_ws_b}, QUANTIZATION_BIT={QUANTIZATION_BIT}')
    print(f'INFO: half smoothing window size for grad_a: {half_ws_a}')
    print(f'INFO: half smoothing window size for grad_b: {half_ws_b}')
    print(f'INFO: Mean squared error between lut_appmult and lut_accmult: {np.mean((lut_appmult - lut_accmult) ** 2)}')
    print(f'INFO: Error rate between lut_appmult and lut_mult: {np.sum(lut_appmult != lut_accmult) / (LUT_ROW_NUM * LUT_COL_NUM) * 100}%')
    print(f'INFO: Mean error distance between lut_appmult and lut_mult: {np.mean(np.abs(lut_appmult - lut_accmult))}')
    print(f'INFO: Normalized mean error distance between lut_appmult and lut_mult: {np.mean(np.abs(lut_appmult - lut_accmult)) / (2**(2*QUANTIZATION_BIT)) * 100}%')
    print(f'INFO: Maximum error distance between lut_appmult and lut_mult: {np.max(np.abs(lut_appmult - lut_accmult))}')
    print(f'INFO: Normalized maximum error distance between lut_appmult and lut_mult: {np.max(np.abs(lut_appmult - lut_accmult)) / (2**(2*QUANTIZATION_BIT)) * 100}%')
    print(f'INFO: Mean of lut_appmult: {np.mean(lut_appmult)}')
    print(f'INFO: Mean of lut_mult: {np.mean(lut_accmult)}')
    print(f'INFO: Mean of grad_a: {np.mean(grad_a)}')
    print(f'INFO: Mean of grad_b: {np.mean(grad_b)}')
    print(f'INFO: Mean of grad_a_ste: {np.mean(grad_a_ste)}')
    print(f'INFO: Mean of grad_b_ste: {np.mean(grad_b_ste)}')
    print(f'INFO: Mean squared error between grad_a_ste and grad_a_smooth: {np.mean((grad_a_ste - grad_a) ** 2)}')
    print(f'INFO: Mean squared error between grad_b_ste and grad_b_smooth: {np.mean((grad_b_ste - grad_b) ** 2)}')
    print(f'INFO: Mean relative error between grad_a_ste and grad_a_smooth: {np.mean(np.abs(grad_a_ste - grad_a) / (grad_a_ste + 1e-1))}')
    print(f'INFO: Mean relative error between grad_b_ste and grad_b_smooth: {np.mean(np.abs(grad_b_ste - grad_b) / (grad_b_ste + 1e-1))}')

    # # plot the curve of lut_appmult
    # x = np.arange(0, LUT_ROW_NUM * LUT_COL_NUM)
    # plt.figure()
    # plt.plot(x, lut_appmult.flatten(), label='lut_appmult', linestyle='dotted')
    # # plt.plot(x, lut_accmult.flatten(), label='lut_accmult', linestyle='dotted')
    # plt.plot(x, smooth_b.flatten(), label='smooth_b')
    # plt.legend()
    
    # plt.figure()
    # plt.plot(x, lut_appmult.T.flatten(), label='lut_appmult', linestyle='dotted')
    # # plt.plot(x, lut_accmult.T.flatten(), label='lut_accmult', linestyle='dotted')
    # plt.plot(x, smooth_a.T.flatten(), label='smooth_a')
    # plt.legend()

    # plt.figure()
    # plt.plot(x, grad_b_ste.flatten(), label='grad_b_ste', linestyle='dotted')
    # plt.plot(x, grad_b.flatten(), label='grad_b')
    # plt.legend()
    
    # plt.figure()
    # plt.plot(x, grad_a_ste.T.flatten(), label='grad_a_ste', linestyle='dotted')
    # plt.plot(x, grad_a.T.flatten(), label='grad_a')
    # plt.legend()

    # plt.show()

    print('LUT for approximate multiplier:')
    for A in range(LUT_ROW_NUM):
        for B in range(LUT_COL_NUM):
            val = int(lut_appmult[A][B])
            assert val >= 0 and val < 2**16
            print(f'{A} {B} {val}')
    print('LUT for the gradient of the approximate multiplier w.r.t. the first operand:')
    FACTOR = 16.0
    for A in range(LUT_ROW_NUM):
        for B in range(LUT_COL_NUM):
            val = int(np.round(grad_a[A][B] * FACTOR))
            assert val >= -2**15 and val < 2**15, f'val={val}, A={A}, B={B}'
            print(f'{A} {B} {val}')
    print('LUT for the gradient of the approximate multiplier w.r.t. the second operand:')
    for A in range(LUT_ROW_NUM):
        for B in range(LUT_COL_NUM):
            val = int(np.round(grad_b[A][B] * FACTOR))
            assert val >= -2**15 and val < 2**15, f'val={val}, A={A}, B={B}'
            print(f'{A} {B} {val}')


if __name__ == '__main__':
    main()