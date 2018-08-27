
import sys

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('GTK')
plt.style.use('ggplot')

import numpy as np 

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--filename', type=str, required=True,
                    help='The path to the output file')
parser.add_argument('--temperature', type=float, default=1.0,
                    help='The temperature to perform T scaling')
parser.add_argument('--N', type=int, default=20,
                    help='Number of bins')
parser.add_argument('--M', type=float, default=1.0,
                    help='Custom')
args = parser.parse_args()

M = args.M
N = args.N
T = args.temperature

EPSILON=1e-7

cnt = 0.0
num_calls = 0.0
bin_array = np.zeros((N+1,))
means_array = np.zeros((N+1,))
num_array = np.zeros((N+1,)) 
val_out = 0
val_incorrect = 0
token_dict = {}
thresholds = np.logspace(-6, -2, 10).tolist() 
# thresholds = [0.001, 0.005, 0.007, 0.01, 0.013, 0.017, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05, 0.055, 0.060, 0.075, 0.08, 0.09, 0.1, 0.15, 0.2] 
mean_tail = [0.0]*len(thresholds)
num_tail = [0.0]*len(thresholds)
bin_tail = [0.0]*len(thresholds)
# token_list_dict={2:0, 4:1, 3:2, 46:3, 12:4, 10:5, 9:6, 6:7, 92:8, 51:9, 20:10, 72:11, 18:12, 68:13, 16:14, 88:15}
token_list_dict={2:0, 4:1, 5:2, 3:3, 7:4, 11:5, 8:6, 46:7, 13:8, 6:9, 35:10, 14:11, 15:12, 17:13, 64:14, 54:15}
# token_list_dict={2: 0, 70: 1, 16:2, 166:3, 106:4, 475:5, 165:6, 99:7, 216:8, 7:9, 95:10, 195:11, 591:12, 46:13, 596:14, 325:15}
# token_list_dict= {2: 0, 4: 1, 5: 2, 3: 3, 7: 4, 11:5, 8: 6, 46: 7, 13: 8, 6:9, 35:10, 15: 11, 14:12, 17:13, 54:14, 64:15}
# token_list_dict = {7350:0, 23:1, 291:2, 767:3, 347:4, 49:5, 229:6, 21:7}
bin_array1 = []
means_array1 = []
num_array1 = []
for i in range(len(token_list_dict)):
    bin_array1.append(np.zeros((N+1, )))
    means_array1.append(np.zeros((N+1, )))
    num_array1.append(np.zeros((N+1, )))

def get_bin_num(val):
    global N
    val = min(int(val*N), N)
    return val

def softmax(arr):
    arr1 = arr - np.max(arr)
    return np.exp(arr1)/np.sum(np.exp(arr1))

def process_one_record_original(tgt, dist, pred):
    global bin_array, means_array, num_array, num_calls, val_out, val_incorrect, token_dict, token_list_dict
    global bin_array1, means_array1, num_array1
    # get_pos = np.where(indices == tgt)
    # print ('POS: ', get_pos)
    num_calls += 1
    # if int(tgt) == 3.0 or int(tgt) == 4.0:
       # return
    # dist = np.max(dist)
    dist = dist.tolist()
    for i in range(len(dist)):
        # if i <= 1:
           # continuei
        # if indices[i] == 2 or indices[i] == 70 or indices[i] == 166 or indices[i] == 16:
           # continue
        bin_num = get_bin_num(dist[i])
        num_array[bin_num] += (dist[i]*M)
        if i == int(tgt):
             bin_array[bin_num] += (dist[i]*M)
        means_array[bin_num] += (dist[i]*dist[i]*M)
        # if bin_num >= 16:
            # with open('high_bins.txt', 'a') as hh:
                # hh.write(str(target) + ' , ' + str(indices[i]) + ' , ' + str(val[i]) + '\n')
        # if bin_num >= 15:
           # if int(indices[i]) not in token_dict:
               # token_dict[int(indices[i])] = 1
           # else:
               # token_dict[int(indices[i])] += 1
        # if bin_num == 20 and not(int(indices[i]) == int(tgt)):
           # val_incorrect += 1
        # with open('high_end_20.txt', 'a') as f:
               # f.write(str(val[i]) + ' ' + str(indices[i]) + ' '+ str(tgt) + ' '+ str(val[i]*M) + ' '+ str(val[i]*val[i]*M) + ' ' + str(max(val))+'\n')

def get_targets_and_probs_from_file(filename):
    targets = []
    probs = []

    with open(filename, 'r') as f:
        line = f.readline()
        while line:
            if 'Targets' in line:
                targets.append(line)
            if 'Probs' in line:
                probs.append(line)
            line = f.readline()

def main():

    targets, probs = get_targets_and_probs_from_file(args.filename)
    assert len(targets) == len(probs), "Corrupt output file"

    def transform_target_line(tline):
        tline = tline[tline.index('Targets[')+8:]
        tline = tline.replace('][', ' ')
        tline = tline.replace(']', '')
        tline = tline.replace('[', '')
        tline = tline.replace('\n', '')
        tline = tline.split(' ')

        tline = [int(x) for x in tline]
        tline = np.array(tline)

        return tline

    def transform_probs_line(pline):
        pline = pline[pline.index('Probs[')+6:]
        pline = pline.replace('][', ' ')
        pline = pline.replace(']', '')
        pline = pline.replace('[', '')
        pline = pline.replace('\n', '')
        pline = pline.split(' ')
        pline = [float(x) for x in pline]
        
        pline = np.array(pline)
        pline = np.reshape(pline, [-1, 10])

        return pline

    targets = [ transform_target_line(x) for x in targets ]
    probs = [ transform_probs_line(x) for x in probs ]

    brier_score = 0.0
    total_num = 0.0
    nll = 0.0
    
    # iterate over batches
    for target_b, prob_b in zip(targets, probs):
        assert target_b.shape[0] == prob_b.shape[0], "targets != nprob distrs"

        # iterate over samples in batch
        for i in range(target_b.shape[0]):
            # get termperature scaled distribution
            distribution = softmax( np.log(prob_b[i] + EPSILON) * args.temperature )
            prediction = tf.argmax(distribution, axis=0)
            target = target_b[i]

            target_one_hot = np.zeros(10); target_one_hot[target] = 1.0

            brier_score += np.sum( (target_one_hot-distrbution)**2 )
            total_num += 1.0



        #   print (xent_index.shape[0])
        #   for i in range(xent_index.shape[0]):
        #     target = xent_index[i]
        #     dist = softmax(np.log(probs[i]+1e-7)*T)
        #     pred = predictions[i]

        #     ttz = np.zeros(10)
        #     ttz[target] = 1.0
        #     brier_score += np.sum((ttz - dist)**2)
        #     total_num += 1.0

        #     nll -= np.log(dist)[target]
        #     # process_one_record_original(target, dist, pred)

        #     bin_num = get_bin_num(dist[pred])
        #     means_array[bin_num] += dist[pred]
        #     num_array[bin_num] += 1.0
        #     if pred == target:
        #         bin_array[bin_num] += 1.0


        #                         # dist = np.array([np.max(dist)])
        #                 # print ('SUM: ',  np.sum(top500_val[i, j, :]))
        #                 # r = max((1-np.sum(top500_val[i, j, :])), 0)
        #                  # p = top500_val[i][j][-1]
        #                  # bin_num = get_bin_num(p)
        #                  # num_array[bin_num] += (r*M)
        #                  # if target not in top500_index[i, j, :]:
        #                      # bin_array[bin_num] += (xent_val[i][j]*M)
        #                      # means_array[bin_num] += (p*r*M)

        #             # process_one_record_original(target, dist, pred)

        #                 # if target == 2:
        #                     # break
        # line = f.readline()

    print ('Brier score: ', brier_score/total_num)
    print ('NLL: ', nll/total_num)
    print (' Means: ', means_array.tolist())
    print ('Num Array: ', num_array.tolist())
    print ('Bin Array: ', bin_array.tolist())
    print ('Cnt: ', cnt)
    print ('Num calls: ', num_calls)
    print ('----------------------------------')
    means_array[N-1] += means_array[N]
    bin_array[N-1] += bin_array[N]
    num_array[N-1] += num_array[N]
    means_array = means_array/(num_array + 1e-5)
    bin_array = bin_array/(num_array + 1e-5)
    print (' Means: ', means_array.tolist()[:-1])
    print ('Num Array: ', num_array.tolist()[:-1]/np.sum(num_array[:-1]))
    print ('Bin Array: ', bin_array.tolist()[:-1])
    print ('Cnt: ', cnt)
    print ('Num calls: ', num_calls)

    ece = np.abs(bin_array - means_array)
    ece = ece*num_array
    print ('ECE Array: ', ece[-3:-1]/np.sum(num_array[-3:-1]))
    ece = np.sum(ece[:-1])/np.sum(num_array[:-1])
    print ('Expected Calibration Error: ', ece)

    sorted_x = sorted(token_dict.items(), key=lambda x: x[1])
    print (sorted_x)
    # sys.exit(0)
    '''
    plt.figure()
    f, axarr = plt.subplots(4, 4)

    # list_tok= [2, 4, 3, 46, 12, 10, 9, 6, 92, 51, 20, 72, 18, 68, 16, 88]
    # list_tok=[2, 4, 5, 3, 7, 11, 8, 46, 13, 6, 35, 14, 15, 17, 64, 54]  
    list_tok = [2, 70, 16, 166, 106, 475, 165, 99, 216, 7, 95, 195, 591, 46, 596, 325]
    # list_tok = [2, 4, 5, 3, 7, 11, 8, 46, 13, 6, 35, 15, 14, 17, 54, 64]
    # list_tok = [7350, 23, 291, 767, 347, 49, 229,21]

    for i in range(len(token_list_dict)):
        print (token_list_dict)
        means_array1[i][N-1] += means_array1[i][N]
        bin_array1[i][N-1] += bin_array1[i][N]
        num_array1[i][N-1] += num_array1[i][N] 
        means_array1[i] = means_array1[i]/num_array1[i]
        bin_array1[i] = bin_array1[i]/num_array1[i]
        print (means_array1[i], bin_array1[i])
        print ('---------------------------------------')
        axarr[i/4, i%4].plot(means_array1[i][:-1], bin_array1[i][:-1])
        axarr[i/4, i%4].plot(0.1*np.arange(11), 0.1*np.arange(11))
        axarr[i/4, i%4].plot(means_array1[i][:-1], num_array1[i][:-1]/np.sum(num_array1[i][:-1]))
        axarr[i/4, i%4].set_title('Token : '+str(list_tok[i]))

    plt.show()
    '''

    plt.figure()
    plt.hist(bin_array[:-1], normed=False, bins=N-1)
    # here
    plt.savefig("histogram.png")

    plt.figure()
    plt.plot(means_array[:-1], bin_array[:-1], linewidth=3.0)
    # plt.scatter(means_array[:-1], bin_array[:-1])
    plt.plot(0.1*np.arange(11), 0.1*np.arange(11), linewidth=3.0)
    # plt.fill_between(means_array[:-1], 0, bin_array[:-1])
    plt.fill_between(means_array[:-1], bin_array[:-1], means_array[:-1], color='grey', alpha=0.9)
    plt.plot(means_array[:-1], num_array[:-1]/np.sum(num_array[:-1]))
    plt.scatter(means_array[:-1], num_array[:-1]/np.sum(num_array[:-1]))
    # plt.xlim(0.78, 1.05)
    # plt.ylim(0.4, 1.05)
    # here
    plt.savefig("other.png")


    # print ('BIN 19: ', val_out)
    # print ('BIN 19, incorrect: ', val_incorrect)

    # print (sorted_x)
    '''
    mean_tail = np.array(mean_tail)
    num_tail = np.array(num_tail)
    bin_tail = np.array(bin_tail)
    y_val = bin_tail/num_tail
    x_val = mean_tail/num_tail

    plt.plot(thresholds, y_val, label='true in tail')
    plt.plot(thresholds, x_val, label='total tail prob')
    plt.legend()
    plt.show()
    '''


if __name__ == '__main__':
    main()



