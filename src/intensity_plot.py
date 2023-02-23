from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
from scipy.signal import find_peaks, savgol_filter

# your own src_path of oringinal images
src_path = './imgs'
file_list = os.listdir(src_path)
file_list.sort()

# your own save path of intensity plots
save_root = './plots'
for file in file_list:
    
    img_list = os.listdir(os.path.join(src_path, file))
    img_list.sort()

    time_list = []
    intensity_list = []
    for img_name in img_list:
        time_list.append(time.mktime(time.strptime(img_name[2:15], '%y%m%d_%H%M%S')))
        img = Image.open(os.path.join(src_path,file,img_name)).convert('L')
        img_data = np.array(img).astype(np.float32)
        photon_data = pixel2photon(img_data)
        intensity_list.append(np.sum(photon_data))

    time_array = np.array(time_list)
    intensity_array = np.array(intensity_list)
    x = [i for i in range(int(time_array[0]), int(time_array[-1]+1))]
    y = np.interp(x, time_array, intensity_array)
    y = savgol_filter(y, 599,3,mode='nearest')

    peaks, properties = find_peaks(y, height=y[0])

    if len(peaks)==0:
        peak = y.tolist().index(max(y))

    elif len(peaks)>1:
        peak = -100
        max_index = y.tolist().index(max(y))
        new_peaks = []
        for p in peaks:
            if p <=max_index:
                new_peaks.append(p)
            else:
                break

        for m in range(len(new_peaks)-1):
            bottom = y.tolist().index(min(y[new_peaks[m]:new_peaks[m+1]]))
            if bottom>=1800 and bottom>=2*new_peaks[m]:
                peak = new_peaks[m]
                break

        if peak==-100:
            peak = new_peaks[-1]

    else:
        peak = peaks[0]
    try:
        peak_index = time_list.index(x[peak])
    except:
        for m in range(len(time_list)):
            if time_list[m]>x[peak]:
                break
        if time_list[m]-x[peak] <= x[peak]-time_list[m-1]:
            peak_index = time_list[m]
        else:
            peak_index = time_list[m-1]

    x_major_locator=MultipleLocator(300)
    ax = plt.gca()
    ax.ticklabel_format(style='sci', scilimits=(-1,2), axis='y')

    plt.xlabel('time (UT)')
    plt.ylabel('intensity ('+'photons cm'+"$^{{{0:d}}}$".format(-2)+'s'+"$^{{{0:d}}}$".format(-1)+')')

    plt.scatter(x[peak],y[peak], marker='.', color = 'green', s=200, zorder=2, label='end')
    plt.plot(x,y, marker='.', color='red', zorder=1, label='smoothed')
    plt.plot(time_list, intensity_list,  marker='.', color='blue', zorder=1, label='unsmoothed')
    
    new_ticks = [time.strftime("%H:%M", time.localtime(int(t))) for t in x]
    plt.xticks(x, new_ticks)
    ax.xaxis.set_major_locator(x_major_locator)

    plt.legend()
    # plt.show()
    plt.savefig(save_root+file+'_'+str(peak_index)+'.jpg')
    plt.close()