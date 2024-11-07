import numpy as np

##### Iterable parameters #####
def afterlen(cutoff_point=125):
    num_blocks_range = [2, 3, 4, 5, 6]
    kernel_size_range = [2, 3, 4, 5, 6, 7, 8, 9, 10]

    option_list = np.zeros([3, len(num_blocks_range) * len(kernel_size_range)])

    o = 0
    for n in range(len(num_blocks_range)):
        num_blocks = num_blocks_range[n]
        for k in range(len(kernel_size_range)):
            kernel_size = kernel_size_range[k]

            rec_field = 1
            for i in range(0, num_blocks):
                rec_field += 2 * (kernel_size - 1) * (2 ** i)

            option_list[0, o] = num_blocks
            option_list[1, o] = kernel_size
            option_list[2, o] = rec_field

            o += 1

    backward_candid_idx1 = np.where(option_list[2,:] > cutoff_point)[0]
    backward_candid_1 = option_list[:, backward_candid_idx1]

    backward_candid_idx2 = np.where(backward_candid_1[2,:] < 2*cutoff_point - 1)[0]
    backward_candid_2 = backward_candid_1[:, backward_candid_idx2]

    return backward_candid_2

num_blocks = 3
kernel_size = 3
rec_field = 1
for i in range(0, num_blocks):
    rec_field += 2 * (kernel_size - 1) * (2 ** i)
print(rec_field)