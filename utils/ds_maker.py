import os

#root = '/l/users/20020067/Datasets/Stanford Cars/Stanford Cars/low_data'
#root = '/l/users/20020067/Datasets/CUB_200_2011/CUB_200_2011/CUB_200_2011/low_data/CUB200/image_list'
root = '/l/users/20020067/Datasets/CRC_colorectal_cancer_histology/low_data/50_50/my'

split = 1 #10 # 50 # 30 # 15 #10 # # 11 (1 image)

path_old = os.path.join(root, 'train_100.txt')
path_new = os.path.join(root, 'train_' + str(split) + '.txt')


prev_target = -1

with open(path_old, "r") as file_input:
    with open(path_new, "w") as file_output: 
        for idx, line in enumerate(file_input):
            #if idx < 10: print(idx)

            split_line = line.split(' ') #()
            target = int(split_line[-1])

            if split == 11:
                if prev_target != int(target):
                    file_output.write(line)
                prev_target = target


            elif split == 1: 
                if (idx+1) % 100 == 0: # medical CRC
                    file_output.write(line)

            elif split == 2: 
                if (idx+1) % 50 == 0: # medical CRC
                    file_output.write(line)

            elif split == 3: 
                if (idx+1) % 33 == 0: # medical CRC
                    file_output.write(line)

            elif split == 4: 
                if (idx+1) % 25 == 0: # medical CRC
                    file_output.write(line)

            elif split == 5: 
                if (idx+1) % 20 == 0: # medical CRC
                    file_output.write(line)
                    
            elif split == 10: 
                if (idx+1) % 10 == 0: # medical CRC
                    file_output.write(line)

            elif split == 15:
                if (idx+1) % 7 == 0: # medical CRC
                    file_output.write(line)

            elif split == 30:
                if (idx+1) % 3 == 0: # medical CRC
                    file_output.write(line)

            elif split == 50:
                if (idx+1) % 2 == 0: # medical CRC
                    file_output.write(line)