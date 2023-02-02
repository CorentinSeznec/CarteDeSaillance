import cv2 as cv
import numpy as np
import os 
import shutil 



## from path to mp4 -> Create list of frames
# arg: path to video repertory - path to frame repertory - name of subject
def create_frame(path_to_video, path_to_frame, extension):
    
    cap = cv.VideoCapture(path_to_video)

    # split_ = list_videos_bowl[i].split('/')
    # print(split_)
    # list_videos_bowl[i] = split_[0]

    if (cap.isOpened()== False): 
        print("Error opening video stream or file")

    j = 0
    while(cap.isOpened()):
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret == True:
            
            cv.imwrite(path_to_frame+extension+'/frame_'+extension+'_'+str(j)+'.png',frame)
     
            j = j+1

        else: 
            break


## from path to png and path to map sailliancy-> create txt file BB
def create_bounding_boxes(idx_object, path_saliency, path_to_frame):
    
    list_saliency = os.listdir(path_saliency)
    for file in list_saliency:
   
        
        img_saliency = cv.imread(path_saliency + file)
        # get index from image
        n_image = file.split('_')
        n_image = n_image[1].strip('.png')
        n_images = list(n_image)
        index =0
        while n_images[0] == "0":
            index+=1
            del n_images[0]
            if index >3:
                break
        n_images = "".join(n_images)
        
        gray = cv.cvtColor(img_saliency, cv.COLOR_BGR2GRAY)
    
        b = np.where(gray > 20)
        if(len(b[0]) != 0 and len(b[1])  !=0):
            x, y = max(b[0]), max(b[1])
            x1, y1 = min(b[0]), min(b[1])
            to = str(n_images) + ' ' + str(idx_object) + ' ' + str(y) + ' ' + str(x) + ' ' + str(y1) + ' ' + str(x1)
        else: 
            x = 0
            to = str(n_images) + ' ' + '0'
            
        with open(path_to_frame+"/BB_file.txt", "a") as file:
            file.write(to)
            file.write('\n')
            



def draw_image(img, bounding_box): 

    print(bounding_box)
    x, y = bounding_box[0], bounding_box[1]
    x1, y1 = bounding_box[2], bounding_box[3]
    start_point = (x, y)
    end_point = (x1, y1)
    thickness = 3
    color = (0, 255, 255)
    image = cv.rectangle(img, start_point, end_point, color, thickness)
    cv.imshow("WINDOW",image)
    cv.waitKey(0)
    cv.destroyAllWindows()

def patched_img(img,x,y,x1,y1):
    return img[x:x1, y:y1]


def save_image(filename, path_to_frame, path_to_BB):

    if  os.path.exists(path_to_frame+'/patches/'):
        shutil.rmtree(path_to_frame+'/patches/')
    os.makedirs(path_to_frame+'/patches/')

    BB_file = os.path.join(path_to_BB)

    with open(BB_file, "r") as file:
        line = file.readlines()

        # get number of frame
        lst = os.listdir(path_to_frame) # your directory path
        number_files = len(lst)
        print(number_files)

        for nb in range(number_files-2):

            l = line[nb]
            file_line =  l.split(' ')
       

            if len(file_line) != 6:
                pass

            else:
                num_img = int(file_line[0])
                # y_max, x_max = int(file_line[2]), int(file_line[3])
                # y_min, x_min = int(file_line[4]), int(file_line[5])
                
                y_max, x_max = int(file_line[2]), int(file_line[3])
                y_min, x_min = int(file_line[4]), int(file_line[5])

                img_name = 'frame_' + filename + '_' + str(num_img) + '.png'
        
                img_path = os.path.join(path_to_frame, img_name)
            
                img_to_show = cv.imread(img_path)

                b_box = [ y_min,x_min , y_max, x_max]
                # cv.circle(img_to_show,(y_max,x_max),50,(0,0,255))
                # draw_image(img_to_show, b_box)
                y_ , x_ = y_max+y_min,x_max+x_min
                b_box = [ y_min,x_min , y_max, x_max]
                new_img = img_to_show[x_max:x_, y_max:y_,: ]
                #print("this is image", num_img)

                # new_img = img_to_show[x_min:x_max, y_min:y_max,: ]
                
               
                cv.imwrite(path_to_frame+'/patches/'+"patch"+str(num_img)+'.png', new_img)
                
                
                

def centralize(objects, dir, path_to_created_object):
    
    if not os.path.exists(path_to_created_object+str(dir)+'/'):
            os.makedirs(path_to_created_object+str(dir)+'/')
            
    for object in objects:
        path_to_object= path_to_created_object +'ressources_'+dir+'/'+object+'/'
        list_dir = os.listdir(path_to_object)

        # if not os.path.exists('../ressources_test1/'+object+'/centralize'):
        #     os.makedirs('../ressources_test1/'+object+'/centralize')
     
        for index_dir, dir_subject in enumerate(list_dir): 
          
            if(dir == 'centralize'):
                pass
            
            else:
                list_patches = os.listdir(path_to_object + dir_subject+'/patches/')
        
                for patch in list_patches:
                    # rename to keep track of the folder
                    patch2 = patch.split('.')[0]
                    patch2 = patch2 + "_"+str(index_dir) + ".png" 


                    if not os.path.exists(path_to_created_object+dir+'/'+object+'/'):
                        os.makedirs(path_to_created_object+dir+'/'+object+'/')
                    
                    shutil.copyfile( path_to_object+dir_subject+"/patches/"+patch, path_to_created_object+dir+'/'+object+'/'+patch2)

                    
           
            
                    
        
