from utils import *
import os
import shutil



# dir is train1, test1
def createPatchs(dir):
## create bounding boxes
    objects = ['Bowl', 'CanOfCocaCola', 'Jam', 'MilkBottle', 'Mug', 'OilBottle', 'Rice', 'Sugar', 'VinegarBottle' ]

    for object in objects:
        
        directory = '../GITW_light2_TD/'+dir+'/'+object+'/'
        path_to_created_object = '../Saillances/'
        path_to_frame = path_to_created_object+'ressources_'+dir+'/'+object+'/'
        
        if not os.path.exists(path_to_frame):
            os.makedirs(path_to_frame)
            
        list_dir = os.listdir(directory)
        # del list_dir[0]
        print(list_dir)

        for filename in list_dir:
            print(filename)
            if filename == ".DS_Store":
                continue
            
            # create dir for frames
            if  os.path.exists(path_to_frame + filename):
                shutil.rmtree(path_to_frame + filename)
            os.makedirs(path_to_frame + filename)
            
            # get videos
            path_to_video = directory + filename + '/' + filename +'.mp4'
            create_frame(path_to_video, path_to_frame, filename)

            path_to_saliency = directory + filename +'/'+'SaliencyMaps/'
            create_bounding_boxes(1, path_to_saliency, path_to_frame+filename)
            
            save_image(filename, path_to_frame + filename,  path_to_frame + str(filename)+ '/BB_file.txt')
            
    centralize(objects, dir, path_to_created_object)
        
        


createPatchs('test1')
createPatchs('train1')