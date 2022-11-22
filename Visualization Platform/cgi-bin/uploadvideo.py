import os
import shutil

def store(videofile):
    if videofile.filename:
        base_filename = os.path.basename(videofile.filename)
        video_folder = './uploadvideo/'
        if not os.path.isdir(video_folder):
            os.mkdir(video_folder)

        video_path = video_folder + base_filename
        # with open(video_path, 'wb') as fout:
        #     fout.write(videofile.file.read())
        with open(video_path,'wb') as fout:
            shutil.copyfileobj(videofile.file, fout, 100000)
        print('The file "' + base_filename + '" was uploaded successfully')
        print('<br>')