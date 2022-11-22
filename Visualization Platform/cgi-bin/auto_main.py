# /home/ino/anaconda3/envs/TrackNet/bin/python3
import cgitb
import os
if not os.path.isdir('./log'):
    os.mkdir('./log')
cgitb.enable(display=0, logdir='./log')
import cgi
import uploadvideo
import TrackNetPredict
import segmentation
import raw2train as training_preprocess
import training
import predict
import coordinate as coordinate_adjust
import output
import videolist
import time

print("Content-Type: text/html\n\n")    # html type is following
form = cgi.FieldStorage()
print("<br>")
# print(form.keys())
uploadvideomode = form['uploadvideomode'].value
tracknetpredictmode = form['tracknetpredictmode'].value
segmentationmode = form['segmentationmode'].value
predictballtpyemode = form['predictballtpyemode'].value

###################### FILENAME ######################
input_video_name = ''
if uploadvideomode == 'on':
    input_video_name = form['videoname'].filename.split('.')[0]
else:
    print("Selected name : {}".format(form['videoname'].value))
    print("<br>")
    input_video_name = form['videoname'].value.split('.')[0]
ext = ".csv"
mp4_ext = '.mp4'

#TrackNet filename
TrackNet_input_path = './uploadvideo/'
TrackNet_label = 'Badminton_label_'
TrackNet_input = TrackNet_input_path + input_video_name + mp4_ext
TrackNet_output_path = './preprocessing/Data/TrainTest/'
TrackNet_output = TrackNet_output_path + TrackNet_label + input_video_name + ext

# segmentation filename(not used TrackNet output yet)
segmentation_input_path = TrackNet_output_path
segmentation_output_path = "./preprocessing/Data/AccuracyResult/"
segmentation_input = TrackNet_label
segmentation_output = "record_segmentation_"

segmentation_input = segmentation_input_path + segmentation_input + input_video_name + ext
segmentation_output = segmentation_output_path + segmentation_output + input_video_name + ext

# training data preprocessing input params
pre_dir = "./preprocessing/Data/training/data/"
raw_data = input_video_name

# has players' position info? 1/0 : yes/no
# if yes, player_pos_file (.csv) is needed
player_pos_option = 0
player_pos_file = ''
# training with specific frames? 1/0 : yes/no
# if yes, specific_frame_file (.csv) is needed
frame_option = 0
specific_frame_file = ''

# unique_id for get_velocity function
unique_id = ''

preprocessed_filename = pre_dir + raw_data + "_preprocessed" + ext
raw_data = pre_dir + raw_data + ext

if player_pos_option != 0:
    player_pos_file += ext
if frame_option != 0:
    specific_frame_file += ext

# training and predict input params
result_dir = "./preprocessing/Data/training/result/"
model_path = "./preprocessing/Data/training/model/model.joblib.dat"

#name_train = "video3_train"
name_result = input_video_name+"_predict_result"

#filename_train = pre_dir + name_train + ext
filename_result = result_dir + name_result + ext

# output json file
json__ext = ".json"
rally_count_json_filename = "rally_count_predict_" + input_video_name
rally_type_json_filename = "rally_type_predict_" + input_video_name
game_name_json_filename = "game_name"
output_json_dir = "./preprocessing/Data/Output/"

rally_count_json_filename = output_json_dir + rally_count_json_filename + json__ext
rally_type_json_filename = output_json_dir + rally_type_json_filename + json__ext
game_name_json_filename = output_json_dir + game_name_json_filename + json__ext

###################### FILENAME ######################

if __name__ == "__main__":
    if uploadvideomode == 'on':
        # Store video
        print("video type = ",form['videoname'].type)
        print('<br>')
        print("video size = ",len(form['videoname'].value))
        print('<br>')

        previous_time = time.time()
        uploadvideo.store(form['videoname'])
        end_time = time.time()

        print("<br>")
        print("Uploaded time : ",end_time - previous_time)
        print("<br>")

    if tracknetpredictmode == 'on':
        # TrackNet prediction(Local test can commit TrackNet to reduce runtime)
        previous_time = time.time()
        # TrackNetPredict.run(TrackNet_input, TrackNet_output)
        end_time = time.time()

        print("TrackNet time : ",end_time - previous_time)
        print("<br>")

    if segmentationmode == 'on':
        # Run segmentation
        previous_time = time.time()
        segmentation.run(segmentation_input, segmentation_output)
        end_time = time.time()

        print("Segmentation time : ",end_time - previous_time)
        print("<br>")

    if predictballtpyemode == 'on':
        # training and prediction
        previous_time = time.time()
        coordinate_adjust.run(segmentation_output, raw_data)
        training_preprocess.run(raw_data, preprocessed_filename, unique_id, player_pos_option, frame_option, player_pos_file, specific_frame_file)  #preprocess data
        #training.verify(pre_dir, filename_train, model_path)  #train model
        predict.verify(pre_dir, preprocessed_filename, model_path, result_dir, filename_result) #predict testing data
        end_time = time.time()

        print("Predict time : ",end_time - previous_time)
        print("<br>")

        # output json file
        output.run(raw_data, filename_result, rally_count_json_filename, rally_type_json_filename, game_name_json_filename, input_video_name)

    videolist.savelist2json()       