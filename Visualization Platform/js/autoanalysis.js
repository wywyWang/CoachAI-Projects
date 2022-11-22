function checkfile(sender) {
    // accepted file type
    var validExts = new Array(".mp4");
  
    var fileExt = sender.value;
    fileExt = fileExt.substring(fileExt.lastIndexOf('.'));
    if (validExts.indexOf(fileExt) < 0) {
      alert("File tpye is not acceptable,please upload correct file extension：" + validExts.toString());
      sender.value = null;
      return false;
    }
    else return true;
}

function show_file_select() {
    $('.modal').on('show.bs.modal', function (e) {
        // Remove previous result
        $(".file-size").html('');
        $(".upload-finish").html('');
        $('.progress-bar').css('width', '0%');
        $('.progress-bar').removeClass("progress-bar-success");
        $('.progress-bar').html('0%');
        $(".modal-footer").hide();

        var $trigger = $(e.relatedTarget)[0].id;
        $('.analysis-result').html('');         // Clear previous result
        $('.modal-body').append('<form id="autoanalysis_form" enctype="multipart/form-data" method="post"></form>')
        $('#autoanalysis_form').append('<select id="video_name" name="Video Name" class="form-control"></select>');
        filename = '/preprocessing/Data/Output/videolist.json';

        $.getJSON(filename, function(data) {
            if($trigger == "tracknet_btn") {
                data = data.previous_tracknet;
            }
            if($trigger == "segmentation_btn") {
                data = data.previous_segmentation;
            }
            if($trigger == "predict_ball_type_btn") {
                data = data.previous_predict_balltype;
            }
            if($trigger == "one_click_to_complete_btn") {
                data = data.previous_tracknet;
            }
            for(var index = 0;index < data.length;index++) {
                var insertText = '<option value=' + data[index] + '>' + data[index] + '</option>';
                $('#video_name').append(insertText); 
            }
        });
        
        $('#autoanalysis_form').append('<button type="submit" id="model_name" class="btn btn-primary" name=' + $trigger + '>Start analysis</button>');
     
        $('#autoanalysis_form').submit(function(e) {
            $(".modal-footer").show();
            e.preventDefault();
            var formData = new FormData();
            if(document.getElementById('model_name').name == 'tracknet_btn'){
                formData.append('uploadvideomode', 'off');
                formData.append('tracknetpredictmode', 'on');
                formData.append('segmentationmode', 'off');
                formData.append('predictballtpyemode', 'off');
                $(".modal-footer .waiting-content").html("Mode : TrackNet <br> Input : frames <br> Output : ball's visibility and position in each frame <br><br> Description : TrackNet is used to detect shuttlecock in each frames.");
            }
            if(document.getElementById('model_name').name == 'segmentation_btn'){
                formData.append('uploadvideomode', 'off');
                formData.append('tracknetpredictmode', 'off');
                formData.append('segmentationmode', 'on');
                formData.append('predictballtpyemode', 'off');
                $(".modal-footer .waiting-content").html("Mode : Segmentation <br> Input : ball's visibility and position <br> Output : sets of hitpoint's frame <br><br> Description : Segmentation is divided into two parts, first part is hitpoint detection, second part is classify loss reason for every hitpoint.");
            }
            if(document.getElementById('model_name').name == 'predict_ball_type_btn'){
                formData.append('uploadvideomode', 'off');
                formData.append('tracknetpredictmode', 'off');
                formData.append('segmentationmode', 'off');
                formData.append('predictballtpyemode', 'on');
                $(".modal-footer .waiting-content").html("Mode : Predict balltype <br> Input : skeleton in real-world court plane <br> Output : Balltype when hitpoint event <br><br> Description : Predict balltype uses projected skeleton coordinate as input features, through XGBoost classifier to classify which balltype player used when hitpoint.");
            }
            if(document.getElementById('model_name').name == 'one_click_to_complete_btn'){
                formData.append('uploadvideomode', 'off');
                formData.append('tracknetpredictmode', 'on');
                formData.append('segmentationmode', 'on');
                formData.append('predictballtpyemode', 'on');
                $(".modal-footer .waiting-content").html("Mode : One click to complete <br> Input video <br> Output : Balltype when hitpoint <br><br> Description : Only need to click this button, it will automatically run all modes.");
            }

            var dataFile = document.getElementById('video_name').value;
            formData.append('videoname', dataFile);
        
            for (var key of formData.entries()) {
                console.log(key[0] + ', ' + key[1]);
            }
            
            $.ajax({
                type: "POST",
                url: '/cgi-bin/auto_main.py',        
                data: formData,
                contentType: false,
                cache: false,
                processData: false,
                
                success: function(response)
                {
                    // console.log(response)
                },
                error: function(jqXHR, exception) {

                }
            }).done(function(data) {
                $('#file_select').hide(function(event){     
                    $(".modal-body form").remove();
                });
                $('#file_select').modal('toggle');
                console.log(data)
                $('.analysis-result').append(data);
            });
        });
    });                        　                 
}

$(function () {
    $('#submit-video').submit(function(e) {
        // avoid empty file upload
        if(document.getElementById('video-uploader').value.length == 0){
            alert("Please upload file.");
            return false;
        }
        e.preventDefault(); // avoid to execute the actual submit of the form.
    
        var formData_upload = new FormData();
        formData_upload.append('uploadvideomode', 'on');
        formData_upload.append('tracknetpredictmode', 'off');
        formData_upload.append('segmentationmode', 'off');
        formData_upload.append('predictballtpyemode', 'off');
        var dataFile = document.getElementById('video-uploader').files[0];
        formData_upload.append('videoname', dataFile);

        for (var key of formData_upload.entries()) {
            console.log(key[0] + ', ' + key[1]);
        }
    
        $('.file-size').html('File size : ' + parseInt(formData_upload.get('videoname')['size']/1024) + 'KB');
        
        $.ajax({
            type: "POST",
            url: '/cgi-bin/auto_main.py',
            data: formData_upload, 
            contentType: false,
            cache: false,
            processData: false,
            xhr: function() {
                var myXhr = $.ajaxSettings.xhr();
                if(myXhr.upload){
                    myXhr.upload.addEventListener('progress',updateProgress, false);
                    myXhr.upload.addEventListener("load", updateComplete);
                }
                return myXhr;
            },
            success: function(response)
            {
                // console.log(response)
            },
            error: function(error) {
                console.log('Error: ' + error);
            }
        }).done(function(data) {
            console.log(data)
            $('.file-size').append(data);
        });
    });

    $('.modal').on('hidden.bs.modal', function () {
        $(".modal-body form").remove();
    });

    function updateProgress(e){
        // console.log("total size",e.total)
        // console.log("current upload size",e.loaded)
        if(e.lengthComputable){
            var max = e.total;
            var current = e.loaded;
            var Percentage = parseInt((current * 100)/max);
            $('.progress-bar').css('width', Percentage + '%');
            $('.progress-bar').html(Percentage + '%');
        } 
        else{
            console.log("Unable to compute progress information since the total size is unknown")
        } 
     }

    function updateComplete(e) {
        $('.progress-bar').addClass("progress-bar-success");
        $('.upload-finish').html('Upload finished.')
    }
});